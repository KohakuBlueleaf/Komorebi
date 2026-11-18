# Both data movers on one stream

A Tensix core has two data-movement CPUs: RISCV_0 (where Komorebi loads the reader
kernel, on NoC 0) and RISCV_1 (the writer kernel, NoC 1). Its programs give each a role: the reader reads
every input, the writer writes every output. This experiment asks whether one stream can
be shared by both instead: the stream is declared once, and each mover takes the blocks
that a host-chosen pattern gives it.

## The mechanism

| part | file |
|---|---|
| The pattern and the engine, first written here, now Komorebi's `km::Movement` ([reference](../../../docs/reference/movement.md)): a pattern word (its length, 1 to 24, and one bit per position, set when mover B owns that block), streams (direction, tensor, blocks, repetitions, buffer pair, pattern), run by one loop that issues every stream whose half has room or data, one barrier, then push or pop | `komorebi/device/movement/km_movement.hpp` |
| One data-movement source compiled for both movers, plus the classic layout for comparison | `kernels/movement.cpp` |
| Compute: takes block k from the half of its owner, so tile order is kept | `kernels/compute.cpp` |
| Driver: read, write and copy programs, exact against numpy before timing | `probe.py` |

A logical buffer is a pair of circular buffers, one per mover. Compute knows the pattern
and takes block k from mover A's half or mover B's half; a buffer has one filling CPU
and one draining CPU, as circular buffers need.

## Method

Card: one Blackhole p150a, slow dispatch, AICLK 1,350 MHz in every job (sampled during
each run), host load 5 to 11. Every case is checked first: reads by a sum of every word
compute took (and a sum weighted by position), writes by stamps (salt, mover, position,
word) read back, copies against the input. A rate counts only when the check passes.
Rate: R and 4R repetitions of 256 tiles a core, bytes x 3R / (cycles(4R) - cycles(R)),
cycles from the card's wall clock (first mover start to last end), best of three.

## Part A: the mechanism alone

Engine cost, one core (pattern A or B against the classic blocking calls), GB/s:

| | 8 tiles a block | 32 | 128 |
|---|---|---|---|
| read classic / engine A | 22.11 / 21.72 (-1.8%) | 45.97 / 45.13 (-1.8%) | 63.05 / 61.83 (-1.9%) |
| write classic / engine B | 36.73 / 35.10 (-4.4%) | 54.89 / 53.17 (-3.1%) | 62.63 / 60.97 (-2.7%) |

One core doubles: reads A / AB 21.72 / 41.60 (1.92x), 45.13 / 88.57 (1.96x),
61.83 / 122.44 (1.98x); writes B / AB 35.10 / 67.24, 53.17 / 104.94, 60.97 / 121.18.

Routes, 32 tiles a block, GB/s over all cores:

| reads | classic | A | B | AB | AAAB | AAAAABBB | BBBA |
|---|---|---|---|---|---|---|---|
| row of 8 | 319.2 | 297.6 | 95.5 | 190.6 | 381.1 | 254.0 | 127.1 |
| 120 cores | 422.4 | 422.9 | 245.9 | 477.8 | 498.9 | 495.5 | 328.4 |

| writes | classic | A | B | AB | AAAB | AAAAABBB | BBBA | BBBBBAAA |
|---|---|---|---|---|---|---|---|---|
| row of 8 | 300.5 | 86.5 | 300.8 | 172.5 | 115.2 | 138.3 | 300.1 | 224.1 |
| 120 cores | 313.1 | 178.2 | 312.6 | 354.8 | 237.3 | 284.7 | 413.4 | 460.0 |

On the landed header (2026-11-14, AICLK 1,350 MHz, host load 4.7), 120 cores: reads
423.3, 424.0, 247.4, 481.0, 500.9, 498.3, 330.2; writes 313.0, 178.2, 312.5, 354.9,
237.3, 284.8, 411.4, 458.9, all within 1% of the tables (row of 8: reads within 2%,
writes B +7%, BBBA -10%); `results/primitive-read.json`, `results/primitive-write.json`.

Copy (DRAM to DRAM through L1), GB/s copied, classic (reader reads, writer writes)
against halves (each mover reads and writes its own blocks, AB):

| | 1 core | 8 cores | 120 cores |
|---|---|---|---|
| 8 tiles a block | 20.9 / 32.95 (+58%) | 158.8 / 140.1 (-12%) | 224.2 / 214.3 (-4%) |
| 32 tiles a block | 45.5 / 53.2 (+17%) | 224.5 / 143.6 (-36%) | 225.6 / 214.4 (-5%) |

Compute takes blocks in order, so the faster mover waits for room behind the slower
one (AB reads on 120 cores: mover A busy 0.15 of the run, mover B 0.99). The pair runs at
the slower mover's pace, and the ratio has to follow the two routes' rates.

## Part B: the programs

Per program on main: which mover carries which stream, how busy it is (share of the
slowest core's time), and whether compute waits for it.

| program | streams and movers | busy | verdict |
|---|---|---|---|
| matrix product, resident (lin384: 16384 x 384 by 384 x 384, 120 cores) | A rows: reader, NoC 0; Y: writer, NoC 1 | reader reads 0.24, waits for room 0.64; writer writes 0.82; pack waits 0.78 | Y is mover-bound: split tried |
| matrix product, K split (dec4096: 32 x 4096 by 4096 x 4096, 112 cores, tree reduce) | A and W chunks: reader, NoC 0; partial sums: writer | parts only 92.5 of 103.4 us; the chunk reads set the pace | chunk reads split |
| attention (long64, 32 heads x 4096 x 64) | Q: reader; K/V broadcast between cores; O: writer | Q reads 0.007, writes 0.005; compute waits for K/V 0.01 | compute-bound: no variant |
| rasterizer frame (s10k) | lists: reader; tiles: writer | reader reads 0.06, writer writes 0.31; compute busy 0.96, waits for input 0.03 | compute-bound: no variant |
| ray tracer frame (tiny, 8 commands) | packets: reader; packets out: writer | the reader reads 0.02 to 0.17 of most steps, 0.55 of shade2 and 0.85 of image; compute waits for input 0.3 to 15% | compute-bound: the two read-heavy steps are 9.6% of the frame; at most 0.7% to gain, not tried |

The two variants, in copies of the program's kernels under `dense/` (the program's own
files were unchanged then). Microseconds per call in a loop of about 2 s (the clock
during the loop was not read), Y exact against numpy in every run (sampled rows, rms
3.96e-3, the same as main's):

Resident form, lin384, Y's column blocks split (`dense/kernels/movement.cpp` through the
engine, A's rows on mover A):

| main | B (engine) | BBBA | BBBBBAAA | AB | BBBBBBBA | 15 B : 1 A |
|---|---|---|---|---|---|---|
| 60.62 | 60.38 | 64.16 | 63.71 | 86.55 | 59.31 (-2.1%) | 60.40 |

Mover A then carries both A's rows and its Y blocks on NoC 0 (BBBA: busy 0.90), so the
writes it takes slow the reads compute needs: at most 2%.

K split, dec4096 (14 x 1 x 16 plan, 5 tiles a chunk, tree reduce), the A and W chunks
split (`dense/kernels/split/`; main's buffer of two chunks becomes two halves of one
chunk each, the same L1):

| | main | AB | AAAB | ABBB |
|---|---|---|---|---|
| run 1 | 103.35 | 92.49 (-10.5%) | | |
| run 2 | 103.52 | 92.29 (-10.8%) | 103.51 | 92.42 (-10.7%) |
| parts only (run 2) | 92.53 | 81.62 | 93.02 | 81.67 |

AAAB gains nothing: mover A then owns three chunks in a row with one chunk of buffer, so
only one read is in flight, as in main. AB and ABBB keep two chunk reads in flight on
the two NoCs at once.

The program's K split now reads its chunks this way, through `km::move` under `AB`
(`projects/dense/kernels/split/chunks.hpp`), so `main` in `dense/bench.py` now runs that
form; the copy under `dense/kernels/split/` stays as the record of the variant.

## What follows for a `km::Movement`

- A half's filling and draining CPUs are fixed for the life of a load. The packer keeps
  its own count of pages pushed and writes it whole into the buffer's received word, so a
  buffer filled by compute in one command and by a mover in the next gets its count
  rewound (a first copy run here read 119 wrong pages before this was found).
- The ratio follows the routes, never halves by default: on a row of 8, AB reads run at
  0.6 of classic and AAAB at 1.19; on the full grid AAAB (reads) and BBBBBAAA (writes)
  are best.
- Compute pairs blocks in order, so the slower mover sets the pace.
- A shared stream gains when one mover's stream is the bottleneck and the other NoC is
  free (the K split: -10.5%); it loses when the other mover already carries a stream on
  that NoC (the resident Y split).
- Streams need two-dimensional strided blocks and ragged ends: the K split's chunks are
  strided sub-blocks of A and W, so its variant splits them by hand rather than through
  the engine.
- The engine loop costs 2 to 4% against blocking calls on one core.
- Copies through halves pay only on one core.

## Running it

From the repository root, on a card with slow dispatch and an operation timeout set:

    python experiments/core/movement_probe/probe.py read --cores 1 --blocks 8 32 128 \
        --layouts classic A AB
    python experiments/core/movement_probe/probe.py write --cores 8 0 --blocks 32 \
        --layouts classic B AB BBBA BBBBBAAA
    python experiments/core/movement_probe/probe.py copy --cores 1 8 0 --blocks 8 32 \
        --layouts classic AB
    python experiments/core/movement_probe/dense/bench.py --shape lin384 main B BBBBBBBA
    python experiments/core/movement_probe/dense/bench.py --shape dec4096 \
        --split 14,1,16,5 --reduce tree main AB ABBB

`--cores 0` is the whole grid. Each case prints progress lines and one JSON line.
