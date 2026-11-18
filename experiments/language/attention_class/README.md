# Attention, as a class

Question: can the attention project's program
([projects/attention](../../../projects/attention/README.md)) be written as a class
([Programs written as classes](../../../docs/reference/classes.md)), with every move a
class line and every compute step the hand program's? And does it give the hand's O,
word for word, in the hand's time?

It can. `attention.py` makes the class `Attention`, one handler `attend` with the hand
program's ATTEND words in the hand's order (a0 q .. a8 job_count). The script runs the
hand program (`card.Attention`) and then the class in one session, on the same Q, K, V
and job tables, and compares the two O tensors. The compute steps are in `flash.hpp`:
the hand's compute (`kernels/compute.cpp`, the body of Kohaku-TT's `ktt_flash`, with
`sfpu.hpp` included as it is), as a template on the class's buffers.

```python
for _ in range(max(passes, 1)):
    for job in jobs[first_job:end]:
        r0 = job.q // DT
        qs = km.Stream(
            q.tiles[
                r0 : min(r0 + R * job.stride, r0 + job.rows * job.stride) : job.stride,
                0:DT,
            ],
            block=QB,
        )
        k0 = job.k // DT
        v0 = job.v // DV
        for b in range((job.keys + KB - 1) // KB):
            kr = k0 + b * KB
            vr = v0 + b * KB
            if down.sender():
                kv = down.cast(
                    k.tiles[kr : min(kr + KB, k0 + job.keys), 0:DT].T,
                    v.tiles[vr : min(vr + KB, v0 + job.keys), 0:DV],
                )
            else:
                kv = down.arrive()
            km.cxx(
                "...Flash<kv_buffer, ...>::block(job_rows, ...)...",
                takes=[qs, kv, job, b, ...],
            )
        o0 = job.o // DV
        out = km.Stream(
            o.tiles[
                o0 : min(o0 + R * job.stride, o0 + job.rows * job.stride) : job.stride,
                0:DV,
            ],
            write=True,
            block=DV,
        )
        for row in range(job.rows):
            km.cxx("...Flash<...>::finalize(row)...", takes=[out, row, ...])
```

## Each thing the hand program does, and its line

The rows are the ones this page listed before the class could be written: what the
hand program does that no line said.

| # | What the hand program does | The line |
|---|---|---|
| 1 | Compute's own buffers: P (pack to unpack), the two tokens, the row state and O written in place, the ones and causal tiles | `p: km.L1[km.bf16, 2 * SUB, 32, 32] = km.own(push="pack")`, `state: km.L1[km.fp32, 2 * R, 32, 32] = km.own(push=None)` and the others |
| 2 | The float32 buffers unpacked to DST as float32 | a float32 own field (`state`, `acc`) |
| 3 | The profile page of five CPUs' words | not in this class (below) |
| 4 | `FLASH_CAUSAL` and the sizes the unchanged steps test | `class Attention(km.Program, defines={...})`; `Attention(..., defines={"FLASH_CAUSAL": 1, "FLASH_SCALE_BITS": ...})` at the make |
| 5 | The job table, 8 words a job from word 8, the count at word 0, on all CPUs | `jobs = tables.pages(self.rank, 1).entries(Job, first=8, size=8, count=0)`, `for job in jobs[first_job:end]:` |
| 6 | Each finished row's DV tiles packed and written at O's row `first + row x stride` | a write stream of O's stepped rectangle, `block=DV`, that the row loop's body fills (`out_buffer`) |
| 7 | Q's `rows` rows, `stride` apart, one block of R x DT pages, partly read, held through the job's K / V blocks | the stepped rectangle `q.tiles[r0:min(...):job.stride, 0:DT]` and the held stream `qs = km.Stream(..., block=QB)`: moved at its line, waited for there, freed at the end of the job's lines; a job of no rows pushes its block empty, as the hand reader does |
| 8 | A block of K transposed (`[DT][KB]`) and then V (`[KB][DV]`), read by the sender, then cast | `down.cast(k.tiles[...].T, v.tiles[...])` |
| 9 | Bands of L rows down each column, on the reader's NoC 0 | `km.Broadcast(self, along="column", length=LENGTH, mover="A", window=WINDOW)` |
| 10 | The blocks counted over passes x jobs x the job's blocks, the receivers' padding to the window after the last | the cast inside the three loops: `km_blocks_down`, and `km::broadcast_end` with the count |
| 11 | Compute takes each block whole and indexes K's and V's pages inside it | `takes=[kv]`: `kv_buffer`, the block's first page `kv` (0), freed after the body |
| 12 | Compute's start-up, `flash::init`, the constant tiles, the state reserved once; `restore_constants` at the end | `startup=km.Startup("ones", "causal", "p", order="reverse")`, `km.cxx(..., at="start")`, `km.cxx(..., at="end", on="pack")` |
| 13 | Compute only: each block pushed unread, then padded to the window | `unread=mode // 2` (mode 2) |

## Results

Card 0, a Blackhole p150a, 120 cores; the plan's R 13, KB 12, window 2, L 10 for both
shapes. Q, K and V are the project run's (`sdpa.inputs`, seed 1); 20 commands each,
back to back, the median.

| | t1 (1 x 1 x 128 x 64) | causal64 (1 x 8 x 1,024 x 64, causal) |
|---|---|---|
| O: words that differ from the hand's | 0 of 8,192 | 0 of 524,288 |
| time a command: hand, class | 13.33 us, 13.42 us | 57.64 us, 57.75 us |
| kernel text, hand (mover A, mover B, unpack, math, pack) | 23,440, 8,648, 14,108, 7,796, 9,092 (63,084 B) | 23,440, 8,648, 14,388, 8,436, 9,340 (64,252 B) |
| kernel text, class | 25,836, 10,236, 16,376, 7,340, 8,288 (68,076 B) | 25,836, 10,236, 14,928, 7,988, 8,592 (67,580 B) |
| L1 circular buffers, hand, class | 551,360 B, 653,440 B | 551,360 B, 653,440 B |

The class's L1 is larger by Q's pair (two halves of 26 pages, mover B's unused), the
tokens (pages of 2,048 B where the hand's are 64 B), and the write pair of O (two halves
of two blocks). A first run on causal64 with Q held in a read stream's pair of four
blocks a half took 972,928 B in the same 57.7 us; the held stream's pair is one block
a half now.

The first run on t1 hung: the plan gives most cores of a band a job of no rows, the
held stream of no pages moved no block, and compute waited for one. A held stream of
no pages now pushes one empty block, as the hand reader pushes such a job's Q block.

### 32 calls in one chain command

`attention_chain.py`: causal64 as 32 calls of `attend` in one chain command, call c
with its own Q, K and V (`sdpa.inputs`, seed c + 1) and its own O, the chain command
run three times back to back. O is written at the job table's words, so the chain
takes it as a write at a table's words: no call of the chain reads it. Card 0,
2026-11-18 22:58, AICLK 1,350 and 1,343 MHz at the two reads, host load average 59.6
(1 min); every row of the 32 O against float64 at the project's bounds: relative RMS
5.16e-3, largest 7.15e-3.

| Form | Single call, on the card | 32 as one chain command, wall (on the card) |
|---|---:|---:|
| the class, causal64 | 58.7 | 58.4 (53.0) |

The first chain of two calls hung: a receiver's next record built its broadcast while
its compute still held the last record's padding (its blocks no multiple of the window of 2),
and its wait for an empty buffer sent the sender a DONE marker counted from that
padding. A receiver now sends no marker before its broadcast has begun.

```
python attention_chain.py
```

```bash
python attention.py --shape t1        # the hand, then the class: O compared, times, text
python attention.py --shape causal64
```

## What differs from the hand program

- `flash.hpp` is the hand's compute (`score`, `pv_columns`, `output`, `finalize`,
  `constant_tiles`, the vector and pack-format helpers, and the block loop of `job`):
  each arithmetic line is the hand's, its tokens unchanged (the formatter rewraps a few
  at the deeper indent of a template). The buffer indices are template arguments (the
  bodies' names of the class's buffers, constants) where the hand has `constexpr`
  indices; Q's and the output's are set at run time (a stream's half). With all of
  them set at run time, unpack's text was 30,360 B and the class did not fit (86,512 B
  of 70,656). The profile counters and their `km::wall()` reads are gone.
- `finalize` is a variant: the line reserves and pushes the output row, so the hand's
  `const bool room = compute.reserve_back(cb_output, VALUE_TILES);` and
  `if (room) { cb_push_back(cb_output, VALUE_TILES); }` are removed (and with them the
  `room &&` in the pack loop's condition). Its other lines are the hand's;
  `cb_keys_values` there is the ones buffer, where no K / V block is held: both are
  bf16, so SrcA's restore is the same.
- The start-up is `km.Startup("ones", "causal", "p", order="reverse")`: own fields of
  bf16 tiles, as the hand's Q, K / V and P are, so the unpacker's and the packer's
  configuration is the hand's.
- Mode 1 (broadcast only) and the rest of mode 2 (no Q reads, no O writes) are not
  lines here: only the unread blocks of mode 2 are. The class was run in mode 0 only.
- No `prof` handler: a word of unpack or math into pack's page is still not a line
  (row 3).
- `BROADCAST_ON_WRITER` would be `mover="B"` (NoC 1); `FLASH_APPROX` a define at the
  make. Neither was run.
