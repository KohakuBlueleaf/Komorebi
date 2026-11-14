# Tile front: the emitted SiLU body against `silu_tile`

Question: does the compute body that `komorebi.dsl.tile` emits for `y = x * sigmoid(x)`
match the hand body it stands in for, instruction for instruction and bit for bit, and
what does a tile cost on one core with each?

The hand body is tt-metal's `silu_tile(dst)` after `silu_tile_init()`, with fp32 `Dst`:
the call the dense project's SwiGLU layer makes (`projects/dense/kernels/compute.cpp`).
The SFPI compiler builds it into raw Tensix instructions, so both bodies end up as one
stream of coprocessor words on the math CPU, which this experiment compares.

## How it runs

| part | file |
|---|---|
| reader (RISCV_1): tile t of the input into CB 0 | `kernels/reader.cpp` |
| compute: copy into `Dst`, the body (define `BODY`: 0 hand, 1 emitted), pack into CB 16 | `kernels/compute.cpp` |
| writer (RISCV_0): the output tiles, then the info page (CB 24) | `kernels/writer.cpp` |
| shared layout of the info page | `kernels/front.hpp` |
| driver: emits `kernels/generated/silu.hpp`, builds, compares, runs | `front.py` |

```
python front.py build                          # mock device: both bodies built and compared
python front.py card [--tiles 16] [--seed 1]   # one core: bits and cycles of both bodies
```

- **Off-card comparison.** `build` opens a mock Blackhole device, lets the runtime's JIT
  build both programs, reads the math CPU's ELF of each, and decodes its Tensix words.
  A `REPLAY` that plays is expanded into the words it replays, so the comparison sees the
  stream the coprocessor runs. "Modulo registers" compares each instruction's name, its
  other fields, and for each register it reads, how many instructions back that register
  was written: the same data flow under a different register assignment.
- **Timing.** The math CPU reads the wall clock (AICLK cycles) around the body, between
  two `tensix_sync()` calls, so a span includes one synchronisation (the empty span,
  measured 16 times, is reported beside it). The pack CPU times the whole tile.
- **Values.** 16 tiles of fp32 (`--seed 1`), the same for both bodies: normal values
  scaled by 8, one tile over magnitudes 2^-40 .. 2^40 with random signs, one tile of
  edges (zeros, where e^-x overflows and underflows, the largest finite values) and a
  ramp from -100 to 100.

## Results

Header `7c3f9dca470d`, mock build and card run on the same sources.

Instructions (mock Blackhole p150):

| | hand | emitted |
|---|---|---|
| Tensix words stored | 241 | 240 |
| words run (replays expanded) | 465 | 464 |
| words of the tile body (first face on) | 430 | 430 |
| tile body | | equal modulo registers, not exact |

The differences are in the setup only:
- the hand init resets the read/write counters three times (`SETRWC` mask 15), the
  emitted one once;
- the emitted init has one `SFPNOP` after the `SFPCONFIG` that sets `LReg12` to 2.0 (the
  unit program's config-nop rule; the compiler does not insert it).

Both programs take 7,952 of the 70,656 B kernel buffer (math CPU text: 2,316 B hand,
2,312 B emitted).

On the card (Blackhole p150a, one core, slow dispatch, AICLK 1350 MHz during both runs,
load 2.5, 16 tiles):

| | hand | emitted |
|---|---|---|
| output | | bit for bit equal to the hand body (0 of 16,384 words differ) |
| body cycles a tile, median (min .. max) | 1730 (1730 .. 1733) | 1730 (1730 .. 1733) |
| empty span | 10 | 10 |
| whole tile (copy, body, pack), median | 1866 | 1866 |

The unit program's estimate for the body is 1,738 cycles a tile (an instruction a cycle,
a MAD result after two). Against float64 `x / (1 + exp(-x))` the output's relative error
is at most 1.8e-7.
