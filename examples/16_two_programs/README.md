# 16 two programs

Two programs in one run, each in the **runtime's form**: every kernel source defines
`km_program(kernel, command)` instead of `kernel_main`. Komorebi builds each program as an
overlay. Its own kernels run on every core, and for each command they load the program the
command names and call it ([A program as an overlay](../../docs/reference/overlays.md#a-program-as-an-overlay)).

- `sum`: each core of the group reads its share of 16 tiles, adds them into one tile and
  writes it to DRAM (the sum of [01_mapcollect](../01_mapcollect/README.md)).
- `scale`: each core of the group reads its share of 16 tiles, doubles them and writes them
  back to DRAM.

Both number their handlers from 1, and both declare buffers 0 and 16 in the same way, so
cores that hold both share one buffer of each.

| Program | Handler | Arguments | What a core does |
|---|---|---|---|
| sum | `SUM` (1) | a0 the input's DRAM address, a1 tiles (16), a2 the sums' DRAM address, a3 the first output page, a6 the first tile | adds its share of the tiles into one tile and writes it to page a3 + its rank |
| sum, scale | `EMPTY` (2) | none | nothing: the command costs the load and the call |
| scale | `SCALE` (1) | a0 the input's DRAM address, a1 tiles (16), a2 the output's DRAM address, a6 the first tile | doubles its share of the tiles and writes each to the same page of the output |

The script runs four phases:

- disjoint: `session.load({sum: (0, 88), scale: (88, 32)})`. Commands to the two programs
  alternate, each to a random group inside its program's cores. The host must refuse a
  command of `sum` to the cores of `scale`.
- combined: `session.load([sum, scale])`, every core holds both. Commands alternate, each to
  a random group of any cores.
- price: the same sum in today's form (01_mapcollect's kernels) and in the runtime's form,
  one command at a time, an empty command and a sum of 16 tiles, on 1 core and on every core.
- switch: combined with one slot a kernel, so a command to the other program loads its five
  parts; the same program again is held.

The host checks every value of the sums and the doubled tiles.

## The programs, from the host

```python
sum_program = km.Program("sum", kernels=KERNELS / "sum", buffers=buffers, ...)
scale_program = km.Program("scale", kernels=KERNELS / "scale", buffers=buffers, ...)
run = session.load([sum_program, scale_program])  # or {sum_program: (0, 88), ...}
run.submit(sum_program.handler("sum"), group, words)  # handler 1 of sum
run.submit(scale_program.handler("scale"), group, words)  # handler 1 of scale
```

```c++
// sum/writer.cpp: called for each command of sum; true when the command is complete
#include "komorebi/device/km_program.hpp"

bool km_program(km::Writer& writer, const km_command_t& command) { ... }
```

## Files

| File | What it is |
|---|---|
| `kernels/two_programs.h` | the handlers' numbers, the buffers' indices and the tile size |
| `kernels/read_tiles.h` | the readers' loop: a core's share of the tiles into buffer 0 |
| `kernels/sum/reader.cpp`, `compute.cpp`, `writer.cpp` | program sum, in the runtime's form |
| `kernels/scale/reader.cpp`, `compute.cpp`, `writer.cpp` | program scale, in the runtime's form |
| `two_programs.py` | the four phases; checks every result, prints the medians and one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python two_programs.py                           # every phase, every core of the grid
python two_programs.py --phase combined --cores 10 --rounds 10
```

`--cores 0` (the default) is every core of the grid. The script exits non-zero if a result
differs or the host does not refuse.

## Numbers

A Blackhole p150a, AICLK 1,350 MHz before and after the run, `python two_programs.py`
(120 cores, 40 rounds a phase). Disjoint and combined: 80 commands each, every result
exact; the host refused the command to the wrong cores. The parts of sum are 768 to 1,920 B.
Cycles from the reader taking a command to its completion, median core, median over
commands:

| | 1 core | 120 cores |
|---|---|---|
| empty command, today's form | 596 | 811 |
| empty command, runtime's form | 724 | 941 |
| sum of 16 tiles, today's form | 5,233 | 1,711 |
| sum of 16 tiles, runtime's form | 6,274 | 1,892 |
| empty command, the program held (one slot) | 728 | 936 |
| empty command, the other program loaded (five parts) | 2,919 | 3,971 |

What the numbers say:

- The runtime's form costs about 130 cycles a command: the held `load()` (72 cycles) and
  the call.
- On one core the sum of 16 tiles costs 1,041 cycles more in the runtime's form. Where they
  go was not measured. A program is built as position-independent code, and its calls into
  the runtime go through a table, which is one candidate.
- A command that names the other program loads its five parts: 2,919 cycles on one core,
  3,971 on 120 at once. [overlay_costs](../../experiments/overlay_costs/README.md) measured
  0.79 µs for a held program and 7 µs for a change of program on 120 cores.
