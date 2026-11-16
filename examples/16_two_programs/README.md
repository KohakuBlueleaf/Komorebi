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
cores that hold both share one buffer of each. A third program, `widen`, has other buffers
and other data formats: bfloat16 tiles in at buffer 1 (4 pages), float32 tiles out at
buffer 16 (2 pages of 4,096 B), and a third buffer, 24.

| Program | Handler | Arguments | What a core does |
|---|---|---|---|
| sum | `SUM` (1) | a0 the input's DRAM address, a1 tiles (16), a2 the sums' DRAM address, a3 the first output page, a6 the first tile | adds its share of the tiles into one tile and writes it to page a3 + its rank |
| sum, scale | `EMPTY` (2) | none | nothing: the command costs the load and the call |
| scale | `SCALE` (1) | a0 the input's DRAM address, a1 tiles (16), a2 the output's DRAM address, a3 the first output page, a6 the first tile | doubles its share of the tiles and writes each to page a3 + its place |
| widen | `WIDEN` (1) | as `SCALE` | writes its share of the tiles as float32 to page a3 + its place |

The script runs seven phases:

- disjoint: `session.load({sum: (0, 88), scale: (88, 32)})`. Commands to the two programs
  alternate, each to a random group inside its program's cores. The host must refuse a
  command of `sum` to the cores of `scale`.
- combined: `session.load([sum, scale])`, every core holds both. Commands alternate, each to
  a random group of any cores.
- price: the same sum in the kernel form (01_mapcollect's kernels) and in the runtime's form,
  one command at a time, an empty command and a sum of 16 tiles, on 1 core and on every core.
- switch: combined with one slot a kernel, so a command to the other program loads its five
  parts; the same program again is held.
- live: `session.load({sum: ..., scale: ..., widen: ...})`, and half of sum's cores (the
  part) switch program while the run is live: `run.load(scale or widen, group=part)`, a
  command for it to the part, commands for every program elsewhere, a SUM to the part
  sent without the host's check (the dispatcher must refuse it, reason 13), then
  `run.load(sum, group=part)` and a SUM to all of sum's cores; `--switches` loads.
- load: the cycles of a load command, to another program and to the one held, on 1 core
  and on every core.
- merged: `session.load(km.merge(sum, scale))`, ONE program
  ([merged](../../docs/reference/host.md#programs-merged-into-one)): sum's handlers keep
  1 and 2, scale's become 3 and 4. Commands alternate on random groups (SUM by
  `merged.handler("sum")`, SCALE by scale's own handle, mapped); then the cycles a
  command of sum alone, of the two as a list and merged. `--overlay` places scale's
  code as an overlay with the merge's marker.

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
| `kernels/read_tiles.h` | the readers' loop: a core's share of the tiles into a buffer (0, or widen's 1) |
| `kernels/sum/reader.cpp`, `compute.cpp`, `writer.cpp` | program sum, in the runtime's form |
| `kernels/scale/reader.cpp`, `compute.cpp`, `writer.cpp` | program scale, in the runtime's form |
| `kernels/widen/reader.cpp`, `compute.cpp`, `writer.cpp` | program widen: other buffers, float32 out |
| `two_programs.py` | the six phases; checks every result, prints the medians and one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python two_programs.py                           # every phase, every core of the grid
python two_programs.py --phase combined --cores 10 --rounds 10
python two_programs.py --cards 0,1                # every phase on two cards
```

`--cores 0` (the default) is every core of the grid. The script exits non-zero if a result
differs, a command was given up (`run.drain_given_up`) or the host does not refuse. On two cards the groups are flat ranks over both
islands, and a phase that reads a trace reads each command's rows over both cards
(`run.trace().command(ticket)`, each island's in its own clock).

## Numbers

A Blackhole p150a, AICLK 1,350 MHz before and after the run, `python two_programs.py`
(120 cores, 40 rounds a phase). Disjoint and combined: 80 commands each, every result
exact; the host refused the command to the wrong cores. Live: 200 switches, 600 commands,
every result exact, the 100 SUMs sent to the part while it held another program refused by
the dispatcher with reason 13. The parts of sum are 1,024 to 1,984 B. Cycles from the
reader taking a command to its completion, median core, median over commands:

| | 1 core | 120 cores |
|---|---|---|
| empty command, the kernel form | 594 | 806 |
| empty command, runtime's form | 707 | 918 |
| sum of 16 tiles, the kernel form | 5,233 | 1,723 |
| sum of 16 tiles, runtime's form | 5,593 | 1,874 |
| empty command, the program held (one slot) | 710 | 919 |
| empty command, the other program loaded (five parts) | 2,818 | 3,953 |
| load command to another program, its parts held | 3,124 | 3,360 |
| load command to the program held | 3,249 | 3,475 |

What the numbers say:

- The runtime's form costs about 113 cycles a command: the held `load()` (72 cycles) and
  the call.
- On one core the sum of 16 tiles costs 360 cycles more in the runtime's form (1,038 when
  the program's slots lay above 1 MB of L1, where code runs slower).
- A command that names the other program loads its five parts: 2,818 cycles on one core,
  3,953 on 120 at once. [overlay_costs](../../experiments/code/overlay_costs/README.md) measured
  0.79 µs for a held program and 7 µs for a change of program on 120 cores.
- A load command (`run.load`) costs about 3,100 to 3,500 cycles (2.3 to 2.6 µs) on one
  core or on 120, whether or not the program changes: the five cores meet, wait for their
  buffers to be empty and set them up again; the parts stay in their slots.

### With and without the guard

The table above is from a head before the guard: a command given up at its deadline with
the run going on, and `run.inspect` of a running core. Both are on by default
(`guard=True`) and cost every command cycles and kernel text; `km.Program(guard=False)`
builds the kernels without them ([the hazard](../../docs/reference/hazards.md#a-wait-that-never-completes)).
`python two_programs.py --phase price` and the same with `--no-guard`, 120 cores, 40
rounds, AICLK 1,350 MHz, 2026-11-16, cycles a command (1 core / 120 cores):

| | guard on (default) | guard off |
|---|---:|---:|
| empty command, the kernel form | 1,012.5 / 1,149.5 | 639 / 848 |
| empty command, runtime's form | 1,067.5 / 1,228 | 726.5 / 942 |
| sum of 16 tiles, the kernel form | 5,641.5 / 1,989 | 5,242 / 1,736 |
| sum of 16 tiles, runtime's form | 6,477 / 2,158.5 | 5,586 / 1,886 |
| kernel text, the kernel form | 34,848 B | 25,328 B |
| kernel text, runtime's form (sum) | 30,432 B | 24,032 B |

Off, an empty command costs about 340 to 370 cycles less on one core and 290 to 300 less
on 120, a sum of 16 tiles 250 to 890 less, and the kernels 6.4 to 9.5 KB less text. What it takes away: a
command that never completes is not given up at its deadline (the dispatcher frees it
with its cores still in it), `run.inspect` is refused, and a hang costs a card reset; the
load's log says so once.

### Merged into one program

`python two_programs.py --phase merged` and `--phase merged --overlay`, 120 cores, 40
rounds, AICLK 1,350 MHz before and after, 2026-11-16: 80 commands each, every result
exact. Cycles a command as above (1 core / 120 cores):

| | sum alone | as a list | merged | merged, scale an overlay |
|---|---:|---:|---:|---:|
| empty command (sum's) | 1,051 / 1,216 | 1,057 / 1,206 | 1,092 / 1,235 | 1,093 / 1,239 |
| sum of 16 tiles | 6,451 / 2,136 | 6,461 / 2,118 | 5,831 / 2,175 | 5,847 / 2,158 |
| empty command (scale's, its number mapped) | | | 1,156 / 1,307 | 1,266 / 1,428 |

On this head one program's empty command costs 1,051 cycles, not the 707 of the table
above, which an earlier head measured. Merged adds about 40 cycles (the table and the
call of the source), a command of scale about 65 more (its command copied with scale's
own number), and with scale an overlay about 110 more again (the overlay's check and
call). The merged parts (reader 3,264 B, writer 2,816, unpack 4,544, math 2,432, pack
3,328) are larger than sum's and scale's together; with scale an overlay the part's
reader grows to 5,440 B (the overlay's load is compiled into it).
