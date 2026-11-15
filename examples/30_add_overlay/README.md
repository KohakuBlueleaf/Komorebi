# 30 add overlay

Code added to a run that is loaded and live
([code added to a live run](../../docs/reference/overlays.md#code-added-to-a-live-run)).
The load has one handler, `RUN`, and slots for code that comes later. While the run
goes on, `run.addon` adds:

- `mix`, an overlay with a reader part. `RUN` names it by the number in its word `a0`,
  loads it and calls it, as a kernel names any overlay.
- `scale`, a handler whose body is an overlay with reader, writer and compute parts.
  It is submitted as any handler.
- `mix again`, the same source as `mix`: its parts come from the cache of built parts.

Commands run on every core before, between and after the adds, from the host and from
an X280 thread (`thread/relay.c` submits the command it is given to the cores). Every
word is checked against numpy: `RUN`'s result line on each core (the mix loop on the
core's seed, or the seed when it names no overlay) and `SCALE`'s output tiles (each
input tile times the command's scalar, bfloat16).

| Handler | Words | Does |
|---|---|---|
| `RUN` (1, loaded) | a0 the overlay (0: none), a1 the seed, a2 the lines tensor, a3 the loop's steps | calls overlay a0 on `seed ^ core`, writes [sequence, result, load cycles, call cycles, held, a0] |
| `SCALE` (added) | a3 tiles a core, a4 the scalar's float32 bits, a6 the tiles tensor | the reader fills the input tiles, compute multiplies them, the writer writes them out |

```{ .python }
--8<-- "examples/30_add_overlay/add_overlay.py:program"
```

```{ .python }
--8<-- "examples/30_add_overlay/add_overlay.py:add"
```

## Measured

Card 0, 2026-11-16, AICLK 1,350 MHz before and after, 120 cores, 8 commands a phase
(`python add_overlay.py`). All 56 commands were exact against numpy
([results/add_overlay.json](results/add_overlay.json)). Command times run from the
dispatcher's submit to the completion: for the thread's commands, the thread's own
clock around its submit and wait.

| Phase | First command | Later (median) |
|---|---|---|
| `RUN` naming no overlay, before the adds | 2.88 µs | 2.24 µs |
| `RUN` naming mix (every core loads it, then holds it) | 3.78 µs | 2.76 µs |
| `SCALE`, the added handler (every core loads five parts) | 38.32 µs | 35.96 µs |
| `RUN` naming mix again (another number, the same bytes) | 3.78 µs | 2.78 µs |
| `RUN` naming no overlay, after | 2.24 µs | 2.24 µs |
| `RUN` naming mix, from the thread | 3.52 µs | 2.72 µs |
| `SCALE`, from the thread | 36.76 µs | 35.90 µs |

Mix's `load()` took 1,278 cycles on a core that loads it and 112 on one that holds it.
Its call took 563 and 542 cycles.

| Add | Build | Write of the parts | Entry commands |
|---|---|---|---|
| mix (128 B reader part), first in a fresh cache | 0.84 s | | |
| scale (1,088, 1,408, 1,728, 1,088, 1,728 B), first in a fresh cache | 5.10 s | | |
| mix, cached | 14.4 ms | 0.4 ms | 0.2 ms |
| scale, cached | 42.3 ms | 0.4 ms | 0.1 ms |
| mix again, cached | 8.8 ms | 0.2 ms | 0.1 ms |

The first adds (the parts compiled and checked) were timed off the card, with the same
compiler commands as the job. On the card every add found its parts in the cache that
the earlier build had filled: a second add, in any later session, builds nothing. The
load's kernel text was 28,688 B of 70,656. The slots were four reader slots of
9,088 B, one writer slot of 2,048 B and one compute slot of 4,096 B on each compute
core.

## Files

| File | What it is |
|---|---|
| `add_overlay.py` | the load, the adds, the commands and their checks; one JSON line (`--json` also writes it) |
| `kernels/add.h` | the handler, the buffers and the words the kernels and Python share |
| `kernels/reader.cpp`, `writer.cpp`, `compute.cpp` | the loaded kernels: `RUN` on the reader |
| `kernels/mix/reader.cpp` | the added overlay |
| `kernels/scale/` | the added handler's three parts |
| `thread/relay.c` | the X280 thread: `RELAY` submits a command to cores 0 to n - 1 and returns its ticks |
