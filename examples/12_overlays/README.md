# 12 overlays

Two **overlays**, kernel code that is not in the kernels, loaded into a slot of every
core's L1 while the kernels run and called there ([Overlays](../../docs/reference/overlays.md)).
`mix` has a reader part: a loop that mixes a word. `scale` has a part on all five cores of a
tile:

- the reader fills the core's input tiles with small numbers;
- the three compute cores multiply each tile by a scalar on the vector unit;
- the writer writes the tiles to DRAM.

The kernels are of the kernel form (`kernel_main`); each one names the overlay a command asks for, loads it
and calls it.

| Handler | Arguments | What a core does |
|---|---|---|
| `RUN` | a0 the overlay, a1 a seed, a2 the lines' DRAM address, a3 tiles, a4 the scalar (float32 bits), a5 steps, a6 the tiles' DRAM address | the reader loads the overlay (every part, the writer's and compute's too) and calls its part; the writer and compute load (the meeting) and call theirs. The reader writes its result, the cycles of `load()` and `call()` and whether the part was held to the core's line in DRAM |

The host checks every result: mix's word against the same loop in Python, scale's tiles
value by value, and each core's line names the overlay asked for. It runs the two overlays
in turn (with one slot, each load replaces the other), then each one again and again (held
after the first load). It prints the median cycles of `load()` and `call()`, loading and
held, and one JSON line.

## The program, from the host

```python
program = km.Program("overlays", kernels=KERNELS, handlers=HEADER, buffers=[...])
program.overlay("mix", KERNELS / "mix")  # mix/reader.cpp; kOverlayMix in the header
program.overlay("scale", KERNELS / "scale")  # scale/: reader, compute, writer.cpp
program.slots(count=1)  # one slot a kernel, as large as its largest part
run = session.load(program)  # builds, checks and places every part
```

```c++
// reader.cpp: the reader loads every part of the overlay, the other kernels' too
km::Overlay overlay(reader, command.a0);  // names it; nothing moves
if (!overlay.load()) {  // into a slot of this core; nothing when held
    return;
}
result = overlay.call(MixArgument{command.a1 ^ reader.me(), command.a5});

// compute.cpp and writer.cpp: load() is the meeting with the reader's load()
km::Overlay scale(compute, kOverlayScale);
if (scale.load()) {
    scale.call(ScaleArgument{compute.me(), command.a3, command.a4});
}

// scale/compute.cpp: a part is one function, km_overlay(kernel, argument)
uint32_t km_overlay(km::Compute& compute, const ScaleArgument& argument) { ... }
```

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's kernels: they load and call the overlay a command names |
| `kernels/overlays.h` | the handler, the overlays' numbers (`kOverlayMix`, `kOverlayScale`), the buffers and each part's argument |
| `kernels/mix/reader.cpp` | overlay mix, its reader part |
| `kernels/scale/reader.cpp`, `compute.cpp`, `writer.cpp` | overlay scale: fill, multiply by the scalar, write to DRAM |
| `overlays.py` | runs the commands, checks every result, prints the medians and one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python overlays.py                       # every core of the grid, one slot a kernel
python overlays.py --cores 12 --rounds 10
python overlays.py --slots 2 --tiles 2   # two slots: both overlays stay
```

`--cores 0` (the default) is every core of the grid. The script exits non-zero if a result
differs.

## Numbers

A Blackhole p150a, AICLK 1,350 MHz before and after every run, 10 rounds of each phase
(40 commands). Every result was exact. The parts are 128 B (mix), and 768, 832, 1,280,
1,024 and 1,152 B (scale: reader, writer, unpack, math, pack). Medians over cores and
commands, in cycles:

| | 1 core | 12 cores | 120 cores | 120 cores, 2 slots | Experiments |
|---|---|---|---|---|---|
| `load()` of mix, loading | 998 | 998 | 1,004 | 1,150 | 557, 576, 614 for one 832 B reader body (1, 12, 120 cores; overlay_costs) |
| `load()` of mix, held | 72 | 73 | 72 | 76 | |
| `load()` of scale, loading: five parts, the meeting and one invalidation | 1,318 | 1,316 | 1,347 | 3,126 | 1,061 for five 4 KiB parts on one core; 5,778 median core on 120 (overlay_costs, "a command naming another program") |
| `load()` of scale, held | 72 | 72 | 72 | 97 | 142 for "a command naming the program held" (overlay_costs) |
| `call()` of mix (64 steps) | 532 to 542 | 532 to 542 | 532 to 542 | 540 to 577 | |
| `call()` of scale on the reader (fill one tile, 1,024 values) | 21,824 to 21,858 | 21,825 to 21,861 | 21,829 to 21,860 | 43,581 to 43,687 (2 tiles) | |

Commands: `python overlays.py --cores 1 --rounds 10`, `--cores 12 --rounds 10`,
`--rounds 10`, `--rounds 10 --slots 2 --tiles 2`.

What the numbers say:

- A load costs about one DRAM read and the bookkeeping: 998 cycles for 128 B against 557
  for 832 B in the experiment. The difference is the framework's records, the least
  recently used choice, the checks and the use clock.
- Five parts loaded together cost 1,318 cycles on one core: the meeting, five reads and one
  invalidation. The experiment's five parts of 4 KiB each cost 1,061 cycles without the
  framework's bookkeeping.
- A held `load()` costs 72 cycles on every core, one or five parts, against the
  experiment's 142 for its check: each core reads its own record of the slot, with no
  meeting. With 2 slots and the overlays in turn it costs 76 and 97.
- With 2 slots, only the first command of each overlay loads (120 loading runs). The loading
  median on 120 cores, 3,126 cycles, is that first load on every core at once.
- The calls cost what the same code costs in a kernel: mix's loop takes about 540 cycles
  loaded or held. The slots sit below 1 MB of L1: a loop run from above it takes
  about 4 cycles more an iteration (815 cycles here, 25,900 for scale, before they moved).

[overlay_costs](../../experiments/overlay_costs/README.md) and
[overlay_probe](../../experiments/overlay_probe/README.md) are the experiments these
numbers stand beside.
