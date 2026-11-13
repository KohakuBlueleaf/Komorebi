---
title: Overlays
summary: How a program declares overlays, kernel code built on the host and loaded into a slot of a core's L1 while the kernels run, the rules of loading and calling them, what the builder refuses, and what they cost.
tags:
  - reference
  - runtime
---

# Overlays

Status: implemented on one Blackhole card, through a `Session`. Measured on 1, 12 and 120
cores with parts on all five cores of a tile, every result exact
([12_overlays](../../examples/12_overlays/README.md)).

An **overlay** is kernel code that is not in the kernels when the run starts. The host builds
it when the program loads and puts it in DRAM. A core **loads** it into a **slot**, a fixed
region of its L1, and **calls** it. An overlay has a **part** for each kernel that runs some of
it: the reader, the writer, compute (one part each for the unpack, math and pack cores). A
program without overlays builds the same kernels as before and pays nothing.

## Host

```python
program = km.Program("p", kernels=folder, handlers=folder / "handlers.h")
program.overlay("scale", folder / "scale")  # number: kOverlayScale in handlers.h
program.slots(count=2)  # 2 slots a kernel with parts, each as large as its largest
run = session.load(program)  # builds, checks and places every part
```

| Call | What it does |
|---|---|
| `Program.overlay(name, parts, number=None, defines=None)` | declares an overlay. `parts` is a folder holding any of `reader.cpp`, `writer.cpp`, `compute.cpp`, or a dict `{kernel: source}`. `number` (1 to 63) defaults to the constant `kOverlay<Name>` of the handlers header (`scale_tiles` is `kOverlayScaleTiles`), the way a handler's number does. `defines` go to this overlay's parts only. Returns the program |
| `Program.slots(count=None, bytes=None)` | the slots of each kernel that has parts: `count` 1 to 4, `bytes` a multiple of 64, each an int or `{"reader"\|"writer"\|"compute": value}`. By default, one slot for each overlay with a part there (at most 4), each as large as the largest part. Returns the program |
| `run.program.overlays_built` | after `load`: `images` `{(number, part): bytes}` and `slots` `{part: (count, bytes)}` |
| `km.OverlayError` | the builder refused a part (below); nothing was loaded |
| `km.L1BudgetError` | the buffers, control blocks and slots do not fit a core's L1; `.table` is the L1 budget table of a core |

At load, the session prints the L1 budget table of a core: the circular buffers, the control
block with the overlay area, and each kernel's slots. It also prints the bytes of every part.

## Device

```c++
#include "komorebi/device/km_overlay.hpp"

reader.run([&](const km_command_t& command) {
    km::Overlay scale(reader, kOverlayScale);  // names it; nothing moves
    if (!scale.load()) {  // false: gave up, recorded in the command's state
        return;
    }
    const uint32_t result = scale.call(ScaleArgument{reader.me(), command.a3});
});
```

| Call | What it does |
|---|---|
| `km::Overlay overlay(kernel, number)` | names overlay `number` for `kernel` (a `km::Reader`, `km::Writer` or `km::Compute`) |
| `overlay.load()` | on the reader: reads every part of the overlay that the tile does not hold into a slot, and meets the writer and compute (below). On the writer and compute: the meeting. Nothing moves when every part is held. Returns false when it gave up |
| `overlay.call(argument)` | calls this kernel's part with `argument` (passed by reference) and returns the part's `uint32_t` (0 for a `void` part). Only after a `load()` that returned true |
| `overlay.held()` | whether a slot of this core holds this kernel's part |

### A part

A part is one source file with one function, `km_overlay`, that takes the kernel object and one
argument of the program's own type:

```c++
// scale/compute.cpp
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "../overlays.h"

uint32_t km_overlay(km::Compute& compute, const ScaleArgument& argument) {
    ...
    return argument.tiles;
}
```

The return type is `uint32_t` or `void`. The builder puts the runtime's own includes for that
core before the source (`km_overlay_part.hpp`). After it, the builder adds the entry
(`km_overlay_entry.hpp`), which sits at the slot's first byte, zeroes the part's `.bss` on
every call, and calls `km_overlay`. One `compute.cpp` is built three times, once for each
compute core, as a compute kernel is.

### The rules

- **The reader is the tile's loader.** It reads every part of an overlay from DRAM into its
  slot: its own part, the writer's and the three compute parts. An overlay with a writer or
  compute part needs `load()` called by the reader **and** by that kernel, in the same command.
  The writer's and compute's `load()` park until the reader has written their parts, then
  take the slot from the reader's go. A kernel that loads alone waits until its time limit
  (`wait_limit`), then gives up.
- **Who invalidates.** After the reads have landed, the reader invalidates the instruction
  caches of every core whose part it read, in one write of the tile's register
  (`RISCV_IC_INVALIDATE`). The writer and the compute cores never write or read that register.
  The writer's core cannot reach it at all.
- **When nothing moves.** When the tile already holds every part, `load()` returns at once on
  every kernel. All kernels decide this from the same records, so they skip the meeting
  together.
- **Which slot is replaced.** A load replaces an empty slot, or else the least recently used
  slot with no call running in it. When every slot is running a call, the load gives up.
- **Each core reads its own copy.** Core `i` reads its parts from DRAM bank `i % 8`. Every
  part is stored once in each bank.
- **Every wait is bounded and polls the stop.** A load that gives up goes through the command's
  give-up path:

| `where` | When |
|---|---|
| `overlay_find` | the number is not in the program; the kernel has no part of it; no slot is free; a part is larger than its slot; `call()` without a successful `load()` |
| `overlay_load` | the reads from DRAM did not land within the time limit |
| `overlay_pair` | the meeting did not happen within the time limit, or the kernels met on different overlays |

The gave-up record names the kind `overlay` (for a meeting with the wrong overlay) or the
timeout, the `where` above, and the overlay's number (`"overlay"` in the decoded state).

## The builder

The builder compiles a part with the commands the runtime uses for that kernel and core (the
same flags, includes and defines). It adds position-independent code (`-fpie`), hidden
visibility, and a linker script that keeps the address table (`.got`) inside the image. It
links the part against the persistent kernel's symbols (`--just-symbols`) at two addresses
64 bytes apart. It refuses a part, with an `OverlayError` that names the overlay and the part,
when:

| Check | Why |
|---|---|
| no entry at the image's first byte | the slot's first byte is called |
| relocations left | nothing relocates a part at run time |
| it calls a function in the persistent kernel's text | the kernel's code is not part of the overlay's contract; header-only calls are compiled into the part instead |
| it defines its own copy of the runtime's state (an object the persistent kernel also defines) | the kernel's copy is the one that counts |
| its two links are not byte-identical | it would not run in any slot |
| its code, data and `.bss` are larger than its slot | it would overwrite the next slot |

Parts are cached under the native cache folder, keyed by their preprocessed source, the
recipe, the persistent kernel and the linker script.

The persistent kernels keep and export the state their parts share. Compute exports
`unp_cfg_context`, `pack_sync_tile_dst_ptr`, `math_sync_tile_dst_index`,
`gl_alu_format_spec_reg`, `pc_buf_base`, `regfile` and `mailbox_base`. The writer exports the
NoC counters that its build defines. This happens only in a kernel that includes
`km_overlay.hpp`. A math part resets the math library's cached flag state on entry, and the
math core resets it again after each call.

## L1 layout

| Where | What |
|---|---|
| beside each core's control block, 3,072 bytes | the overlay area: header (table size, each part's first slot, slots and slot bytes), a record per (part, slot) (overlay held, last use, calls running), the reader's meeting lines with each follower, and the table (DRAM address and bytes of every part of every overlay) |
| a sharded L1 tensor, 64-byte aligned | the slots, by part: reader, writer, unpack, math, pack |
| one interleaved DRAM tensor | every part, a copy in each of the 8 banks |

The area is part of the control L1 tensor only when the program has overlays. The host writes
the header and table at load; after that, the cores write the records and lines.

## A leader casts the overlay

A group's leader can pick the overlay its group runs from data it reads while the run is
live, and cast its number as a broadcast's word ([Broadcast](broadcast.md)): every core of
the broadcast reads the word with `broadcast.word()`, then names, loads and calls that
overlay on its reader. Compute does not see the word; the reader hands it to its own
compute through a small buffer, and compute's `load()` meets the reader's as above. On
one card, from the leader's cast to the last call done, an overlay every core loads cost
2.15 to 2.26 µs, one every core holds 1.55 to 1.67 µs, and the same code compiled in 1.00
to 1.12 µs, on a row, a column of 10 and the whole grid
([17_leader_overlay](../../examples/17_leader_overlay/README.md)).

## Costs

On one Blackhole card, AICLK 1,350 MHz, medians over cores and commands
([12_overlays](../../examples/12_overlays/README.md)):

| | 1 core | 120 cores |
|---|---|---|
| `load()` of a 128-byte reader part, loading | 1,013 cycles | 947 cycles |
| `load()` of the same, held (the check) | 420 cycles | 376 cycles |
| `load()` of five parts (768, 832, 1,280, 1,024, 1,152 B), meeting included | 1,382 cycles | 1,414 cycles |
| `load()` of the same, held | 954 cycles | 911 cycles |
| `call()` of a 64-step loop | 802 cycles | 803 cycles |

The experiments these numbers stand beside are
[overlay_costs](../../experiments/overlay_costs/README.md),
[overlay_probe](../../experiments/overlay_probe/README.md) and
[remote_load](../../experiments/remote_load/README.md).
