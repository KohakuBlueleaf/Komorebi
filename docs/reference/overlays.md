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
| `overlay.load()` | on the reader: reads every part of the overlay that the tile does not hold into a slot, and meets the writer and compute (below). On the writer and compute: the meet. Nothing moves when every part is held. Returns false when it gave up |
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
  take the slot from the reader's go: the reader and they **meet**. A kernel that loads alone waits until its time limit
  (`wait_limit`), then gives up.
- **Who invalidates.** After the reads have landed, the reader invalidates the instruction
  caches of every core whose part it read, in one write of the tile's register
  (`RISCV_IC_INVALIDATE`). The writer and the compute cores never write or read that register.
  The writer's core cannot reach it at all.
- **When nothing moves.** When the tile already holds every part, `load()` returns at once on
  every kernel. All kernels decide this from the same records, so they skip the meet
  together. A kernel that found the tile holding the overlay marks its slot's record with
  the tile's **epoch**, a count the reader steps whenever it changes which overlay a slot
  holds. While the epoch is unchanged, that kernel's next `load()` of the overlay is a
  look at its slots' records: about 70 cycles (below).
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
| `overlay_meet` | the meet did not happen within the time limit, or the kernels met on different overlays |

The gave-up record names the kind `overlay` (for a meet on the wrong overlay) or the
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
| beside each core's control block, 3,328 bytes | the overlay area: header (table size, each part's first slot, slots and slot bytes), a record per (part, slot) (overlay held, the part's check at an epoch, calls running, the slot's address), the clock and the epoch, the reader's meet lines with each follower, the last use of each overlay, and the table (DRAM address and bytes of every part of every overlay) |
| a sharded L1 tensor, 64-byte aligned, just under 1 MB of L1 when it fits above the circular buffers | the slots, by part: reader, writer, unpack, math, pack. A loop runs from L1 at or above 1 MB about 4 cycles an iteration slower; the load logs a line when the slots end up there |
| one interleaved DRAM tensor | every part, a copy in each of the 8 banks |

The area is part of the control L1 tensor only when the program has overlays or the run
holds programs in the runtime's form. The host writes
the header and table at load; after that, the cores write the records and lines.

## A leader casts the overlay

A group's leader can pick the overlay its group runs from data it reads while the run is
live, and cast its number as a broadcast's word ([Broadcast](broadcast.md)): every core of
the broadcast reads the word with `broadcast.word()`, then names, loads and calls that
overlay on its reader. Compute does not see the word; the reader hands it to its own
compute through a small buffer, and compute's `load()` meets the reader's as above. On
one card, from the leader's cast to the last call done, an overlay every core loads cost
2.12 to 2.23 µs, one every core holds 1.19 to 1.31 µs, and the same code compiled in 1.00
to 1.13 µs, on a row, a column of 10 and the whole grid
([17_leader_overlay](../../examples/17_leader_overlay/README.md)).

## A message's handler as an overlay

A dynamic handler's body can be an overlay's compute part with the entry `km_message`
([Dynamic handlers](dynamic.md#handlers-as-overlays)). The messages reader is the tile's
loader there too: it sees each message's handler before compute does, groups a packet's
messages by handler and runs `load()` of each group's overlay after it hands compute the
packet; compute meets it only for a handler the tile does not hold. On 120 cores, three
handlers as parts move 10^6 messages as fast as the same handlers compiled into the
compute kernel ([20_message_overlays](../../examples/20_message_overlays/README.md)).

**An overlay sent by name.** A message carries its overlay's name, not its bytes (code
is larger than a 4 KiB body, and it is in DRAM already): `messages.send(rank, message,
handler)` or `forward(rank, message, handler)` names a handler whose part is an overlay,
and the core the message lands on loads that part from DRAM into a slot, if it does not
hold it, before it calls it. So a thread brings its code with it, and a core that holds
the overlay pays nothing. In [23_message_body](../../examples/23_message_body/README.md)'s
`--send` run (card 1, 12 ranks, 1200 hops, a 4 KiB body changed at each hop) only the
`RUN` command's small handler is loaded everywhere; each rank loaded the chain handler
(7,232 bytes of parts) once, at the thread's first arrival: a first arrival took 8.32 µs,
a later one 6.51 µs.

## Costs

On one Blackhole card, AICLK 1,350 MHz, medians over cores and commands
([12_overlays](../../examples/12_overlays/README.md)):

| | 1 core | 120 cores |
|---|---|---|
| `load()` of a 128-byte reader part, loading | 998 cycles | 1,004 cycles |
| `load()` of the same, held (the check) | 72 cycles | 72 cycles |
| `load()` of five parts (768, 832, 1,280, 1,024, 1,152 B), the meet included | 1,318 cycles | 1,347 cycles |
| `load()` of the same, held | 72 cycles | 72 cycles |
| `load()` held, two overlays in turn in two slots | | 76 and 97 cycles |
| `call()` of a 64-step loop | 532 to 542 cycles | 532 to 542 cycles |

## A program as an overlay

A program in the runtime's form ([host](host.md#several-programs-in-one-run)) is an
overlay under kernels the runtime owns (`src/komorebi_native/kernels/program_reader.cpp`,
`program_writer.cpp`, `program_compute.cpp`). Program field `p` is overlay number `63 - p`
(`km::program_overlay(p)`). For each command the runtime's kernels run

```c++
km::Overlay program(reader, km::program_overlay(km::program(command)));
if (!program.load()) {
    return;  // the give-up path
}
program.call(command);
```

and the same on the writer (which completes the command unless the program's
`km_program` returned false) and on compute. A NOP is not passed to a program.

- **The parts.** The builder builds a program's three sources as an overlay's parts with
  the function `km_program` in place of `km_overlay`, and with the program's
  `compile_args` and `defines` in place of the runtime kernel's. A part holds no state
  from one command to the next (its `.bss` is zeroed at every call): what it must keep it
  reads from the buffers' own pointers or the command.
- **The slots.** By default a slot a program on each kernel (at most 4), each as large as
  the largest part; the first program's `slots()` sets them. With fewer slots than
  programs, a command to a program the tile does not hold loads its five parts.
- **The compute start-up.** The runtime's compute kernel calls
  `compute_kernel_hw_startup(in, out)` once, on the group's first buffer below index 16 and
  its first from 16 on (0 and 16 when there is none); in a run loaded with a dict, each
  load of a program runs it instead, on that program's buffers. A part that uses other
  buffers calls the compute API's own `*_init` calls for them, as any compute code does.
- **Each group apart.** With `{program: group}`, each group's runtime kernels are built
  apart, on its cores, with its buffers, and its overlay table holds only the program it
  holds; the dispatcher refuses a command for another (reason 13), and a core that still
  got one would give up (`overlay_find`).

What it costs ([16_two_programs](../../examples/16_two_programs/README.md)), cycles from
the reader taking a command to its completion, medians, one command at a time:

| Command | the kernel form | the runtime's form |
|---|---|---|
| empty, 1 core | 596 | 724 |
| empty, 120 cores | 811 | 941 |
| a sum of 16 tiles, 1 core | 5,233 | 6,274 |
| the first command after a change of program (five parts loaded), 1 core | | 2,919 |
| the same, 120 cores | | 3,971 |

## A group switches program

In a run loaded with a dict, `run.load(program, group=)` ([host](host.md#a-group-switches-program))
or an X280 thread's `kmx_load` ([threads](threads.md#a-thread-gives-a-group-another-program))
is a command with the reserved handler 240 (`KM_HANDLER_LOAD`): its program field names
the new program, its word `a0` the program's record in DRAM (in every bank, as the parts
are: the program field, its five overlay table entries, and per buffer index its
address, size, pages and page size). The rule: code loads on demand, the L1 layout
changes only by this command. On each core of the group:

1. The writer and the three compute cores mark their arrival on the switch line and park.
2. The reader waits for the four, then reads the record (the last load's record is theirs
   until they arrive), and waits until every buffer of the old and the new program is
   empty (pages received equal pages acknowledged).
3. It zeroes those buffers' counters and sets up its own view of the new buffers
   (`setup_local_cb_read_write_interfaces`), clears the old program's table entries,
   writes the new ones, steps the epoch (a slot's records are checked again) and lets
   the four go.
4. Each of the four sets up its own view (the writer reads and writes, unpack reads,
   pack writes) of the new buffers; the runtime's own buffers (indices 29 to 31) are not
   in the mask and stay as they are.
5. The command goes on as any command for the new program: its parts are loaded if the
   tile does not hold them, and called; the compute parts run
   `compute_kernel_hw_startup` on the program's first input and output buffers, so the
   unpacker and packer take its data formats.

**Who may issue a load.** The host and the thread of a work cluster; from a stream or a
graph the dispatcher rejects it (reason 1), and a cast or a pull cannot name one. Both
loads mean the same: exclusive (no job queued, no command in flight on a core of the
group), and the per-core table moved when the dispatcher submits it, so it judges every
later command from any source (reason 13). They differ only in the wait: the host's
load waits in its ring, keeping the host's later commands behind it; a thread's that
cannot go now is rejected at once with reason 11 and a wake owed, as every thread
command is, and `kmx_load` sends it again after the wake
([21_thread_load](../../examples/21_thread_load/README.md): a load to the core of a long
command returns after that command, 37 µs after the command was sent).

A load to the program a core already holds sets its buffers up again (empty) and checks
the host's layout against the live one: the session's start-up loads do this, and a
mismatch gives up with where 39 (`ProgramSwitch`). A data command for a program whose
load did not run on a core gives up on compute with where 42 (`ProgramStart`).

What it costs, cycles from the reader taking the load to its completion (medians,
[16_two_programs](../../examples/16_two_programs/README.md), `--phase load`):

| Load | 1 core | 120 cores |
|---|---|---|
| to another program (buffers set up, its parts held in their slots) | 3,124 | 3,360 |
| to the program held (buffers set up, parts held) | 3,249 | 3,475 |

That is 2.3 to 2.6 µs at 1,350 MHz, beside `overlay_costs`' 0.68 µs for setting up the
buffers alone and 7 to 95 µs for loading five parts on 120 cores: a load command costs the
meeting of the five cores and the wait for their buffers to empty, not the code.

From an X280 thread a load costs the dispatcher the same (submit to completion, the
cores that change program in a round: 3.9 to 4.1 µs against 3.8 to 4.0 µs from the
host), and `kmx_load` returns 6.1 to 6.2 µs after it is called, where the host's
`run.load` and its wait take 134 µs
([21_thread_load](../../examples/21_thread_load/README.md#numbers)).

## Transfers to and from the host

Two more reserved handlers move pages between a DRAM tensor and the session's region of
the host channel while a run is loaded ([host](host.md#the-host-channel)): 241
(`KM_HANDLER_PUSH`, card to host) and 242 (`KM_HANDLER_PULL`, host to card). They are
not overlays. Their code is compiled into the runtime's loop of every program
(`komorebi/device/km_transfer.hpp`), as EXIT's is, so a program declares nothing and
holds no slot for them. The cost is a larger reader kernel, about 4.4 KB of its text in
the ray tracer. Handlers 241 to 248 are the runtime's for this: the writer and the
compute kernel skip them with one range check, so a later handler of the runtime in
that range adds nothing to them, only to the reader.

- **Who may issue one.** The host only: the session itself, for a read or a write. The
  dispatcher rejects them from a thread, a stream or a graph (reason 1), as `submit`
  does from the host.
- **The command.** Its group is cores `[first, first + count)` that all hold one
  program, and its program field is that program, so no switch is needed. Its words
  name the tensor (address, page bytes, first page, pages), the region (offset, the
  word slots, its size), a nonce, and the core's L1 buffer (two chunks and a word slot)
  that the session reserves at the load: the row "host transfer buffer" of the L1
  budget.
- **On each core.** The reader checks the bounds first: the data and the word slots
  inside the region and apart, offsets in multiples of 16, the chunks inside L1. A
  failed bound gives up with where `tohost` or `fromhost` and gave `bad_transfer`,
  having written nothing. Then the reader moves its share (`km::partition` of the
  pages) through the two chunks: set the PCIe state, the writes or reads, wait for
  them, clear. Then it writes its sequence word. Each wait is bounded by the run's wait
  limit. The writer only completes the command, and compute skips it.

A third, 245 (`KM_HANDLER_COPY`), copies pages from DRAM to DRAM through the same two
chunks: from a range of one bank or a tensor's pages into another range or tensor
(`komorebi/abi/km_copy.h`). It is compiled in beside the two, and the dispatcher
accepts it from the host and from a work cluster's thread, which sends it with
`kmx_fetch` and `kmx_store` ([X280 threads](threads.md#reading-and-writing-another-bank)).
The runtime's loop of every program serves it, so it needs no program field: the
dispatcher does not check which program its cores hold. A failed bound gives up with
where `copy`.

The experiments these numbers stand beside are
[overlay_costs](../../experiments/overlay_costs/README.md),
[overlay_probe](../../experiments/overlay_probe/README.md) and
[remote_load](../../experiments/remote_load/README.md).
