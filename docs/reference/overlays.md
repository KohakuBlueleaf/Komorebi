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
| `Program.overlay(name, parts, number=None, defines=None)` | declares an overlay. `parts` is a folder holding any of `reader.cpp`, `writer.cpp`, `compute.cpp`, or a dict `{kernel: source}`. `number` (1 to 62; 63 is the runtime's service overlay) defaults to the constant `kOverlay<Name>` of the handlers header (`scale_tiles` is `kOverlayScaleTiles`), the way a handler's number does. `defines` go to this overlay's parts only. Returns the program |
| `Program.slots(count=None, bytes=None, service=None)` | the slots of each kernel that has parts: `count` 1 or more (only the L1 budget table refuses a count), `bytes` a multiple of 64, each an int or `{"reader"\|"writer"\|"compute": value}`; `service` `"own"` or `"shared"` places the service overlay (below), None lets the load choose. By default, one slot for each overlay with a part there, each as large as the program's largest part there; a count not given is lowered toward 1, the largest slots first (in a load whose groups switch program: the part with the most slots first, the largest of those, so the counts stay level), while the L1 budget table does not fit (one log line a part: `slots: 4 a reader where the program's overlays would take 5: L1 left N B`). Over 4 slots of a part, the kernels are built with `KM_OVERLAY_SLOTS` and the records of slots 4 and up follow the overlay area (16 B a slot a part); a program with up to 4 keeps its kernel text. Returns the program |
| `run.program.overlays_built` | after `load`: `images` `{(number, part): bytes}`, `slots` `{part: (count, bytes)}` (the program's), `service` (the bytes of the service overlay's own slot, 0 when it shares the reader slots) and `why` (the load's reason) |
| `km.OverlayError` | the builder refused a part (below); nothing was loaded |
| `km.L1BudgetError` | the buffers, control blocks and slots do not fit a core's L1; `.table` is the L1 budget table of a core |

At load, the session prints the L1 budget table of a core: the circular buffers, the control
block with the overlay area, where the service overlay goes and why, its own slot if it
has one, and each kernel's slots. It also prints the bytes of every part.

## Device

```c++
#include "komorebi/device/overlays/km_overlay.hpp"

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
  The service overlay's own slot, when it has one, is a record more than the reader's
  four: only the service loads there, and it loads nowhere else; a program's overlay
  that is held finds its record before that one is looked at. A slot's address is in
  its record.
- **Each core reads its own copy.** Core `i` reads its parts from DRAM bank `i % 8`. Every
  part is stored once in each bank.
- **Every wait is bounded and polls the stop.** A load that gives up goes through the command's
  give-up path:

| `where` | When |
|---|---|
| `overlay_find` | the number is not in the program; the kernel has no part of it; the service overlay's own slot is busy; a part is larger than its slot; `call()` without a successful `load()` |
| `overlay_slots` | the reader found no free slot for a part: every slot of that kernel holds a part that runs; the row says "no free slot for overlay N: S slots, all held" (give the kernel more slots) |
| `overlay_load` | the reads from DRAM did not land within the time limit |
| `overlay_meet` | the meet did not happen within the time limit, or the kernels met on different overlays |

The gave-up record names the kind `overlay` (for a meet on the wrong overlay) or the
timeout, the `where` above, and the overlay's number (`"overlay"` in the decoded state).

## The builder

The builder compiles a part with the commands the runtime uses for that kernel and core (the
same flags, includes and defines). It adds position-independent code (`-fpie`), hidden
visibility, and a linker script that keeps the address table (`.got`) inside the image. It
links the part against the persistent kernel's symbols (`--just-symbols`) at two addresses
64 bytes apart. A copy or a clear that the compiler turns into a call to `memcpy` or
`memset`, at any level, links the C library's copy into the part; when the persistent
kernel defines that function (Wormhole's writer), the part takes a copy of its own
instead, linked in only when called (`overlay.part_memory`,
`src/komorebi_native/kernels/program/part_memory.cpp`, through `--wrap`). On Wormhole a writer part is built at `O2` (`chips.Chip.writer_part_opt`),
not at the persistent writer's `Os`: a part runs from L1, and its slot bounds its size.
From writer level 4 the writer's parts are built at the writer's own `Os`
(`kernels.PART_OWN_LEVEL_FROM`; [hazards](hazards.md#a-wormhole-writer-part-at-o2-from-writer-level-4)).
It refuses a part, with an `OverlayError` that names the overlay and the part, when:

| Check | Why |
|---|---|
| no entry at the image's first byte | the slot's first byte is called |
| relocations left | nothing relocates a part at run time |
| it calls a function in the persistent kernel's text | the kernel's code is not part of the overlay's contract; header-only calls are compiled into the part instead |
| a call or jump of the linked part leaves it (read from its disassembly, so a call the compiler made at the link is seen; named with its address) | the kernel's or the firmware's code is at another offset at each address of the part |
| it defines its own copy of the runtime's state (an object the persistent kernel also defines) | the kernel's copy is the one that counts. A movement part uses the kernel's words of which halves were filled and drained (`km_movement.hpp`, linked against its symbols), or words of its own when the kernel never moves; the build's 22 constant tables of the buffers are not state when their bytes are the kernel's (else refused, naming the table); a math part's `throttled_mop_status` is built set unknown at entry, and the kernel's after the call |
| its two links are not byte-identical | it would not run in any slot |
| its code, data and `.bss` are larger than its slot | it would overwrite the next slot |

Parts are cached under the native cache folder, keyed by their preprocessed source, the
recipe, the persistent kernel and the linker script. The preprocessed source and the
persistent kernel carry the path of the folder their source is in, so a class's build
folder is named by its text (a class made twice with equal text builds its parts once).

The persistent kernels keep and export the state their parts share. Compute exports
`unp_cfg_context`, `pack_sync_tile_dst_ptr`, `math_sync_tile_dst_index`,
`gl_alu_format_spec_reg`, `pc_buf_base`, `regfile` and `mailbox_base`. The writer exports the
NoC counters that its build defines, and at writer level 4 `km::chain_now` (a part links
to the kernel's global symbols only). This happens only in a kernel that includes
`km_overlay.hpp`. A math part resets the math library's cached flag state on entry, and the
math core resets it again after each call.

## L1 layout

| Where | What |
|---|---|
| beside each core's control block, 3,328 bytes | the overlay area: header (table size, each part's first slot, slots and slot bytes, the service overlay's own slot), a record per (part, slot) (overlay held, the part's check at an epoch, calls running, the slot's address), the clock and the epoch, the reader's meet lines with each follower, the last use of each overlay, and the table (DRAM address and bytes of every part of every overlay) |
| the run's first circular buffer, at the bottom of the L1 left after the kernel buffer (0x1B480 with the default one), so below 1 MB, with the other circular buffers above it | the slots, by part: reader, writer, unpack, math, pack. A taken branch to code at or above 1 MB of L1 costs about 4 cycles more: slots that would end there raise `km.SlowCodeError` unless the session was opened with `slow_code=True` |
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
overlay under kernels the runtime owns (`src/komorebi_native/kernels/program/reader.cpp`,
`program/writer.cpp`, `program/compute.cpp`). Program field `p` is overlay number `62 - p`
(`km::program_overlay(p)`); 63 is the service overlay ([below](#transfers-to-and-from-the-host)). For each command the runtime's kernels run

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
- **The slots.** By default a slot a program on each kernel (one more where a class named ahead has an
  own overlay; counted before the kernels are built, so past 4 they get
  `KM_OVERLAY_SLOTS`), each as large as
  the largest part; the first program's `slots()` sets them. With fewer slots than
  programs, a command to a program the tile does not hold loads its five parts. A
  kernel where a program has a part of an overlay of its own (a class's handler) has 2
  at least, even in a dict of one program: that overlay loads into a slot beside the
  running program's. A `slots(count=1)` there is refused by name at the load, and so
  are slots the L1 left lowers to 1 there.
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

Two more reserved handlers move pages between a tensor in DRAM or L1 and the session's
region of the host channel while a run is loaded ([host](host.md#the-host-channel)): 241
(`KM_HANDLER_PUSH`, card to host) and 242 (`KM_HANDLER_PULL`, host to card). Their code
(`komorebi/device/tensors/km_transfer.hpp`) is the reader part of the runtime's **service
overlay**, number 63 (`km::kOverlayService`, source
`src/komorebi_native/kernels/program/service_reader.cpp`), which every run builds and places
beside the program's own overlays: a program declares nothing. A core's reader loads it
on the first such command (`km::serve`) and holds it while no other reader overlay
needs its slot. The load chooses where it goes and the L1 budget table says why (`slots:
service (reader), ...`): its own slot, a record more than the reader's, as large as the
service part (9,088 bytes on one card, 41,280 on an island of two cards), when the L1
left has the room beside the program's slots; else the reader slots, as large as the
service part at least, when that holds more of the program's parts at once or leaves
more L1 (as before this choice). A program's slots are sized to its own parts and
keep their count either way; `slots(service="own")` or `"shared"` decides it. A
program with no overlays of its own (and no added handlers) has the reader slots
alone, the same layout: its reader is built without the own slot's code (320 to 400
bytes of BRISC text less: the gate class, Linear, attention). In the
reader
kernel only the load stays, about 3.7 KB less text in the ray tracer than with the
code compiled in. Handlers 241 to 248 are the runtime's for this: the writer and the
compute kernel skip them with one range check, so a later handler of the runtime in
that range adds nothing to them.

On Wormhole the service overlay has a **writer part** too
(`src/komorebi_native/kernels/program/service_writer.cpp`), as the writer's CPU
(NCRISC) holds 16 KiB of code. A program's writer is built at **writer level** 0,
everything resident; only a writer over the 16 KiB is built again a level up, to
`chips.Chip.writer_levels` (4 on Wormhole), and the level found is kept by program and
chip in the remembered store (`kernel_buffer.json`, `writer_levels`), so a later load
builds once, at it. Each level moves more of the writer's paths that run after a
command's work out of the resident writer into the part (`KM_WRITER_COLD`,
`kernels.WRITER_LEVELS`): 1 the page-bound lookup of a missed bound, 2 also
`Core::finish` with its clear of the buffers, 3 also `km::complete`, 4 also a chain
record's marks to the next record's cores (`km::chain_mark`, for the chain's own shape). The reader loads
the part at the start of the load, before the first command, into a writer slot held
for the life of the load (3,776 bytes of L1 a core at level 3, the L1 budget table's
`slots: writer` line); the writer calls it through one pointer. The moves and the waits
stay resident. A program that fits at level 0 has no part and runs as before. On
Blackhole every program is at level 0 and its kernels are as before, byte for byte.

**The near build.** On an island load, a writer still over the 16 KiB at the highest
level is built once more **without the far paths** (`KM_ISLAND_NEAR`,
`komorebi.program.checks.near`), walking the levels from 0 again, when the host can
keep every command of the load on its own island. That holds for a load whose
programs are each a class's, not dynamic, not on a card split into islands, with no
X280 thread and no chain gather, whose make's island tensors are copies, and whose
movers' sources name no collective and no put or ask line (`near.NAMES`, comments and
strings aside). Its movers keep the tensor table and lose the page links' client side
(`near.DROPPED`; the link cores keep them): a `km::Tensor` is this island's bank
address of its pages (`km::here_address`), so its streams and page calls are the
one-card ones. The listing says so, with the far build's size and what is left:
`writer code: 13,160 of 16,384 B, margin 3,224 B, writer level 0, built without the
far paths (with them the writer is 26,552 B of 16,384 B at the highest writer level)`.
A load over at the highest level that cannot be near is refused with its size and
the reason. That a program was built near is kept with its level, so its next load
(the next island's) builds near at first, at that level. At the submit the host refuses by name, before anything is queued:

- a tensor that is not whole on the island that runs the command (a copy is; a
  tensor one island holds is whole there alone; a split one is whole nowhere);
- a command of a handler that reduces, uses ports or broadcasts, when its group (or
  one of its reduce's groups of S ranks) holds ranks of two islands;
- a collective of the host (`run.all_reduce()` and the others);
- a command given as words (`run.submit_many`, or a handler named by its name or
  number): it holds no tensor to check; submit the class's handler with its tensors.

Behind those checks, on the card, a page of a tensor that is not whole on its island
(under `KM_GUARD`), a put or ask line, a collective or a chain's gather gives up by
name (where `near`). Blackhole has no writer levels and never reaches it.

- **Who may issue one.** The host: the session itself, for a read or a write. An X280
  thread too, when its command's region word is the session's region (the load hands
  the dispatcher its size); with any other region word, and from a stream or a graph,
  the dispatcher rejects them (reason 1), as `submit` does from the host. The other
  transfers of the range (pages, gather, reduce) stay the host's.
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
  them, clear. For a tensor in L1 (a chunk of `KM_TRANSFER_L1`, one page a core) the
  reader moves its own page in one write or read instead. Then it writes its sequence
  word. Each wait is bounded by the run's wait limit. The writer only completes the command, and compute skips it.

A third, 245 (`KM_HANDLER_COPY`), copies pages from DRAM to DRAM through the same two
chunks: from a range of one bank or a tensor's pages into another range or tensor
(`komorebi/abi/km_copy.h`). It is in the service overlay beside the two, and the dispatcher
accepts it from the host and from a work cluster's thread, which sends it with
`kmx_fetch` and `kmx_store` ([X280 threads](threads.md#reading-and-writing-another-bank)).
The service overlay of every run serves it, so it needs no program field: the
dispatcher does not check which program its cores hold. A failed bound gives up with
where `copy`.

## Code added to a live run

`run.addon(name, parts, handler=False, number=None, defines=None, timeout=10.0,
message=False, compile_args=None)` adds an overlay to a run that is loaded and live. Its
parts are built against the loaded kernels, exactly as the load builds its overlays (with
`compile_args`, `{kernel: [words]}`, in place of the kernels' own: a load in the runtime's
form runs the runtime's kernels, which have none, so a part that reads a program's
compile arguments is given them), then written into
room the load reserved, and every core is given the overlay's table entry by a command
of the runtime's service overlay. The host writes no core's L1. The call returns the
overlay's number. From then on the overlay is named, loaded and called as any overlay
of the load is ([30_add_overlay](../../examples/30_add_overlay/README.md)).

```python
run = session.load(program)  # program.slots(...) gave slots for the code added later
number = run.addon("mix", folder / "mix")  # an overlay: a kernel names it by number
scale = run.addon("scale", folder / "scale", handler=True)  # a handler's body
run.wait(run.submit(scale, (0, 120), words))  # submitted as any handler
```

What happens:

1. **The build.** The overlay gets its number: `number`, or the lowest number from 1
   to 62 that is neither a loaded overlay nor an added one. For a handler, the number
   is also not a handler of the run's programs. Each group of cores with its own build
   of the kernels gets its parts built against those kernels: the same compiler
   command, the persistent kernel's ELF, its symbols and link script. Each part also
   goes through the instruction check and the part checks ([the builder](#the-builder)),
   and must fit the kernel's slot: the program's slots, never the service overlay's
   own. A load whose program has no reader part and named no reader bytes has no
   reader slot of its own, so it names the reader's bytes for an added reader part.
2. **The cache of built parts.** A part is kept under a key of what decides its bytes:
   its source file, its defines, the recipe of the kernel it is built against (compiler,
   levels, flags, includes, defines), the persistent kernel's ELF, the link script and
   a hash of the builder's and the checks' Python source (`program/overlay.py`,
   `program/checks/instructions.py`; a merged part's also `merge.py`), so a changed check checks a kept
   part again. The instruction check's markers on kernels hold its source's hash too.
   The key does not hold the overlay's name or number. Beside the key, the cache keeps
   the digest of every file the part includes. A part whose key and files are unchanged
   is not built again, in this session or any later one, and the instruction check is
   not run again either.
3. **The room.** `session.load(reserve=)` gives the overlays' DRAM tensor that many
   bytes more, after the load's parts. Each part is stored in every bank (default
   256 KiB, one row of the tensor in each of the 8 banks). The parts are written there
   through the host channel while the run goes on, and the same bytes are written once
   for all groups.
4. **The entry.** One command a group, handler 246 (`KM_HANDLER_ENTRY`, host only),
   served by the service overlay. Each core writes the overlay's table entry (each
   part's DRAM address and bytes) into its own overlay area and raises the table's
   count. For a handler it also sets the number's bit in a mask of the area
   (`kOverlayHandlers`); for a dynamic handler, a bit in `kOverlayMessages`. The
   host's copy of the areas gets the same entry. A run that keeps reset lists
   (`load(recover=True)`) hands them to the dispatcher again, so a core it resets
   comes back with the added entries.

**A handler added to the live run.** The kernels of a load whose program names slot
bytes (`Program.slots(bytes=...)`, which defines `KM_ADDED_HANDLERS`) look at the
handler mask before the body. A command whose handler has its bit set is entered: the
reader loads the overlay of that number as any `load()`, and each kernel with a part
calls it with the command. A tile's first command of that handler loads the parts,
and later commands find them held. In the runtime's form the define is given to the
runtime's kernels when every program of the load names slot bytes (the gate class in a
list of two: 30,416 B of kernel text without it, 33,088 B with it). Without that define
the kernels have no such path and cost nothing. It costs 2,560 B of kernel text on the decode program (148 B on the
reader, 844 B on the writer, 484 to 544 B on each compute core).

**The path is compiled only when asked.** What asks is `Program.slots(bytes=...)` (an
int, a `{kernel: n}`, or `bytes={}` for the default sizes), a class's `km.ANY` among a
handler's sets, or the define itself in a program's `defines`; `session.load(reserve=)`
only sets the DRAM room aside and does not ask. Without it the loop's check of the
handler mask, the load of an added handler's overlay and the service overlay's entry
command are left out of the kernels: Linear is 1,520 B smaller (90,304 B of 91,136), the
gate class 448 B, the attention class 448 B, 01_mapcollect 544 B, and the service
overlay's reader slot 448 B (8,640 B instead of 9,088 B, a slot on each core: 1,792 B of
L1 on the decode program's four). A program that asks is built as before, byte for
byte. `run.addon` of any kind, an overlay as well as a handler, on a run whose
program did not ask is refused before any command, by name. A load that names slot bytes for a kernel gives that kernel slots
even when no overlay of the load has a part there.

**Every way names it.** An added overlay is named by every way an overlay of the load
is, with its number:

| Way | How it names the overlay | Run on the card |
|---|---|---|
| a handler's `km::Overlay` | the number in a command's word | [30_add_overlay](../../examples/30_add_overlay/README.md): RUN names mix, 120 cores, from the host and the X280 thread |
| a handler | `run.addon(handler=True)`, then `run.submit(number, ...)` or a thread's command | 30_add_overlay: SCALE, three kernels' parts, from the host and the thread |
| a leader's cast | the broadcast's word | [17_leader_overlay](../../examples/17_leader_overlay/README.md) `--addon shift`: shift (reader and compute parts) added, cast to a column of 10 |
| a pull's answer | the thread's `KMX_PULL_WORD` | [18_thread_split](../../examples/18_thread_split/README.md) `--addon sparse`: sparse added, 256 units on 120 cores |
| a message's handler | `run.addon(..., message=True)`: dynamic handler n is overlay n, n from 1 to 16 | [20_message_overlays](../../examples/20_message_overlays/README.md) `--addon`: report added, 10,000 reports on 120 cores |

A message's handler that is not in the program's list (`KM_MESSAGE_HANDLERS`) is looked
up once in the area's mask `kOverlayMessages`. The reader does that lookup when it
groups a packet. A handler in the list is found as before, so a message of the load's
handlers costs what it did: ex20's kernel text grew by 16 B, and its run took 11.05 ms
with report loaded and 11.11 ms with report added (medians of 3, one hour apart). A
dynamic handler added this way is a message's handler only: a RUN command names one of
the program's.

**Refused**, with `km.ProgramError` (`komorebi.program.addon.code.AddError`) and nothing written to
the card:

| Case | Message names |
|---|---|
| the load reserved no room (`reserve=0`, a load over islands) | `session.load(reserve=0)` |
| the name was added before | its number |
| a number outside 1 to 62, an overlay's, or (a handler) a handler's | the number |
| no number left | how many overlays and handlers take them |
| a part on a kernel without slots | `Program.slots(bytes={kernel: n})` |
| a part larger than its slot, or one the instruction check or the part checks refuse | the part, its bytes and the slot's |
| the room is short | the bytes the parts take and the bytes left |
| an overlay, a handler or a dynamic handler in a load without `KM_ADDED_HANDLERS` | `Program.slots(bytes=...)`, or `km.ANY` |
| a dynamic handler in a run with no handlers as parts, outside 1 to 16, or one the run has | the number |

**Costs**, one Blackhole card at AICLK 1,350 MHz, 120 cores
([30_add_overlay](../../examples/30_add_overlay/README.md)):

| | |
|---|---|
| a first add (the parts compiled): mix, one reader part / scale, five parts | 0.85 s / 5.09 s (the build off the card, the same commands) |
| a cached add: mix / scale | 8.6 to 9.0 ms / 42.8 ms |
| the parts' write into the room | 0.3 to 0.4 ms |
| the entry commands, with the reset lists kept again / without | 3.0 to 3.1 ms / 0.1 to 0.2 ms |
| a command of RUN naming mix: first (every core loads) / later (held) | 3.78 µs / 2.76 µs (2.22 µs naming none) |
| mix's `load()` on a core: loading / held | 1,286 / 115 cycles |
| a command of SCALE, the added handler: first / later | 38.32 µs / 35.98 µs |
| the same from the X280 thread: first / later | 36.74 µs / 35.96 µs |

**A class's handler on its first use.** A class made with `km.ANY` among a handler's
sets has a set that was not built compiled at its first `run.submit`, as that
handler's overlay parts on the load's buffers, and added by `run.addon(...,
handler=True)` ([classes, a set compiled on its first use](classes.md#a-set-compiled-on-its-first-use)).
The gate's added sets ran within 1.6 % of the same blocks loaded.

**Built ahead.** `program.ahead(name, parts, defines=..., compile_args=...)` names code
a run of the program will add (run.addon's arguments), and `program.ahead(Gate.gate,
size=4)` a class's handler at a set its first use compiles (compiled then, the same
name and parts the first use adds). The load builds each part for each group against
its kernels into the parts' cache, after the kernels and before the launch, and logs
`ahead <name>: N parts for G groups in S s (H built before)`; the add later finds them
there. A part's key holds no overlay number, so the number the add takes does not
matter. A build off the card (the same kernels) fills the same cache, so the card's
load finds them too. The gate's `GateBlocks` made with `km.ANY`, block 4 given to
`ahead`: the build off the card built its five parts in 7.16 s; on the card the load
found the five built (0.06 s), the first use's add took 56.8 ms (cached), and the
command ran 0 wrong on 120 cores. A load that reserved no room builds none; a set of a
handler without `km.ANY` is refused (a listed set is built with the program).

### Another class

`run.addon(program)` adds a class made in the runtime's form to a run loaded with a dict
as one more program ([host](host.md#a-class-added-to-a-live-run)). Its parts are built
against group 0's recipes (a switch run's groups share one build), each with a copy of
the kernel build's `chlkc_descriptors.h` whose format tables hold the class's buffers,
first among the includes; the part's index in the cache records that copy, and the add
checks it. Its parts and its record go into the reserved room. A class named with
`Program.ahead(other)` is built with group 0's overlays at the load, so the slots fit
its parts and the add finds them in the cache. A class with per-core args has its parts
built with `KM_ARGS_L1` (the record word holding its row's L1 address) and the writer's
and compute's offsets in the row, so its `arg()` reads the row its load command copied
in; the record's words 12 to 14 hold the row table's address, a row's bytes and that L1
address (0, 0, 0 for a class without args). Its own overlays take the lowest free
numbers below every program field's (a shift added to the numbers its parts name,
`KM_OVERLAY_SHIFT`); their parts go into the room and each group's cores write their
entries by the entry command (built for it by `KM_ADDED_ENTRY` when such a class is
named ahead).

### Several programs with overlays

Programs in the runtime's form ([host](host.md#several-programs-in-one-run)) may each
declare overlays of their own. The 62 numbers below the service overlay are shared:

- Program field p is overlay 62 - p, from 62 down.
- Each program's own overlays are numbered after those of the programs before it: its
  numbers plus the largest number of each earlier program. A part names its own
  numbers, and the runtime adds the shift (`KM_OVERLAY_SHIFT`).
  `komorebi.program.several.own_overlays(program)` gives a program's own overlays as
  the run numbers them, shift added; `program.overlay_shift` is that shift, set at the
  load.
- Overlays added to the live run take the lowest numbers left.
- In a run loaded with a dict, every group holds every program's own overlays, and
  each is built, on every group, with the recipes of the group that holds its program
  (`komorebi.program.addon.program.home_overlays`), as the program's part is: its
  buffers' format tables and Dst mode are its program's. Built with another group's
  recipes, a part read its buffers through that group's tables: a group switched to
  a program whose buffers that group lacks ran the program's overlay with no format
  for them, and the core hung (four float32 copies into a bfloat16 pack, on a
  Wormhole chip).

At load the plan refuses (`ProgramError`) programs whose own overlays reach the
programs' numbers, and one slot on a kernel with own parts: a program's part runs from
one slot while it loads its own overlay into another. An add with no number left is
refused with the counts.

On the card ([overlay_shift](../../experiments/code/overlay_shift/README.md)): first
(times 1, xor 2) and second (plus 1, the run's 3) on 120 cores, every core holding both,
and one overlay added to the live run (4). Each program's TAG named its overlays in its
own numbers: 16 commands, 0 wrong in the result, the number named and the shift.
A command took 4.04 to 4.22 µs after one of its own program and 5.5 to 5.7 µs after the
other program's: that is the program's switch.

A program of the kernel form in such a run would need:

1. its kernels to be the runtime's (a loop that gives the program field's overlay the
   command), or the runtime's loop compiled around its body;
2. its buffers in one L1 layout with the other programs', and the overlay slots after
   them;
3. its own overlays numbered as above, which its kernels name without a shift today;
4. kernel text for the overlay path beside its own code (the decode program has 16 B
   left);
5. the dispatcher's per-core program table to hold it like any other program field.

## Not in this version

- The dynamic handlers of two programs in one run.
- Code added to a live run over islands.
- A message with a body in a run that has X280 cores as ranks: a message carries a
  [body](../../examples/23_message_body/README.md) between Tensix cores, and X280
  cores are [ranks](../../examples/24_thread_rank/README.md) of a run whose messages
  are one line.
- A switch of program in a run where every core holds every program.
- A load issued by a cast or a pull: the host issues `run.load`, an X280 thread
  `kmx_load`.
- A switch of a dynamic program.
- What the old program kept in its buffers, which a switch loses.

The experiments these numbers stand beside are
[overlay_costs](../../experiments/code/overlay_costs/README.md),
[overlay_probe](../../experiments/code/overlay_probe/README.md) and
[remote_load](../../experiments/code/remote_load/README.md).
