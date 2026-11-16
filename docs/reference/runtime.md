---
title: Tensix runtime reference
summary: The device headers of the persistent kernel on the Tensix cores; the three kernels' classes and their calls, the command, every timed wait and its limit, the gave-up and where codes, the control block map, and what a program may and may not do.
tags:
  - reference
  - device
  - runtime
---

# Tensix runtime reference

Status: implemented. Runs on one Blackhole card under the dispatcher
([01_mapcollect](../../examples/01_mapcollect/)) and under the test feeder. This page
describes the headers in `src/komorebi_native/include/komorebi/`. The design behind
them is in [the Tensix runtime](../concepts/runtime.md).

A program has three persistent kernels on every core it uses: a reader and a writer on
the core's two data-movement CPUs, and the compute kernel on the three TRISC threads.
The easy layout runs the reader on RISCV_0 with NoC0 and the writer on RISCV_1 with
NoC1, but the two are roles, not "the CPU that reads" and "the CPU that writes": either
CPU can read, write and send to other cores. What differs is the NoC each uses, that a
circular buffer has one CPU filling it and the other draining it, and that the writer
reports the command. The runtime owns each kernel's loop: it takes the next command, and
the program supplies the body that runs it, usually a switch on the handler.

```c++
#include "komorebi/device/km_worker.hpp"

void kernel_main() {           // reader.cpp
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        switch (km::handler(command)) {
            case kHandlerMap:
                if (!reader.reserve(kCbIn, tiles)) return;  // gave up: the tile is stopping
                ...                                         // read the tiles, then cb_push_back
                break;
            default:
                break;
        }
    });
}

void kernel_main() {           // writer.cpp: the body says whether to report
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMap) return true;
        if (!writer.wait_front(kCbOut, 1)) return false;  // nothing to report
        ...                                           // write, flush, cb_pop_front
        return true;
    });
}

void kernel_main() {           // compute.cpp: the body runs on all three threads
    km::Compute compute;
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMap) return;
        bool ok = compute.wait_front(kCbIn, 1);  // the same answer on all three threads
        ...
    });
}
```

## Headers

| Header | What it holds | tt-metal headers |
|---|---|---|
| `abi/km_abi.h` | the command, flags, reserved handlers, the control block offsets, events (plain C, shared with the X280 and the host) | none |
| `device/core/km_command.hpp` | command fields, sequence order, `in_group`, `partition`, `report_to`, the ring scan | none |
| `device/core/km_core.hpp` | runtime buffers, state words, gave-up and where codes, start-up arguments, the clock, `poll_stop`, `command_gave_up`, `small_write`, `km::Core` | yes |
| `device/km_worker.hpp` | `km::Reader`, `km::Writer`, `km::Compute` | yes |
| `device/core/km_completion.hpp` | `km::complete`: a core's completion report | yes |
| `device/core/km_trace.hpp` | `km::Trace`: per-command timestamps | yes |
| `device/core/km_group.hpp` | groups of a group, `local()` and `leaders()`, reduce schedules, the arrival buffer ([ports](ports.md)) | none |
| `device/core/km_island.hpp` | the island table, rank resolve, the island questions ([islands](#islands)) | none |
| `device/movement/km_port.hpp` | emit, arrive, take, reduce ([ports](ports.md)) | yes |
| `device/movement/km_broadcast.hpp` | broadcast ([broadcasts](broadcast.md)) | yes |
| `device/movement/km_movement.hpp` | both data movers on one stream: patterns, buffer pairs, streams, `km::move`, `km::Movement` ([movement](movement.md)) | yes |

The headers without tt-metal can be compiled on the host; the CPU tests do so.

## The command

One 64-byte line of four 16-byte chunks (`km_command_t`). Word 0 of every chunk
carries the sequence number. The command has landed when all four agree
(`KM_LANDED`), and a core runs only a command that has landed.

| Word | Field | Read with |
|---|---|---|
| `sequence` | sequence number, from 1, increasing (compared across the wrap) | `km::newer(sequence, other)` |
| `hdr` | `handler \| flags << 8 \| slot << 16 \| program << 24` (the program field: 0 is the run's first program) | `km::handler(command)`, `km::flags(command)`, `km::slot(command)`, `km::program(command)` |
| `group` | `first \| count << 16`: cores `first` to `first + count - 1` in core order | `km::first(command)`, `km::count(command)`, `km::in_group(command, me)` |
| `a0` to `a9` | the handler's ten argument words; with a block, `a7` and `a8` name it ([the block](#the-block)) | `command.a0` ... `command.a9` |
| `land1` to `land3` | copies of `sequence` (landing) | `KM_LANDED(&command)` |

| Handler | Meaning |
|---|---|
| 1 to 239 (`KM_HANDLER_USER_MAX`) | the program's handlers |
| 0 (`KM_HANDLER_NOP`) | no work; passed to the bodies like any other |
| 255 (`KM_HANDLER_EXIT`) | every core leaves its kernels; completes as a group of all workers, whatever group it names |
| 250 to 254 | commands to the dispatcher; never reach a core |

| Flag | Meaning |
|---|---|
| `KM_FLAG_EXCLUSIVE` (2) | never in flight together with a command whose group overlaps (`km::groups_overlap`). Required for commands that use ports or broadcasts |
| `KM_FLAG_RESULT` (4) | the leader acknowledges with `{ack, result}`; see `Writer::result` |
| `KM_FLAG_QUEUE` (8) | dispatcher: the group allocator chooses `first` |

A command's header names its ring slot. A command found in another slot is not run and
is counted in `kStSlotWrong`. Other helpers in `km_command.hpp`:

- `km::partition(items, index, count)` gives `{start, len}`, rank `index`'s contiguous
  part of `items` items. The first `items % count` cores take one item more.
- `km::report_to(command, me, row)` gives `{parent, want}`, the completion tree described below.

## The block

A command has ten argument words. For more, it carries a block: up to 1,024 bytes
(`KM_BLOCK_BYTES`) in DRAM that its handlers read into L1, once per command and
kernel. Two argument words name it:

| Word | Meaning |
|---|---|
| `a7` (`KM_BLOCK_ADDRESS_WORD`) | its address in a DRAM bank, a multiple of 64 |
| `a8` (`KM_BLOCK_SIZE_WORD`) | its bytes \| bank field << 16 (`KM_BLOCK_BANK_SHIFT`). Field 0: a copy at that address in every bank, and a core reads bank `me % 8`; field b + 1: one copy, in bank b |

The host's `run.submit(..., block=)` writes a copy to every bank (field 0,
[host](host.md#run)); an X280 thread names bytes of its own region with `kmx_block`
(one copy, in its cluster's bank, [threads](threads.md#the-block-of-a-command)).
`reader.block(command)` and `writer.block(command)` return a `km::Block`: `words`, the
block in that kernel's area of `KM_CONTROL_BLOCK`, and `bytes`. The first call for a
command reads the block (whole 64-byte lines) and waits for it; later calls for the
same command return it at once. `bytes` is 0 for a command without a block (`a8` 0),
and also for one over 1,024 bytes, an address off a 64-byte line or a field over 8:
the handler decides what that means. Compute has no block; the reader passes what it
needs through a buffer. A read costs about 1 microsecond per command (64 words, a
command alone on 2 to 120 cores: 2.7 to 3.0 against 1.8 to 2.0 microseconds from
submit to acknowledgement).

## The three kernels

### `km::Core` (reader and writer)

| Call | What it does |
|---|---|
| `args()` | the start-up arguments (`km::Args`, below) |
| `me()` | this core's index in the core order |
| `arg(index)` | the program's start-up argument `index` (runtime argument `kArgs + index`) |
| `control(offset)`, `state()` | a word of the control block, the state words |
| `stop()` | the tile's stop flag is set (call `invalidate_l1_cache()` first) |
| `core_noc(index, address)` | the NoC address of `address` on core `index` of the core table |
| `disp_noc(address)` | the NoC address of `address` at the dispatcher |
| `poll(ready, limit, grace = 0)` | `km::poll_stop` on this tile's stop flag; records nothing, the code is the caller's |
| `wait(ready, where = kWhereData, line)` | `poll_at(where, ready, args().wait_limit)` that records: on any code but `kGaveNone` the give-up (this kernel's word, `where`, the command's slot and sequence), the caller's source line in live word `kLvLine`, and the stop flag; false then ([your own waits](#your-own-waits-and-copies)) |
| `gave_up_at(code, where, line)` | the same record for a check of the program's own: false |
| `gave_up(word, code, where, slot, sequence)` | `km::command_gave_up` for this tile (`kStReader` or `kStWriter`) |
| `block(command)` | the command's block in this kernel's L1 (`km::Block`: `words`, `bytes`), read from DRAM at the first call for the command ([the block](#the-block)) |

### `km::Reader`

| Call | What it does |
|---|---|
| `run(body)` | starts up, then calls `body(command)` for each command of this core until EXIT or a wait gives up. The body returns nothing |
| `reserve(cb, pages)` | timed `cb_reserve_back` of a contiguous item. False: the reader gave up (recorded) |
| `wait_stop(command)` | waits only for the stop flag, up to the idle limit, and commands a give-up at `stall`. True on stop. For tests that make a core stall |

The reader's start-up runs once and in this order:
1. Clears the ring, the DONE counters, the state words, the stop flag, the broadcast areas, the ports and the two block areas.
2. Loads the core table.
3. Checks in: each core of the group writes to its leader. A leader waits for its cores, then writes READY to the dispatcher.
4. Sends the hello, if a mailbox is given.

Then for each command it waits for the two token pages and scans the ring. It copies
the command into `KM_CONTROL_JOB` and hands a token to compute and to the writer.

### `km::Writer`

| Call | What it does |
|---|---|
| `run(body)` | per command: `body(command)` returns whether to report. The writer then waits for its writes (`noc_async_write_barrier`), waits until compute has left its body for this command, and runs `km::complete`. An EXIT command completes, then the writer leaves |
| `wait_front(cb, pages, times = 1)` | timed `cb_wait_front` of a contiguous item, timeout `times` wait limits. False: gave up (recorded) |
| `result(value)` | the command's result word: sent with the leader's ack when the command has `KM_FLAG_RESULT`, 0 if never called. On other cores of the group it is not sent |

A body that returns false reports nothing, so the command does not complete. A body
returns false only after a give-up, when the tile is stopping anyway. A handler the
writer has nothing to do for returns true; otherwise its group never completes.

### `km::Compute`

| Call | What it does |
|---|---|
| `run(body)` | per command: all three threads copy the command and call `body(command)`. After the body, unpack marks the command done in its DONE slot, which the writer waits for |
| `me()`, `arg(index)` | this core's index; the program's compute argument `index` (after `kComputeArgs`) |
| `wait_front(cb, pages, times = 1)` | timed `cb_wait_front`. Unpack decides and passes the answer to math and pack, so all three threads take the same path |
| `reserve_back(cb, pages)` | timed `cb_reserve_back`, decided by pack. False on pack only: gave up (recorded); true on unpack and math |

### Groups and completion (`km_completion.hpp`)

`km::complete(core, command, result)` runs in the writer after the body. Within one grid
row of `row` cores (or with `row` 0), every core of the group reports to the first core. Across
rows, cores report to their row's leader, and row leaders report to the first
core. A core that receives reports waits (with a timeout) for `want` of them in
`DONE[slot]` and clears it. A core other than the first then increments its parent's
`DONE[slot]` with an atomic. The first core writes the ack word, and with
`KM_FLAG_RESULT` also the result, into the dispatcher's ack line of the slot. Nothing is
acknowledged once the tile's stop flag is set.

### Start-up arguments

Reader and writer: runtime arguments 0 to `kArgs - 1` (19 words, `km::Args`), then
the program's own.

| Word | Meaning |
|---|---|
| `control` | control block address, the same on every core (a multiple of 16) |
| `me` | this core's index in the core order |
| `cores` | DRAM address of the core table: core index to NoC `x \| y << 16` |
| `dispatcher_coordinates`, `ack_base` | the dispatcher's NoC x, y and its ack words (slot j at `ack_base + 16 j`) |
| `mailbox` | the hello's address at the dispatcher; 0 sends none |
| `leader`, `rank`, `peers` | start-up group: leader's index, this core's check-in slot, cores the leader waits for |
| `ready_slot`, `ready_base` | the leader's READY word at the dispatcher |
| `tag` | per-process random tag, 16 bits; check-ins carry it |
| `workers` | cores 0 to `workers - 1` run commands (the group of an EXIT) |
| `depth` | ring slots scanned, 1 to 64 (`KM_RING_SLOTS`; 0: 64) |
| `row` | cores per grid row for completion; 0: one level always |
| `wait_limit` | timeout of a wait for another core, in 1024-cycle units |
| `idle_limit` | timeout of a wait for the next command, in 1024-cycle units; 0: none |
| `trace`, `trace_cap` | DRAM address of the trace (a page per core; 0: none), sequences traced |

Compute: `kCArgControl`, `kCArgMe`, `kCArgWait`, `kCArgIdle` (4 words), then the
program's. The host gives the limits in seconds: `session.load(..., wait=2.0,
idle=120.0)` becomes `abi.limit_units(seconds, mhz)`.

## Timed waits

Every wait of the runtime is a `km::poll_stop(stop, ready, limit, grace)`. It polls
`ready()` and returns `kGaveNone` when it holds. It returns `kGaveLimit` after
`limit` units (0: no limit). It returns `kGaveStop` when the stop flag is set: at once
with `grace` 0, otherwise after `grace` more units, so a wait already about to end
can still end. A wait that gives up records its first give-up and sets the tile's stop
flag, so all three kernels leave. The writer reports a command only after the reader's
and compute's bodies for it have returned, so a give-up in any of a core's three
kernels fails the command it happens in. The one exception is a command the dispatcher
gives up at its deadline ([giving up one command](#giving-up-one-command)).

While it waits, each kernel keeps the wait's where code in its live word, so an inspect
can name it while the run goes on.

| Kernel | Wait | Limit | Where |
|---|---|---|---|
| reader | a leader for its cores' check-ins (start-up) | wait | `checkin` |
| reader | free token pages for the next command | idle; wait after a stop | `token` |
| reader | the next landed command of this core | idle | `next` |
| reader | `reserve` | wait | `data` |
| reader | `wait_stop` (the stop flag only, not the abort word) | idle | `stall` |
| reader | after a give-up: token pages for the EXIT it hands on (not recorded) | wait | - |
| writer | the next token | idle; wait after a stop | `writer` |
| writer | `wait_front` | wait x `times` | `data` |
| writer | compute and the reader have left their bodies for this command | wait | `cdone` |
| writer | its cores' completion reports | wait | `complete` |
| compute | the next token (unpack) | idle; wait after a stop | `ctoken` |
| compute | `wait_front` (unpack) | wait x `times` | `cinput` |
| compute | `reserve_back` (pack) | wait | `coutput` |
| reader | `km::arrive`: a port's count | wait | `port` |
| writer | `km::emit`: write and atomic flush | wait | `emit` |
| writer, compute | a wait behind a port's wait (`reduce_result`, `take`) | 2 x wait (`km::kBehindPort`) | `data`, `cinput` |
| reader or writer | broadcast calls | wait | `broadcast*` ([broadcasts](broadcast.md)) |
| reader or writer | the runtime's own NoC barriers (`Core::flushed`) | wait | `nocflush` |
| compute (math, pack) | unpack's word: the command's verdict, a `wait_front`'s | none | `cmailbox` |
| compute (unpack) | math's and pack's confirmations | none | `cconfirm` |
| reader, writer, unpack | after a give-up: the producer's pages to clear | wait | `clear` |

The runtime's own barriers (start-up, check-in, command block, completion, trace)
are `Core::flushed`: timed, and polled with the stop flag. A program's
`noc_async_read_barrier`, `noc_async_write_barrier`, `noc_async_writes_flushed` and
`noc_async_atomic_barrier` have no timeout. They end once the transactions to valid
addresses complete; the buffer wrappers' checks keep those addresses valid. A
program's own waits go through `poll` and `gave_up` in the same way.

Math and pack wait for unpack's words without a limit: unpack's own waits are timed,
and unpack tells them the outcome. Their live word shows `cmailbox` meanwhile, and
unpack's shows `cconfirm` while it waits for theirs.

### Your own waits and copies

In a mover's C++ of your own (a `km.cxx` body, a header it includes), wait through
`core.wait(ready)`, not `core.poll`, unless you handle the code yourself. `poll`
returns the code and records nothing: a body that drops it and returns false leaves
no give-up word and does not set the stop flag, so the rest of the group waits for the
command's deadline and the core's row names no wait
([a wait whose code is dropped](hazards.md#a-wait-whose-code-is-dropped)). `wait`
shows `where` (`data` by default) in the live word while it polls and, on a timeout,
the stop or the deadline, records the give-up with the line of the call: `run.inspect`
and a timeout's core rows print `writer gave up (limit) in data at line 1057`.

A copy inside L1 goes through `km::copy_l1(core, to, to_cb, from, from_cb, words)`
rather than a hand loop: with the guard on it checks that each range lies inside its
buffer's ring and gives up `kGaveBadItem` at the caller's line when one does not.
Both cost nothing on the path that succeeds beyond the wait or the copy itself.

## Gave-up and where codes

A gave-up word is `code | where << 8 | slot << 16`, and its sequence number is in the
next word. For `port` and `emit` the top byte holds the port. Only the first give-up
of each kernel is kept, because it names the cause. The host decodes the words with
`komorebi.session.state.gave_up`; the names below are the ones it prints.

| Code | Name | Meaning |
|---|---|---|
| 0 | - | did not give up |
| 1 (`kGaveStop`) | `stop` | the stop flag was set |
| 2 (`kGaveLimit`) | `limit` | the wait timed out |
| 3 (`kGaveBadItem`) | `bad_item` | an item of several pages would pass the buffer's end, or a buffer is too small for its schedule; nothing was accessed |
| 4 (`kGaveBadBroadcast`) | `bad_broadcast` | a broadcast's arguments do not fit the command |
| 7 (`kGaveDeadline`) | `deadline` | the dispatcher gave the command up at its deadline: the run goes on |

| Where | Name | Kernel: waiting for |
|---|---|---|
| 1 | `checkin` | reader (leader): its cores' start-up check-ins |
| 2 | `token` | reader: free token pages |
| 3 | `next` | reader: a command |
| 4 | `stall` | a handler waiting for the stop flag (tests) |
| 5 | `complete` | writer: its cores' completion reports |
| 6 | `writer` | writer: a token |
| 7 | `data` | reader or writer: a buffer wait |
| 8 | `ctoken` | compute (unpack): a token |
| 9 | `cinput` | compute (unpack): input pages |
| 10 | `coutput` | compute (pack): output space |
| 11 | `cdone` | writer: compute's and the reader's bodies to return |
| 12 | `broadcastbegin` | broadcast: arguments or buffer at a command |
| 13 | `broadcastdone` | broadcast sender: DONE markers, its NoC flushes |
| 14 | `broadcastvalid` | broadcast receiver: VALID |
| 15 | `broadcastfree` | broadcast: its compute's pops, its last markers |
| 16 | - | free |
| 17 | `port` | reader: a port's count (`km::arrive`) |
| 18 | `emit` | writer: a put or a bump to flush (`km::emit`) |
| 45 | `nocflush` | reader or writer: one of the runtime's NoC barriers |
| 46 | `cmailbox` | compute (math or pack): unpack's word |
| 47 | `cconfirm` | compute (unpack): math's and pack's confirmations |
| 48 | `clear` | after a give-up: a buffer's producer to finish the command |

How each kernel left is one byte of `kStEnds` (reader, writer, compute):
1. `kEndExit`: left on an EXIT command.
2. `kEndGaveUp`: left after its own wait gave up.
3. `kEndAbort`: left on the EXIT the reader hands on after a give-up.

### State words

144 words (`KM_STATE_WORDS`, 2 per ring slot and 16) from `KM_CONTROL_STATE`, read
back after the stop (`komorebi.session.state`). Counters saturate. The words after the two
per-slot parts are counted from `kStReader` (128).

| Word | Meaning |
|---|---|
| 0 to 63 (`kStLast`) | last sequence consumed, per ring slot |
| 64 to 127 (`kStReported`) | last sequence this core reported, per slot |
| 128, 129 / 130, 131 / 132, 133 | gave-up word and its sequence: reader (`kStReader`) / writer / compute |
| 134 `kStRuns`, 137 `kStCompleted`, 138 `kStSeen` | commands the reader took, the writer finished, compute took |
| 135 `kStPasses`, 136 `kStNextCycles` | ring passes and cycles / 16 spent waiting for commands |
| 139 `kStWrong` | bytes unpack, math, pack: job copies that did not match the token |
| 140 `kStEnds` | how each kernel left |
| 141 `kStMagic` | `0x4B000000 \| tag << 8 \| me`: the reader started with this tag |
| 142 `kStSlotWrong` | commands not run: found in a slot their header does not name |
| 143 `kStTraceDropped` | commands past the trace's capacity |

## Control block

Every core has one control block in L1, at the same address everywhere (`args().control`,
a multiple of 16). Offsets are in bytes, and every offset has a static assert in
`km_abi.h`. The DONE, state and ring parts follow `KM_RING_SLOTS` (64); every offset
is derived from the one before. The ring is last, so the words the kernels use most sit
at small offsets.

| Offset | Name | Contents |
|---|---|---|
| 0x000 | `KM_CONTROL_RESERVED` | reserved, 16 bytes |
| 0x010 | `KM_CONTROL_STOP` | stop flag: nonzero makes every wait give up |
| 0x040 | `KM_CONTROL_DONE` | 64 slots of 16 bytes: word 0 counts completion reports, word 1 (`kDoneCompute`) is the last sequence whose compute body returned, word 2 (`kDoneReader`) the last whose reader body returned |
| 0x440 | `KM_CONTROL_BOOT` | start-up check-ins at a leader, one 16-byte slot per rank |
| 0x540 | `KM_CONTROL_STATE` | 144 state words |
| 0x780 | `KM_CONTROL_JOB` | the command being run, copied by the reader for compute and writer |
| 0x7C0 | `KM_CONTROL_OUT` | 4 source slots of 16 bytes for small NoC writes: check-in, READY, hello, ack |
| 0x800 | `KM_CONTROL_CORES` | core table: 128 words, core index to NoC `x \| y << 16` |
| 0xA00 | `KM_CONTROL_PORT` | 12 port slots of 16 bytes: count, wanted count; ports 0 to 7 a kernel names ([ports](ports.md)), 8 and 9 (`KM_PORT_PAGES`) the reader's and the writer's page calls' ([islands](islands.md)), 10 and 11 (`KM_PORT_LEFT`, `KM_PORT_RIGHT`) the collectives' words: an island's ranks' meetings and the leader's release (`KM_PORT_MET`, `KM_PORT_GO` of 10), an all-reduce's 1 (the count of 11) |
| 0xAC0 | `KM_CONTROL_BROADCAST` | 2 broadcast areas of 0x140 bytes: VALID, out slot, give-up, statistics, DONE-BY words ([broadcasts](broadcast.md)) |
| 0xD40 | `KM_CONTROL_MESSAGES` | routed messages: the tail as last read, head, sent, consumed, quota, idle words, status-line slot, first give-up, reservation answer ([dynamic](dynamic.md)) |
| 0xE00 | `KM_CONTROL_TAILS` | routed messages on a counter tile: its READY line, two read slots, and one 16-byte tail line per core ([counter tiles](dynamic.md#counter-tiles)) |
| 0x1640 | `KM_CONTROL_X280` | X280 port: the X280 core's tell and answer, the core's out slot, its X280 core's tile and mailbox ([X280 ports](ports.md#x280-ports)) |
| 0x1680 | `KM_CONTROL_BLOCK` | 2 block areas of 1,088 bytes (`KM_BLOCK_AREA_BYTES`), the reader's then the writer's: a line whose word 0 is the sequence of the command read in, then up to 1,024 bytes of its block ([the block](#the-block)) |
| 0x1F00 | `KM_CONTROL_ABORT` | 64 abort words, one per slot: the sequence of the slot's command the dispatcher gave up |
| 0x2000 | `KM_CONTROL_LIVE` | live words (`km::kLv*`, 512 bytes): per kernel its wait's where code, its command's sequence and slot, the commands it has left; the trace's address and capacity, the commands given up; then what a give-up clears ([live words](#live-words)) |
| 0x2200 | `KM_CONTROL_ISLANDS` | the island area, 192 bytes: a 32-byte header (this island, the islands, its first rank, its ranks, its first core of the card), then the island table ([islands](#islands)) |
| 0x22C0 | `KM_CONTROL_RING` | 64 ring slots of one command each (64 bytes) |
| 0x32C0 | `KM_CONTROL_BYTES` | size of the control block |

### Live words

Each word has one writer: kernel r (`kRoleReader` 0, `kRoleWriter`, `kRoleUnpack`,
`kRoleMath`, `kRolePack` 4) writes word `kLv*` + r. The dispatcher's inspect reads the
first 128 bytes (`KM_LIVE_READ`) while the run goes on; `komorebi.session.state.live` decodes
them.

| Word | Meaning |
|---|---|
| 0 + r (`kLvWhere`) | the runtime wait kernel r is in, a where code; 0: none |
| 5 + r (`kLvSequence`), 10 + r (`kLvSlot`) | the command it runs and its slot; sequence 0: none |
| 15 + r (`kLvFinished`) | the commands it has left, a count |
| 20 (`kLvTrace`), 21 (`kLvTraceCap`) | the trace's L1 address (0: none) and the sequences it holds |
| 22 (`kLvGivenUp`) | the commands this core gave up (the writer's count) |
| 23 (`kLvLine`) | role << 16 \| line of the last `core.wait`, `gave_up_at` or `km::copy_l1` that gave up since the load; 0: none |
| 24 to 26 (`kLvProduces`) | per producer (reader, pack, writer), the buffers it has filled |
| 32 on (`kLvEnds`) | the pages each producer pushed by the end of a command, by its parity |

Runtime buffers: circular buffers 29 and 30 carry the tokens (`kCbTokenCompute`,
`kCbTokenWriter`, 64 bytes each). Buffer 31 (`kCbTrace`) holds the trace when tracing
is on. Buffer 32 (`kChainBuffer`) holds a chain's words in a program that runs chains.

## Islands

An island is a set of cores that reach the others only through links; its table entry
names its ranks. Ranks are flat and contiguous by island, so a group word `first |
count << 16` names flat ranks, and a core's rank is its island's first rank plus its
local core, its index in the core table less the island's base (`KM_ISLAND_BASE`: the
island's first core of the card, 0 unless the card is split into islands; `Islands::offset()`
is the rank of core 0). `me()` of `km::Core` (reader, writer, movement)
and of `km::Compute` is that flat rank, so `me() - km::first(command)` is the rank in
the group on every island; the core table and every NoC lookup take the core
(`core_of(rank)` turns a rank of this island into it). The dispatcher holds its island's
first rank, count and base (`kmd_args.island`: first | count << 16 | base << 32, 0 on one
island) and cuts each group by them: it takes a group with a part on its island, places
the line unchanged over its island's rows, and keeps the part (in cores, from the base)
for its overlap, program and completion checks; a group with no
part there is rejected as outside (reason 3). A command reaches every core of the
island; the cores of the group's part there run it, and the part completes through the
first of its cores in the group (`km::local_part`), so an event's first core is a local
core of the island whose dispatcher sends it. A message's destination is a flat rank of 16 bits;
the writer turns it into a local core. A dynamic run's ranks, counter tiles and end test
are flat as well (its counter tiles are the group's last ranks); queues, tail lines and
status lines are addressed by local core, so a run on the group (first rank, n) of an
island behaves as (0, n) does on one island. Engine streams and graphs check a group
against the island's ranks, and `Run` checks programs held on the group's part here. The host writes the island area at load: on one
island the header is island 0 of 1 with every core a rank, and every answer below is
the one of a run without islands.

| Call | Answer | On one island |
|---|---|---|
| `km::island()`, `km::islands()` | this core's island, the islands of the run | 0, 1 |
| `km::island_of(rank)`, `km::crosses(rank)` | the island that holds `rank`; whether a send to it crosses a link | 0, false |
| `km::resolve(rank)` | its island, then the local core (this island) or this island's next link toward it | `{0, true, rank}` |
| `km::hops(i)`, `km::neighbours()` | links crossed to island `i`; the islands one link away, a bit each | 0, 0 |
| `km::rate(i)` | bytes a second toward `i`: `km::kRateLocal` here, 0 (not known) for another (the table holds no rate) | `kRateLocal` |
| `group.local()`, `group.leaders()` | the group's part on this island; the first rank of each island part | the group; its first rank |
| `km::partition(items, part, parts)` | part `part` of `parts` of `items` | `partition(items, 0, 1)`: all items |

A table entry is 8 bytes, little-endian: first rank (16 bits), ranks (16), next link (8,
`KM_ISLAND_NO_LINK` 0xFF in the holder's own entry), hops (8), zero (16), as the host's
`komorebi.islands.graph.ENTRY`; `KM_ISLANDS_MAX` (16) entries. On an X280 thread,
`kmx_island()` and `kmx_islands()` answer the same (0 and 1 on one island).

The calls that act by island (`km_group.hpp`, `km_port.hpp`):

| Call | What it does | On one island |
|---|---|---|
| `km::reduce_arrive`, `km::reduce_emit`, `km::reduce` on a group over islands | each island's part reduces as a group of its own (owner or tree), then each part's first rank sends its fold once to the group's rank 0: by owner into the slot after its island's (port 0), by tree on the one port after the root part's rounds, part j into slot j - 1 after the root part's (rank 0 waits there for all of them); where the port or the slots do not fit, every core gives up (`kGaveBadItem`); the calls do not change | the flat schedule |
| the same with `km::local` (`reduce_emit(..., km::local)`, `reduce_arrive(..., km::local)`, `reduce(compute, arrival, group, form, km::local, combine)`) | the reduce over `group.local()`: it stays in this island | the whole group |
| `group.leader(j)` | the first rank of island part `j` of the group | `group.leader(0)`: its first rank |
| `km::to_island(i).put(writer, command, arrival, cb, pages, rank, port, slot)` | `km::emit` to `rank` after a check that island `i` holds it (else a give-up, bad item) | a local emit |
| `km::emit` to a rank of another island | by `km::write_l1`: the pages reach the rank's arrival slot and its port counts 1, so its `arrive` waits as for a local sender | never crosses |
| `km::Broadcast` on a group over islands (`km_broadcast.hpp`) | a broadcast an island part: the sender casts in its part and sends each block once by `km::write_l1` to the next part's first core, whose `arrive` takes it and casts in its part as the sender does (its kernel sees a receiver); that core's popped blocks and taken words go back the same way as its row in the sender's DONE-BY table; the calls do not change | a broadcast of its own |
| `km::Broadcast(reader_or_writer, command, arg0, km::local)` | each island part is a broadcast of its own: a part's first core is its sender and fills the blocks itself | the same broadcast |

Across, both calls are `km::write_l1` (`km_far.hpp`): the pages go from the sender's L1
into the rank's arrival slot (a broadcast's block, its VALID line, its marker back into
the next part's or the sender's L1), through the link cores toward the rank's island, any
number of links away, and the call returns once they landed (the port's 1 after them).
A sender's wait is bounded and gives its command up (`emit`, the broadcast's own give-up).

A broadcast over islands takes three more `BroadcastArgs` words, which
`komorebi.layout.island_broadcast_args` computes (`broadcast_args` gives none of them a
part): `next`, the flat rank of the next part's first core (`kBroadcastNoNext`: none);
`fed`, that core's row in its feeder's DONE-BY table (0: not fed); `from`, the feeder's
flat rank. The layout stands the islands' grids one below the other or side by side; a
column, or a row across an island's edge (a row still splits at each panel inside an
island), that spans islands is a part an island. Its markers that cross carry sequence 0,
since each island numbers its commands. A plan respects three things: a part followed by
another holds at most 15 cores (its DONE-BY table keeps a row for the next part); a
broadcast enters each island once; every island has the same grid shape, and the session
checks that the control blocks sit at one address (a block lands at its sender's slot and
area addresses). It is built with the link cores serving (`KM_PAGE_LINKS`); without them
a core given a `next` or `fed` word gives up (`kGaveBadBroadcast`).

## Trace

With a trace address, each core keeps 4 wall-clock words per sequence number, for
sequences 1 to `trace_cap`:
- `kTraceSeen`: the reader took the command.
- `kTraceDone`: the writer's outputs were written.
- `kTraceSent`: completion was sent.
- `kTraceExtra`: the reader's ring passes for this command.

The writer writes the page to DRAM when it leaves. Commands past the capacity are
counted in `kStTraceDropped`.

## Giving up one command

At a command's deadline the dispatcher can give up that one command and keep the run.
This applies to a program's handler from any source (the host, a stream or a graph of
the engine, an X280 thread, the other island), outside a dynamic run: the timeout
event goes to that source (a thread's `kmx_wait` returns it; a stream or graph ends
with its deadline error, the run going on). The
dispatcher first reads the group's state and live words (an inspect), then writes the
command's sequence into its slot's abort word on every core. On each core:

1. Every wait that polls the abort word (`poll`, `poll_at`, the buffer waits, the
   tokens, `km::move`'s passes with no block to move: live where `move`) returns
   `kGaveDeadline` once the word holds the command it runs. The handler
   returns through `gave_up`, as on any other give-up, but the tile's stop flag stays
   clear.
2. The kernels leave the body. The writer does not report the command, and the leader
   clears its slot's completion count.
3. The reader, the writer and unpack each pop the pages their command's producer left in
   the buffers 0 to 28, once that producer has finished the command (`clear`). Then
   each takes the next command.

The dispatcher then reads each core's live sequence words until no kernel of the group
runs the command. Only then does it free the slot and send `KM_EVENT_TIMEOUT`, which
the host reads as given up (the count in the core block's `KMD_H_GIVEN_UP` rose). The
state word records `deadline` with the where code of the wait. The writer counts the
command in `kLvGivenUp`. The next command on the same cores runs as usual.

Where the line is: a core that still runs the command 20 ms after the abort
(`KMD_GIVE_UP_TICKS`) ends the run as before, with `KMD_ERROR_DEADLINE` and the cores
that did not report. That is a wait that polls nothing: a program's own loop, a NoC
barrier of a program's, the DST hand-over between math and pack, a compute unit that
never returns. `wait_stop` is one on purpose: it waits for the stop flag only, so the
tests of the deadline still end the run. A consumer that waits for more pages than its
command's producer sends can take the next command's pages if that command is already
in flight.

A kernel wait's limit is not a deadline. A wait that reaches the run's wait limit
(`wait=` of the load) before the command's deadline gives up with `limit`. Like every
give-up but the deadline's, that sets the tile's stop flag and ends the kernels (a kept
reset list loads them again). Only the deadline gives up the one command and keeps the
cores running. A load whose given-up commands should leave the cores in the run sets its
wait limit above its deadline.

A program built with `guard=False` (`KM_GUARD` 0) has none of this. Its kernels read
no abort word, keep no live words and wait at a NoC barrier with no limit. The
dispatcher still reaches the deadline, reads the group and writes the abort word, but
finds in the live words no kernel running the command: it frees the slot and sends the
give-up while the cores stay in the wait, and resets only a core whose own wait limit
ended its kernels (and only with a kept list). `run.inspect` of its cores raises, and
the load's log says once that its commands are not given up at a deadline.

## What a program may and may not do

A program may:
- use handlers 1 to 239, the ten argument words, and its own start-up arguments
  (`arg(index)`);
- use any circular buffer except 29 to 31;
- wait on its own conditions with `wait(ready)` (or `poll(ready, args().wait_limit)`
  and a give-up with `gave_up(...)` of its own), under a where code of the runtime
  (`kWhereData`, `kWhereStall`) or of a call it uses;
- write to DRAM and to other cores' L1 through the NoC from the reader and the writer;
- send a result word with `writer.result(value)` on a `KM_FLAG_RESULT` command.

A program must:
- take and release pages of its buffers only through the wrappers (`Reader::reserve`,
  `Writer::wait_front`, `Compute::wait_front`, `Compute::reserve_back`) before the
  matching `cb_push_back` or `cb_pop_front`. A buffer is a multiple of its item, and
  every push and pop on it is a multiple of the item;
- give every wait of its own a timeout and check the stop flag, through `poll`;
- return from a body after a give-up (the wrappers return false) without touching the
  buffer it was waiting for;
- in compute, take the same path on all three threads: branch only on the command and
  on what `wait_front` returned;
- send a small write between two cores' L1 only with `km::small_write`, slot to slot.
  The two addresses must share their low four bits, or the write is silently wrong
  (under `KM_GUARD` the runtime's write calls refuse it by name; a raw
  `noc_async_write` is not checked);
- give commands that use ports or broadcasts `KM_FLAG_EXCLUSIVE`;
- fence (`invalidate_l1_cache()`) between the stores that build data in L1 and the
  `cb_push_back` or NoC send that hands it on: the core's stores pass through a small
  write-through cache, and a send that starts before they land carries old bytes.

A program must not:
- write the control block except through the runtime's calls;
- loop without a limit, or wait on a NoC barrier for an address it has not checked;
- write to the NoC from the compute kernel;
- return false from the writer's body except after a give-up or in a test that means
  the command to time out: the group then never completes.
