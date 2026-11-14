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
| `device/km_command.hpp` | command fields, sequence order, `in_group`, `partition`, `report_to`, the ring scan | none |
| `device/km_core.hpp` | runtime buffers, state words, gave-up and where codes, start-up arguments, the clock, `poll_stop`, `command_gave_up`, `small_write`, `km::Core` | yes |
| `device/km_worker.hpp` | `km::Reader`, `km::Writer`, `km::Compute` | yes |
| `device/km_completion.hpp` | `km::complete`: a core's completion report | yes |
| `device/km_trace.hpp` | `km::Trace`: per-command timestamps | yes |
| `device/km_group.hpp` | groups of a group, reduce schedules, the arrival buffer ([ports](ports.md)) | none |
| `device/km_port.hpp` | emit, arrive, take, reduce ([ports](ports.md)) | yes |
| `device/km_broadcast.hpp` | broadcast ([broadcasts](broadcast.md)) | yes |
| `device/km_movement.hpp` | both data movers on one stream: patterns, buffer pairs, streams, `km::move`, `km::Movement` ([movement](movement.md)) | yes |

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
| `a0` to `a9` | the handler's ten argument words | `command.a0` ... `command.a9` |
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
| `poll(ready, limit, grace = 0)` | `km::poll_stop` on this tile's stop flag |
| `gave_up(word, code, where, slot, sequence)` | `km::command_gave_up` for this tile (`kStReader` or `kStWriter`) |

### `km::Reader`

| Call | What it does |
|---|---|
| `run(body)` | starts up, then calls `body(command)` for each command of this core until EXIT or a wait gives up. The body returns nothing |
| `reserve(cb, pages)` | timed `cb_reserve_back` of a contiguous item. False: the reader gave up (recorded) |
| `wait_stop(command)` | waits only for the stop flag, up to the idle limit, and commands a give-up at `stall`. True on stop. For tests that make a core stall |

The reader's start-up runs once and in this order:
1. Clears the ring, the DONE counters, the state words, the stop flag, the broadcast areas and the ports.
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
can still end. A wait that gives up commands its first give-up and sets the tile's stop flag, so all three kernels leave. The writer reports a command only after the
reader's and compute's bodies for it have returned, so a give-up in any of a core's
three kernels fails the command it happens in.

| Kernel | Wait | Limit | Where |
|---|---|---|---|
| reader | a leader for its cores' check-ins (start-up) | wait | `checkin` |
| reader | free token pages for the next command | idle; wait after a stop | `token` |
| reader | the next landed command of this core | idle | `next` |
| reader | `reserve` | wait | `data` |
| reader | `wait_stop` | idle | `stall` |
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

The NoC barriers have no timeout: `noc_async_read_barrier`,
`noc_async_write_barrier`, `noc_async_writes_flushed` and `noc_async_atomic_barrier`.
They end once the transactions to valid addresses complete. The buffer wrappers'
checks are what keep those addresses valid. A program's own waits go through
`poll` and `gave_up` in the same way.

## Gave-up and where codes

A gave-up word is `code | where << 8 | slot << 16`, and its sequence number is in the
next word. For `port` and `emit` the top byte holds the port. Only the first give-up
of each kernel is kept, because it names the cause. The host decodes the words with
`komorebi.state.gave_up`; the names below are the ones it prints.

| Code | Name | Meaning |
|---|---|---|
| 0 | - | did not give up |
| 1 (`kGaveStop`) | `stop` | the stop flag was set |
| 2 (`kGaveLimit`) | `limit` | the wait timed out |
| 3 (`kGaveBadItem`) | `bad_item` | an item of several pages would pass the buffer's end, or a buffer is too small for its schedule; nothing was accessed |
| 4 (`kGaveBadBroadcast`) | `bad_broadcast` | a broadcast's arguments do not fit the command |

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

How each kernel left is one byte of `kStEnds` (reader, writer, compute):
1. `kEndExit`: left on an EXIT command.
2. `kEndGaveUp`: left after its own wait gave up.
3. `kEndAbort`: left on the EXIT the reader hands on after a give-up.

### State words

144 words (`KM_STATE_WORDS`, 2 per ring slot and 16) from `KM_CONTROL_STATE`, read
back after the stop (`komorebi.state`). Counters saturate. The words after the two
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
`km_abi.h`. The ring, DONE and state parts follow `KM_RING_SLOTS` (64); every offset
after them is derived from the one before.

| Offset | Name | Contents |
|---|---|---|
| 0x000 | `KM_CONTROL_RING` | 64 ring slots of one command each (64 bytes) |
| 0x1000 | `KM_CONTROL_RESERVED` | reserved, 16 bytes |
| 0x1010 | `KM_CONTROL_STOP` | stop flag: nonzero makes every wait give up |
| 0x1040 | `KM_CONTROL_DONE` | 64 slots of 16 bytes: word 0 counts completion reports, word 1 (`kDoneCompute`) is the last sequence whose compute body returned, word 2 (`kDoneReader`) the last whose reader body returned |
| 0x1440 | `KM_CONTROL_BOOT` | start-up check-ins at a leader, one 16-byte slot per rank |
| 0x1540 | `KM_CONTROL_STATE` | 144 state words |
| 0x1780 | `KM_CONTROL_JOB` | the command being run, copied by the reader for compute and writer |
| 0x17C0 | `KM_CONTROL_OUT` | 4 source slots of 16 bytes for small NoC writes: check-in, READY, hello, ack |
| 0x1800 | `KM_CONTROL_CORES` | core table: 128 words, core index to NoC `x \| y << 16` |
| 0x1A00 | `KM_CONTROL_PORT` | 8 ports of 16 bytes: count, wanted count ([ports](ports.md)) |
| 0x1A80 | `KM_CONTROL_BROADCAST` | 2 broadcast areas of 0x140 bytes: VALID, out slot, give-up, statistics, DONE-BY words ([broadcasts](broadcast.md)) |
| 0x1D00 | `KM_CONTROL_MESSAGES` | routed messages: the tail as last read, head, sent, consumed, quota, idle words, status-line slot, first give-up, reservation answer ([dynamic](dynamic.md)) |
| 0x1DC0 | `KM_CONTROL_TAILS` | routed messages on a counter tile: its READY line, two read slots, and one 16-byte tail line per core ([counter tiles](dynamic.md#counter-tiles)) |
| 0x2600 | `KM_CONTROL_X280` | X280 port: the X280 core's tell and answer, the core's out slot, its X280 core's tile and mailbox ([X280 ports](ports.md#x280-ports)) |
| 0x2640 | `KM_CONTROL_BYTES` | size of the control block |

Runtime buffers: circular buffers 29 and 30 carry the tokens (`kCbTokenCompute`,
`kCbTokenWriter`, 64 bytes each). Buffer 31 (`kCbTrace`) holds the trace when tracing
is on.

## Trace

With a trace address, each core keeps 4 wall-clock words per sequence number, for
sequences 1 to `trace_cap`:
- `kTraceSeen`: the reader took the command.
- `kTraceDone`: the writer's outputs were written.
- `kTraceSent`: completion was sent.
- `kTraceExtra`: the reader's ring passes for this command.

The writer writes the page to DRAM when it leaves. Commands past the capacity are
counted in `kStTraceDropped`.

## What a program may and may not do

A program may:
- use handlers 1 to 239, the ten argument words, and its own start-up arguments
  (`arg(index)`);
- use any circular buffer except 29 to 31;
- wait on its own conditions with `poll(ready, args().wait_limit)` and command a
  give-up with `gave_up(...)`, under a where code of the runtime (`kWhereData`,
  `kWhereStall`) or of a call it uses;
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
  The two addresses must share their low four bits, or the write is silently wrong;
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
