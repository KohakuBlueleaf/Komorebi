---
title: The chain
summary: Calls written once as records in DRAM and run as one command, each core going to its next record by itself; a record's input waits for marks from the cores that wrote it. The record, the device calls, the host and thread calls, the earlier and the direct input, what is refused and what it costs.
tags:
  - reference
  - device
  - host
---

# The chain

Status: implemented in the Tensix runtime and run on one Blackhole card: the decode
project's whole token (140 records a command) and
[29_chain](../../examples/29_chain/) (`python chain_steps.py`).

A chain is a list of calls of one program, each reading what the call before it wrote.
The host writes them once as **records** in DRAM and submits one command; every core of
the command walks the records by itself and runs the ones it is in. A record's input
does not wait for the host: the root of each group of the call before **marks** the
cores that read the block it wrote, once that block is written, and a core of the next
record waits until it has the marks it needs. A chain's calls run on one island's
cores.

From [29_chain](../../examples/29_chain/)'s script, where `calls` returns its four
`kc.Call`s:

```python
steps = calls(t, args.cores, args.width)
chain = run.chain(steps, C["kChainMode"])  # proven, then written once
run.wait(run.submit(chain, factor=5))  # one command
```

## What a program states

A program runs chains with three things of its own; the rest is the framework's.

| What | Where |
|---|---|
| a chain mode: a value of a4 that its handler takes as "this command is a chain" | the program's header; `run.chain(calls, mode)` |
| buffer index 32 (`kChainBuffer`), one page of at least 1,808 bytes: the chain's words and two record slots on each core | the program's buffers; the index is the runtime's, kept for the chain, so a chain's program is traced as any program |
| in each kernel, the chain command handed to the framework's walk, and in the step, the wait for its input and the marks after its output | the kernels (below) |

Each record names its handler, one of the chain's program: a core goes from one
handler to the next by itself (example 29: SCALE, SCALE, ADD, SUM). A record whose
handler field is 0 runs the chain command's handler (the decode program's five calls
are one handler's, its records keep 0). The chain command goes to the first call's
handler with the chain mode in a4, so every kernel hands a chain command of any of the
program's handlers to the walk.

## The record

16 words, 64 bytes, 64-byte aligned, at most 256 a chain (`komorebi/abi/km_chain.h`,
plain C for the device, the X280 and the host).

| Word | Holds |
|---|---|
| 0 to 9 | the call's a0 to a9 |
| 10 | its cores: first \| count << 16 |
| 11 | the info word (below) |
| 12 to 14 | three range words: base \| step << 12 \| width << 20 \| root only << 31 |
| 15 | the next record's address in the same DRAM bank (0: the last), its bits 0 to 5 the record's stamp row |

| Info bits | Holds |
|---|---|
| 0 | marked: the record's input waits for the marks of the record before's roots |
| 1 | sends: its parts go up its group's tree (the credits) |
| 2 | direct: its input is sent into its cores' L1 |
| 4 to 7 | the command word the chain command's a0 (the per-submit word) is ORed into; 0: none |
| 8 | what a core reads of the record before's output: 0 its rank's share of a K range (range word 0's base and width, over the group's ranks), 1 the range words (range i from base + step x its group's index, width tiles) |
| 9 to 15 | the group's size (0: the program says) |
| 16 to 23 | the output tiles a group writes (0: the program says) |
| 24 to 31 | the record's handler (0: the chain command's) |

Group g of a record is cores first + g x size on; its root (rank 0) writes block g of
the output, tiles g x width to (g + 1) x width - 1.

## The device calls

`komorebi/device/km_chain.hpp`. A **shape** gives a record's group size, output width,
K range and tree form: `km::ChainShape` reads them from the record; a program can pass
its own (the decode program reads them from its plans, named by a word of the call).

| Call | Kernel | What it does |
|---|---|---|
| `km::chain_start()` | reader, before its first command | zeroes the chain's words |
| `km::chain_reader(reader, command, step)` | reader | reads record r + 1 from DRAM once compute and the writer took r - 1, runs `step` for each record this core is in |
| `km::chain_writer(shape, writer, command, step)` | writer | takes each record in turn; before `step`, a credit to each core that sends to this one in the record's tree |
| `km::chain_compute(compute, command, step)` | compute | takes each record in turn (unpack reads it, math and pack get its words) |
| `km::chain_input(shape, core)` | the step, before its input | waits until this core's marks are in (a record not marked: at once) |
| `km::chain_wait(core, target, body)` | the step, before its input | the same wait with `body` run while waiting: it returns idle, worked (the wait's limit starts again) or stopped |
| `km::chain_mark(shape, writer, command, direct, dram_too)` | the writer of a group's root, its output written | its writes acknowledged (a direct input sent first), then one mark to each core of the next record for each of its ranges this block meets |
| `km::chain_credit(writer)` | a sender of the tree, before its emit | waits for its receiver's credit for this record |
| `km::chain_peek(words, wanted)`, `km::chain_member(record, me)` | any | the next record if the reader has it already; whether a core is in a record |
| `km::chain_command(chain, record)` | any | the command a record names, the per-submit word ORed in |

Marks and credits are counted per record parity (an even and an odd word each), so a
mark or a credit for the record after can never meet this record's wait. Every wait
is bounded by the run's wait limit and stops at the stop flag. Stamps of each record's
steps are under the compile switch `KM_CHAIN_STAMPS`.

The reader of example 29:

```c++
km::chain_start();
const auto step = [&](const km_command_t& command) {
    if (!km::chain_input(km::ChainShape{}, reader)) {  // a chain's record: marked
        return;
    }
    if (km::handler(command) == kHandlerSum) {
        sum_tiles(reader, command);
    } else {
        map_tiles(reader, command);
    }
};
reader.run([&](const km_command_t& command) {
    if (step_handler(km::handler(command)) && command.a4 == kChainMode) {
        km::chain_reader(reader, command, step);
    } else {
        step(command);
    }
});
```

Outside a chain the same step runs as a plain command: the wait and the marks return at
once.

## The host calls

`komorebi.chain` describes a call; `run.chain` and `run.submit` run it.

| Name | What it is |
|---|---|
| `Call(handler, args, group, size, width, output, input, k=, ranges=, sends=, reads=, direct=, word=, row=, name=)` | a call: its handler and words, its cores (first, count), its groups' size, the block width each root writes of `output`; `input` the call before's output it reads, by `k` (base, count) or up to three `ranges`; `sends` a tree; `reads` its other inputs; `direct` its input sent; `word` (name, argument) the per-submit word |
| `Range(base, step, width, root_only)` | tiles base + step x group index, `width` wide; `root_only`: rank 0 only |
| `Read(tensor, k=, ranges=, earlier=True)` | another input: written further back in the chain (`earlier`), or an outside tensor no call of the chain writes (`earlier=False`) |
| `Direct(tiles, tile_bytes)` | a direct input's landing area on each reading core, as the program declares it |
| `run.chain(calls, mode, flags=0)` | proves and checks the calls, writes the records once (a copy in every bank, the chain from bank 0), returns a `Chain` |
| `run.submit(chain, **word)` | the chain as one exclusive command on the cores of all its calls; `word` its per-submit word by name |
| `chain.report` | a direct input's bytes each root sends and to how many cores, by call |
| `chain.free()` | frees the records |

From an X280 thread, `kmx_chain.h` writes the same records and the chain command:

| Call | What it does |
|---|---|
| `kmx_chain_record(region, index, &call, next, &address)` | `call` (a `kmx_chain_call`: the command, flags, word, a K range or three ranges, size, width, row, handler: 0 the chain command's) as record `index` of a region, its next record at `next`; `address` what the chain command and other records name it by |
| `kmx_chain_command(&chain, region, handler, cores, mode, a0, a1)` | the chain command from the region's record 0 |
| `kmx_chain_range(base, step, width, root_only)` | a range word |

## The earlier input and the direct input

**An earlier input** is a tensor a call reads that the call before does not write: it
was written further back. No mark covers it, so `run.chain` proves it: a core knows a
block is complete when a mark it waited for came from a root that knew it, and a root
knows what its group's cores knew when they sent it their parts, and the block it
wrote. For each core of the call and each tile of the earlier input it reads, the last
call before it that wrote that tile must be in what the core knows; otherwise
`run.chain` refuses, naming the call, the tensor, the tile and the core.

**A direct input** is sent by the call before's roots into the L1 of the cores that read
it, in place of a read from DRAM. `run.chain` checks that it is the call before's
output, read by a share of K tiles; that each reading core's share comes from one
root; and that it fits the landing area the program declares. It reports the bytes each
root sends and to how many cores.

## What is refused

| Refused | Text names |
|---|---|
| an input that the call before does not write, not stated as earlier or outside | the call, the tensor, what the call before writes |
| an earlier input a core reads before anything tells it complete | the call, the core, the tensor and tile, the call and block that wrote it |
| an outside tensor that a call of the chain writes | the call, the tensor, the call that writes it |
| an earlier input that the call before writes (it is this call's input) | the call, the tensor |
| a direct input not read by a share of K, from more than one root, or past its landing area | the call, the core, the tiles |
| a group size that does not divide the call's cores, over 127, or a width over 255 | the call |
| a call of a handler of another program than the first call's | the call, both program fields |
| a handler number over 255, a stamp row over 63 | the call |
| more than one per-submit word, more than 256 calls | the chain |
| a range word's base over 4,095, step over 255 or width over 2,047 | the range |

On the card a record with a bad address, a next record not 64-byte aligned or a chain
of more than 256 records is given up by the reader with the chain's own code (0xC2 in
the give-up's slot field).

## What it costs

[29_chain](../../examples/29_chain/): four calls on 8 cores, 200 runs a way, one
Blackhole p150a, 2026-11-15.

| | A command a step | One chain |
|---|---:|---:|
| on the host, us a run | 703.6 | 266.9 |
| on the card (the dispatcher's submit to done), us a run | 130.6 | 126.3 |

The card's work is the same either way; the chain removes three of the host's four
round trips (about 145 us each on this host).

The decode project, a whole token of 28 layers (140 records) as one chain on 120 cores,
10 tokens: 314.9 / 287.0 / 278.5 us a layer at T 4096 / 1024 / 256 on the framework's
chain, within the spread of its own chain before (315.0 to 315.2, 287.1 to 288.2, 278.6
to 279.9), the token equal to a command a step at 2 and 28 layers, the real prompt's 16
tokens equal to the reference. The chain's words take one page of buffer 32 on each
core; the index past 31 adds one buffer's 16 bytes to the kernel config, nothing to a
kernel's text.

| Decode, 28 layers, 10 tokens | T 4096 | T 1024 | T 256 | kernels (of 86,016 B) |
|---|---:|---:|---:|---:|
| its own chain before the move, us a layer | 315.2 | 287.1 | 278.6 | 85,488 |
| the framework's chain, one handler a chain | 314.9 | 287.0 | 278.5 | 85,568 |
| the framework's chain, a record names its handler | 315.2 | 286.9 | 278.7 | 85,856 |
| the chain's words on buffer 32, untraced | 315.1 | 287.1 | 278.5 | 85,872 |
| the chain's words on buffer 32, traced (512 commands a core) | 315.3 | 286.8 | 278.7 | 85,872 |

A record naming its handler costs decode nothing measurable (the last two rows differ
by +0.3 / -0.1 / +0.2 us, inside the run-to-run spread) and 288 bytes of kernel code:
the header set from the record and the next word's row masked off, at every place a
record's command or its next address is read, in all five kernels. With the chain's
stamps compiled in (`KM_CHAIN_STAMPS`) the decode kernels are 90,240 bytes, 128 over
88 K (90,112), so the stamped build alone takes a 92 K kernel buffer (94,208 B); its
L1 still fits with the plans of T 4096 to 256, 112 bytes left. The plain build keeps
84 K. Traced (`trace=512`, 8,192 bytes of buffer 31 a core), the plain build's L1 fits
with 112 bytes left; `run.trace()` after the stop holds each core's commands.

## A handler as an overlay

A record's handler can load and call overlays in its step as any handler does: the
decode program's records call their operations (the norms, the rotation, attention, the
residual) as overlays, at the [overlays' costs](overlays.md#costs) (72 cycles a load
when the slot holds it). A record cannot name a handler of a program placed as an
overlay ([a program as an overlay](overlays.md#a-program-as-an-overlay)) other than
the chain's own: the record's command keeps the chain command's program field, and the
walk runs inside one program's kernels, so the runtime's load of another program
between two records does not happen. Programs merged into one
([host](host.md#programs-merged-into-one)) are one program: a record names a handler of
any of its sources ([below](#a-record-of-another-programs-handler)).

## In the runtime's form

A program in the runtime's form ([a program as an overlay](overlays.md#a-program-as-an-overlay))
runs chains as one of the kernel form does: its parts call the same device calls
(`km::chain_reader`, `chain_writer`, `chain_compute`, the marks and the credits) from
`km_program`, and the walk of a chain command runs inside the one call of the part.
The runtime's reader zeroes the chain's words once before its first command
(`km::chain_start`) when a program of the run declares buffer 32; a part does not call
it. The chain's stamps (`KM_CHAIN_STAMPS`) are not compiled into parts.

Example 29 runs both forms (`--form runtime`: `kernels/runtime/`, parts that call the
kernel form's step functions), 200 runs a way, one Blackhole p150a, 2026-11-16:

| Example 29 | kernel form | runtime's form |
|---|---:|---:|
| on the host, us a run: a command a step / one chain | 704.5 / 266.9 | 838.9 / 302.8 |
| on the card, us a run: a command a step / one chain | 130.6 / 126.1 | 131.5 / 126.7 |
| every word of y, z, s, t | exact | exact |

The card's time is the same in both forms within 1 us. The host's is about 34 us a
command more in the runtime's form; the chain still removes three of its four round trips.

A run whose groups switch program (`run.load(program, group=)`) sets up buffers 0 to 31
only, so it refuses a program that declares buffer 32: a chain runs in a run whose
cores hold their programs from the start.

## Over islands

On a run of several islands (`km.Session(cards=[0, 1])`), `run.chain` takes calls whose
cores are flat ranks, as the run's are. Each island walks the chain on its own cores:
its records are written in its own DRAM, and the chain command is one ticket with a part
an island, each part carrying its own island's record address (the islands' DRAM
allocators place the records apart once their tensors do). A core marks and credits
cores of its own island only; what must cross is moved by the calls a step already has,
which wait for the far side themselves.

| What a step may do across islands | How |
|---|---|
| read or write another island's pages | `km::read_pages`, `km::write_pages` and the byte calls, as in any command |
| add or gather a copy over the islands | `km::all_reduce`, `km::all_gather`, called by every rank of the record |
| a tree within an island | a group an island (`size` its ranks), the credits inside it |

**Refused** when the chain is made (`chain.ChainRefused`): a record whose input waits for
a mark from a root on another island ("call 3 (sum): core 118 on island 1 waits for a
mark from call 2 (add)'s root 0 on island 0: a mark from one island to another (the far
mark) is not built"), a direct input from another island (the same text), and a record
whose parts go up a tree of a group that spans islands (its credit would cross). A tree
over every rank is therefore a tree an island followed by an all-reduce. The far mark (a
relay ask that adds to a word it names, or port slots for the chain) is not built.

[Example 29](../../examples/29_chain/README.md#over-two-cards) on two p150a cards, 236
ranks, 2 tiles a rank, the median of 50 runs, every word exact both ways, 2026-11-16
(card: submit to done of the slowest island):

| | a command a step | one chain |
|---|---:|---:|
| SCALE, SCALE, ADD, SUM a group an island: on the card, us | 187.5 | 252.6 |
| the same, on the host, us | 1,292.1 | 498.3 |
| with TOTAL (an all-reduce of SUM's copy over the islands) and ADD of it into w, which island 0 holds alone: on the card, us | 535.7 | 603.0 |
| the same, on the host, us | 2,181.5 | 841.3 |

The chain removes the host's round trips as on one card. On the card it is slower over
two islands: a chain of one call takes what its command does (25.7 against 27.1 us), and
each record after the first about 30 us more than a command (two calls 84.3 against
54.2, three 148.2 against 84.3); the same three calls on island 0's 118 ranks alone take
108.1 against 83.9, and on one card a chain is not slower (four calls: 126.1 as a
chain, 130.6 as commands). Where a marked record loses that time in an island build is not found yet.

## A record of another program's handler

Programs merged with `km.merge` ([host](host.md#programs-merged-into-one)) are one
program, so a chain's records name handlers of every source: the record holds the
merged number, and the walk, inside the source of the chain command's handler, hands
each record to the merged part's table, which calls the record's source with its own
number. [29_chain](../../examples/29_chain/)'s `--form merged` merges `sums` (SUM, its 3
kept) and `maps` (SCALE and ADD, 1 and 2 as 4 and 5): the chain walks SCALE, SCALE,
ADD in maps and SUM in sums. One Blackhole p150a, 2026-11-16, 200 runs a way (the
runtime's form as measured [above](#in-the-runtimes-form)):

| Example 29 | runtime's form (one program) | merged (two sources) |
|---|---:|---:|
| on the card, us a run: a command a step / one chain | 131.5 / 126.7 | 127.7 / 127.1 |
| on the host, us a run: a command a step / one chain | 838.9 / 302.8 | 824.9 / 305.9 |
| every word of y, z, s, t | exact | exact |

Between programs of a list (several program fields), not built. A record would name a
handler of another program held by the same cores only with:

- **The record.** A program field per record: the 16 words have no free bits (the
  info word is full, the group word and the range words too, the next word's low bits
  are the stamp row), so the record grows to 32 words or the stamp row gives up bits.
  `run.chain` would take calls of several programs, check that every core of each
  call holds its program, and keep refusing a run whose groups switch program.
- **The walk.** In the runtime's form the walk runs inside one program's part, and the
  runtime's kernels load a program's part per command, not per record (a part loading
  another program's part has not been tried). The walk would move into the
  runtime's three kernels (`program_reader.cpp`, `program_writer.cpp`,
  `program_compute.cpp`): for each record, load the part of the record's program on
  every kernel of the core (as a command to that program does) and call it with the
  record's command. The marks, the credits and the record slots stay as they are.
- **The buffers.** Both programs' buffers in the run's one layout, as a list's
  programs already share it (an index two programs declare differently is refused);
  buffer 32 once for both.
- **The cost.** A record whose program the slots hold: a load of 72 cycles a part; one
  they do not hold: the five parts loaded, 2,919 cycles on one core and 3,971 on 120
  ([a program as an overlay](overlays.md#a-program-as-an-overlay)).
