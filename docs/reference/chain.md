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
| buffer index 31, one page of at least 1,808 bytes: the chain's words and two record slots on each core | the program's buffers; it is the runtime's trace buffer's index, so a chain's program runs untraced |
| in each kernel, the chain command handed to the framework's walk, and in the step, the wait for its input and the marks after its output | the kernels (below) |

The record's handler is the chain command's: every call of a chain names the same
handler, and a program with several operations selects them by a word of the call
(example 29's a0, the decode program's a4 and a6).

## The record

16 words, 64 bytes, 64-byte aligned, at most 256 a chain (`komorebi/abi/km_chain.h`,
plain C for the device, the X280 and the host).

| Word | Holds |
|---|---|
| 0 to 9 | the call's a0 to a9 |
| 10 | its cores: first \| count << 16 |
| 11 | the info word (below) |
| 12 to 14 | three range words: base \| step << 12 \| width << 20 \| root only << 31 |
| 15 | the next record's address in the same DRAM bank; 0: the last |

| Info bits | Holds |
|---|---|
| 0 | marked: the record's input waits for the marks of the record before's roots |
| 1 | sends: its parts go up its group's tree (the credits) |
| 2 | direct: its input is sent into its cores' L1 |
| 4 to 7 | the command word the chain command's a0 (the per-submit word) is ORed into; 0: none |
| 8 | what a core reads of the record before's output: 0 its rank's share of a K range (range word 0's base and width, over the group's ranks), 1 the range words (range i from base + step x its group's index, width tiles) |
| 9 to 15 | the group's size (0: the program says) |
| 16 to 23 | the output tiles a group writes (0: the program says) |
| 24 to 31 | the record's stamp row |

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
    if (command.a0 == kSum) {
        sum_tiles(reader, command);
    } else {
        map_tiles(reader, command);
    }
};
reader.run([&](const km_command_t& command) {
    if (km::handler(command) == kHandlerStep && command.a4 == kChainMode) {
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
| `kmx_chain_record(region, index, &call, next, &address)` | `call` (a `kmx_chain_call`: the command, flags, word, a K range or three ranges, size, width, row) as record `index` of a region, its next record at `next`; `address` what the chain command and other records name it by |
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
| calls of different handlers, more than one per-submit word, more than 256 calls | the chain |
| a range word's base over 4,095, step over 255 or width over 2,047 | the range |

On the card a record with a bad address, a next record not 64-byte aligned or a chain
of more than 256 records is given up by the reader with the chain's own code (0xC2 in
the give-up's slot field).

## What it costs

[29_chain](../../examples/29_chain/): four calls on 8 cores, 200 runs a way, one
Blackhole p150a, 2026-11-15.

| | A command a step | One chain |
|---|---:|---:|
| on the host, us a run | 721.0 | 268.2 |
| on the card (the dispatcher's submit to done), us a run | 130.5 | 126.3 |

The card's work is the same either way; the chain removes three of the host's four
round trips (about 150 us each on this host).

The decode project, a whole token of 28 layers (140 records) as one chain on 120 cores,
10 tokens: 314.9 / 287.0 / 278.5 us a layer at T 4096 / 1024 / 256 on the framework's
chain, within the spread of its own chain before (315.0 to 315.2, 287.1 to 288.2, 278.6
to 279.9), the token equal to a command a step at 2 and 28 layers. Its kernels grew by
80 bytes with the move (85,568 of 86,016). The chain's words take one page of buffer 31
on each core.
