---
title: Movement
summary: Both data-movement CPUs of a Tensix core carrying one stream under a pattern; the device calls, the host side, the rules a program follows and the measured rules for choosing a pattern.
tags:
  - reference
  - device
---

# Movement

Status: implemented in the Tensix runtime and run on one Blackhole card through a
`Session` and `Run` ([11_movement](../../examples/11_movement/): `python movement.py
--pattern AB AAAB`).

A Tensix core has two data-movement CPUs. Komorebi loads the reader kernel on RISCV_0,
which issues on NoC 0, and the writer kernel on RISCV_1, on NoC 1, unless the program
gives them the other NoCs ([below](#the-noc-of-each-cpu)); reader and writer are roles,
and a program written with `km::Reader` and `km::Writer` keeps them. Movement is an
addition beneath those roles for a stream that both CPUs, the **movers**, should carry:
mover A is RISCV_0, mover B is RISCV_1, each on its kernel's NoC.

A stream is cut into blocks. A **pattern** of 1 to 24 letters says which mover moves
block k: letter k mod the length, `A` or `B` (`AAAB`: mover B takes every fourth). Each
mover moves its blocks through its own **half** of a **buffer pair**, two circular
buffers, and compute takes the blocks in order, each from the half its owner filled.

## Device calls

```c++
#include "komorebi/device/movement/km_movement.hpp"

constexpr km::BufferPair kInput = {kInputMoverA, kInputMoverB};  // two buffer indices

using Input = km::Tensor<0>;  // the host's tensor_args() from compile argument 0 on

// One source for both movers (movement.cpp): a km::Reader on mover A, a km::Writer on B.
km::Movement movement;
movement.run([&](const km_command_t& command) {
    const km::Stream read(
        km::kMoveRead,                              // or km::kMoveWrite
        Input(command.a0),                          // the tensor
        first_page, tiles, tiles_per_block,         // the blocks
        repetitions, kInput,
        km::Pattern::of(command.a4)                 // the host's pattern word
    );
    return movement.move(read);  // this mover's blocks; false: gave up (recorded)
});
```

```c++
// Compute: block k is in the half its owner fills.
for (uint32_t block = 0; block < blocks; block++) {
    const uint32_t half = km::block_half(kInput, pattern, block);
    if (!compute.wait_front(half, tiles_per_block)) return;
    // ... use the block's pages ...
    cb_pop_front(half, tiles_per_block);
}
```

| Name | What it is |
|---|---|
| `km::Pattern` | `length` and `owners` (bit k set: mover B owns position k). `Pattern::of(word)` from the host's word, `word()`, `valid()`, `owner(block)` |
| `km::pattern("AAAB")` | the pattern of a word of letters, at compile time; `kPatternA`, `kPatternB`, `kPatternAB`, `kPatternAAAB`, `kPatternBBBBBAAA` |
| `km::BufferPair` | `mover_a`, `mover_b`: the two halves' buffer indices; `half(mover)` |
| `km::block_half(pair, pattern, block)` | the half that holds block `block` (compute and movers alike) |
| `km::kMoverA`, `km::kMoverB`, `km::kMover` | the movers; `kMover` is this CPU's (data-movement kernels only) |
| `km::Walk` | a stream's blocks (below); `Walk::run(first_page, tiles, tiles_per_block)` is the plain case |
| `km::Stream` | direction (`kMoveRead`: tensor pages into this mover's half, `kMoveWrite`: pages of its half to the tensor), tensor (a `km::Tensor`, or a tensor accessor whose pages are all on this island), walk, repetitions, pair, pattern, port (default 0). Two constructors: the plain one above and `(direction, tensor, walk, repetitions, pair, pattern, port)` |
| `km::move(core, streams...)` | runs this mover's share of every stream given, from any data-movement handler (`core`: the handler's `km::Reader`, `km::Writer` or `km::Movement`). False: gave up |
| `km::move(moved, core, streams...)` | the same, adding to `moved` (`km::Moved`: tiles moved, cycles in passes that moved a block) |
| `km::Movement` | for a source compiled for both movers: on mover A a `km::Reader`, on mover B a `km::Writer`, plus `mover()` and `move(streams...)` |
| `km::Gated(stream, gate, bits)` | a read stream given to `move` that offers nothing while `gate.holds(bits)`; the other streams go on. A give-up met while it holds is recorded by `gate.give_up(core, code)`. A chain build's gate is `km::ChainGate` (`km_chain.hpp`, [classes, chains](classes.md#chains)) |

`move` runs a loop: in each pass it issues the next block of every stream whose half has
room (a read) or data (a write), waits once for all of them at the NoC barrier, and
pushes or pops each, until this mover's blocks of every stream are done; writes are
acknowledged before it returns. A pass that moves nothing checks the stop flag and the
wait limit every sixteenth time, as `reserve` and `wait_front` do.

A `km::Reader` handler and a `km::Writer` handler of a three-kernel program can share a
stream: each calls `km::move` with the same stream description, and each moves its own
blocks.

### A tensor's pages on any island

Two calls read and write a tensor's pages wherever they live (`km_ask.hpp`; on one
island each is exactly the tt-metal page call it stands for). A kernel takes the tensor
as a `km::Tensor<offset>` (its accessor arguments from compile argument `offset` on,
`tensor.tensor_args()` on the host, and its reference word):

| Name | What it is |
|---|---|
| `km::read_pages(core, command, tensor, first, pages, to)` | pages `first` to `first + pages - 1` into L1 `to` on (page k at `to` + k aligned pages), from a reader or a writer: this island's read as `noc_async_read_page` reads them (the caller's barrier waits), each run another island holds asked for and waited for at the CPU's page port |
| `km::write_pages(core, command, tensor, first, pages, from)` | the other way: this island's pages written as `noc_async_write_page` writes them (the caller's barrier waits), each run another island holds put into its DRAM and waited for at the CPU's page port until written; `from` is free once the caller's `noc_async_write_barrier` has returned (this island's pages are not waited for) |

The page port is a port of its own for the reader and one for the writer
(`KM_PORT_PAGES`, `KM_PORT_PAGES + 1`), past the ports 0 to `KM_PORTS - 1` a kernel
names: a reduce, an arrive or an emit never counts there, so pages across and a
reduce's rounds never share a count. `km::put_pages`, whose wait comes later
(`km::put_wait`), counts at a port of its own in the CPU's own words (`KM_OWN_PUTS`),
so a read across between a put and its wait passes on its own pages alone.

The forms by word take the accessor arguments and the word apart:
`km::read_pages(core, command, args, word, first, pages, to)` and the same for
`write_pages`. `args` are the tensor's `TensorAccessorArgs` (`tensor.accessor_args()`
on the host), `word` its reference word (`tensor.address`: a bank address on one card,
an entry of the tensor table over islands), the same on every island. All return false
when they gave up (recorded, the island named). How the pages cross:
[islands](islands.md#pages-across-a-link).

### A page call past its tensor's end

Under `KM_GUARD` (the default) every page call checks its range against the tensor's
pages before it reads or writes one (a bank range's against its bytes: below): a call
whose last page is past the tensor's last gives up with `bad_item` in `pagerange`, as any bad item does (a call
of no pages is inside, wherever its first page is): the
command is given up at its deadline (`CommandTimeout`) and the run goes on; nothing is
read or written, and the host names the call's last page, whole (the core's live
words, `page`; the give-up's own word names the command's slot only). Over islands the
bound is the tensor table's entry (its pages beside the copy bit), for its word and for
the bank address of its part on the island (a `km::SizedTensor` given that address) when
the part holds the whole tensor (a copy, or a tensor one island holds); any other bank
address over islands is not checked. On a card with no tensor table the session lists its live
DRAM tensors at each load, sorted by address, after each core's control block: 12 bytes
of L1 a tensor, its address, its bytes and its bytes in each bank (its pages over the 8
banks, rounded up, times the aligned page bytes), a row of the load's L1 budget ("page
bounds list": 1,216 B at 100 tensors, 12,016 B at 1,000, 12,304 B at 1,024). A call's
bound there is the smaller of the tensor's bytes in pages of the call's size and the
pages of that size its bytes in each bank hold, times the 8 banks, so a
`km::SizedTensor` of a page size of its own stops where its tensor ends in a bank: 9
pages of 4,096 B on a tensor of 18 pages of 2,048 B (6,144 B a bank) are refused,
where the ninth wrote 2,048 B past the tensor in bank 0 before (2026-11-19). In the
tensor's own page size the two are the same bound or the second is larger. The run
holds the listed tensors until it ends. A bank range (`km::BankRange`, its word a bank
address on one card) is checked against the listed tensor whose bank bytes hold its
bank address (the list's search, every call: no lines): its call's last byte lies
inside them, else the call gives up the same way; a bank address in no listed
tensor's bank bytes is not checked, as a page call whose tensor word is no listed
tensor's is not. Over islands its entry bounds it. A tensor made
while a run is loaded, one made at a given `address=`, and those past
`Session.page_bounds`
(1,024; the load names them) are not checked. A stream's walk (`km::move`) is checked
once a command, its first page to its last, before its first block. The byte calls
(`km::read_bytes`, `km::ask_bytes`, `km::put_bytes`, `km::write_bytes`, by word, by
`km::Tensor` and by `km::SizedTensor`) check their page the same way, and
`km::gather_pages` checks its run against both tensors, the one it reads and the one it
writes. Each CPU keeps the bounds of its last four tensors (by word and page size, in
pages); a call whose tensor is among them compares two words and its range, and only
another looks the bound up (the list's search or the table's entry). The lines are
read again after a tensor table row is written while the run lives: the island
header's `KM_ISLAND_BOUNDS` word, which the load sets to 0 and each such write bumps,
is read once a command. An overlay part keeps no lines (it looks every bound up). The
cost, on card 0 at 1,350 MHz: example 07's thread loop (two page calls a command on
each core) takes 11.15 ms against 10.68 ms with `guard=False` (12.11 ms looking every
bound up), and the text grows by a few hundred bytes a CPU (07's writer 8,284 to
8,528 B); the bound in each bank added 132 B to example 01's reader and to its writer
(13,096 to 13,228 B and 8,500 to 8,632 B, a mock build). `KM_GUARD` 0 builds the calls
as before, byte for byte.

### A page write's source

A NoC write lands at its destination's low four address bits, whatever its source's.
Under `KM_GUARD` every `km::write_pages` checks that its L1 source and its page bytes
are multiples of 16 before it writes a page, and otherwise gives up with
`bad_transfer` in `pagesource`, naming the caller's line (the core's live words):
nothing is written, and the command is given up at its deadline as for a page past the end.
`KM_GUARD` 0 builds the call as before, byte for byte.

A class's command is also checked by the host when it is submitted: each
`tensor.pages(first, count)` of the handler's lines, for the ranks and range-loop turns
that bound it (the first and last of each, or every one when the expression takes `%`,
`//`, `min` or `max`), against the tensor's pages. A range past the end raises
`ValueError` before the command is sent, naming the line, the tensor, its pages, the
rank and the turn. A line under an `if` other than the leader's, or one whose value
the card computes, is left to the card's check. Each words and group is checked once a
run.

### A stream over islands

A stream's tensor given as a `km::Tensor` reads and writes its pages through these
calls, so a stream over a tensor whose pages are on several islands works with the same
kernel: the pages this island holds are read or written as on one card; a row of a
block whose pages another island holds is asked for or put (one line a row of
consecutive pages) and waited for before the block is handed on, counted at port
`port` on mover A and `port + 1` on mover B. A crossing block is slower than a local
one by the link's rate and the line's round trip; the movers still share the blocks by
the pattern.

A program loaded on one island is built without the tensor table: the stream keeps the
tensor's accessor and its build is the one-card stream's: the movement example's two
kernels have the same code size within 0.6% (mover A 15,292 bytes against the plain
accessor stream's 15,384, mover B 11,040 against 11,020). Its one-island loop is the
plain stream's; measured on the whole grid of a p150a before the reference word (the
same loop, mover A 15,248 bytes), patterns `A`, `AB`, `AAAB`, `B` read at 427.9, 486.4,
502.7 and 247.2 GB/s against the example's 427.8, 485.9, 502.6 and 247.4.

Over two cards, island 0's first four cores copy 192 pages that island 1 holds alone,
exact, with the movement example's kernels unchanged (`examples/11_movement/islands.py`):
a READ of 48 pages a core takes 275.8 µs, 1.43 GB/s in all. With `--write-across` the
copy's output is island 1's and every page is put across: exact, 313.4 µs. Each row of
a block crosses as one line and its answer; the cores wait for it before the block is
handed on.

### The blocks of a stream

A `km::Walk` is a two-level strided walk over the tensor's pages: block b of a
repetition starts at page `first_page + b * block_step` and holds `rows` rows of
`columns` pages, row r at `row_step * r` pages from the block's start, page c of a row
at `column_step * c`. In L1 a block is one item of `tiles_per_block` pages, row r at
`row_pitch * r` pages from the item's start. The last block of a repetition moves
`last_rows` rows of `last_columns` pages (a ragged end); its item stays whole.

| Field | The plain case `Walk::run(first_page, tiles, tiles_per_block)` |
|---|---|
| `first_page`, `blocks`, `block_step` | `first_page`, tiles / tiles_per_block rounded up, `tiles_per_block` |
| `rows`, `row_step`, `columns`, `column_step`, `row_pitch` | 1, 0, `tiles_per_block`, 1, `tiles_per_block` |
| `tiles_per_block`, `last_rows`, `last_columns` | `tiles_per_block`, 1, the pages left for the last block |

The matrix product's K split walks a chunk of A (`rows` rows of the block, `row_step`
K tiles, `columns` = the chunk's K tiles) and of W (`rows` = the chunk's K tiles,
`row_step` N tiles, `columns` = the block's width) this way.

## Host

```python
import komorebi as km

pattern = km.Pattern("AAAB")
pattern.word  # 0x804: a command argument (km::Pattern::of reads it)
pattern.define()  # "0x804": the text of a define, for a pattern fixed at build
pattern.share("B")  # Fraction(1, 4): the part of the blocks mover B moves
pattern.blocks("A", 10)  # 8: how many of blocks 0 to 9 mover A moves

program = km.Program(
    "movement",
    kernels="kernels/",  # movement.cpp (built for both movers) and compute.cpp
    buffers=[km.BufferPair(0, 1, "bfloat16", 2048, pages=64)],  # or pages=(48, 16)
    compile_args={"movement": tensor.tensor_args()},  # both movers: a km::Tensor<0>
    handlers="kernels/movement.h",
)
```

`kernels` takes a directory with `movement.cpp` and `compute.cpp`, or a dict
`{"movement": path, "compute": path}`; the one source is built for both data-movement
CPUs. The three-file form is unchanged, and a directory holding both forms is refused.
`compile_args` and `opt` take `"movement"` for both movers, or `"reader"` and
`"writer"` each, not both.

A `km.BufferPair(mover_a, mover_b, format, page_bytes, pages)` stands for two
`km.Buffer`s with one format and page size; `pages` is both halves' count or
`(mover A's, mover B's)`. The program refuses a half declared twice, and at `load` it
reads the kernel sources: a source that names a half, by a number or a
`constexpr uint32_t` constant of the source or its headers, in a filling call
(`cb_reserve_back`, `cb_push_back`, `reserve`, `reserve_back`) and a draining call
(`cb_wait_front`, `cb_pop_front`, `wait_front`), or two sources that both fill or both
drain one half, are refused (`ProgramError`). Halves named through a variable or moved
by `km::move` are not seen; the rules below hold for them all the same.

### The NoC of each CPU

```python
program = km.Program("swapped", kernels="kernels/", noc={"reader": 1, "writer": 0})
program.kernel_noc("reader")  # 1
```

`noc` gives the NoC of the reader kernel (RISCV_0, mover A) and of the writer kernel
(RISCV_1, mover B); without it they are on NoC 0 and NoC 1. Each CPU has a NoC of its
own: a `noc` that puts both on one NoC, names another kernel or another NoC than 0 and 1
is refused (`ProgramError`). The device calls read the kernel's NoC; nothing in a source
changes. What does: a broadcast's arguments name the NoC of the kernel that runs it
(`komorebi.layout.broadcast_args(..., noc, ...)`; a broadcast run on the other NoC gives
up at its start), and the direction of its multicast follows. Programs that share cores,
and every program of a load whose groups switch, give the same `noc`.
Swapped, the column broadcasts run on NoC 0 beside the X280's command multicasts: the
collision experiment saw no fault there, but only the default arrangement does not depend
on that observation.

Measured on one p150a: example 11 with every program swapped is exact, pattern `A` at
246.8 GB/s (mover A now reads at NoC 1's rate, as `B` does above) and `AB` at 484.7.
The dense linear kernel with the reader on NoC 1 and the writer on NoC 0 (its X stream
on NoC 1 and its W stream on NoC 0, as in Kohaku-TT's kernel) is exact and slower:
4,978.6 us a call at 8192 cubed against 4,488.1, 36,489.1 at 16384 cubed against
34,544.8 (`projects/dense/README.md`).

## Rules for a program

1. A half has one filling CPU and one draining CPU for the whole load. Compute's packer
   keeps its own count of the pages it pushed and writes it whole into the buffer's
   count, so a half that compute fills in one command and a mover fills in another gets
   its count rewound. Give a stream that compute writes its own pair. `move` records the
   roles of each half on each mover and gives up (`bad_item`, at `data`) when a half's
   roles change within a load.
2. Both movers run `move` with the same stream description (direction, walk,
   repetitions, pair, pattern) for a command, and compute takes the same blocks in
   order: block k from `block_half(pair, pattern, k)`, a whole item each.
3. A half holds a whole number of items (`tiles_per_block` divides its pages), and a
   walk's rows fit the item (`(rows - 1) * row_pitch + columns <= tiles_per_block`, and
   so for the last block); `move` gives up with `bad_item` otherwise.
4. A ragged block occupies a whole item: compute takes and pops `tiles_per_block` pages
   and leaves the pages the walk did not move alone; a write stream writes only the
   pages its walk names.
5. A mover's halves are its own: no other code of that mover pushes or pops them while
   a stream through them is open.

A wait in `move` that runs past the wait limit, or a stop, gives up with the mover's
word (`kStReader` on mover A, `kStWriter` on mover B) at `data`, like `reserve` and
`wait_front`.

## Choosing a pattern

There is no automatic choice: the program gives the pattern. The measured rules, from
`experiments/core/movement_probe` (one p150a, AICLK 1,350 MHz, 32 tiles a block, 256 tiles
a core, GB/s over all cores):

**The ratio follows the routes.** NoC 0 and NoC 1 reach DRAM at different rates, and
the rates change with the cores that share the routes:

| reads | classic reader | A | B | AB | AAAB | AAAAABBB | BBBA |
|---|---|---|---|---|---|---|---|
| row of 8 | 319.2 | 297.6 | 95.5 | 190.6 | 381.1 | 254.0 | 127.1 |
| 120 cores | 422.4 | 422.9 | 245.9 | 477.8 | 498.9 | 495.5 | 328.4 |

| writes | classic writer | A | B | AB | AAAB | AAAAABBB | BBBA | BBBBBAAA |
|---|---|---|---|---|---|---|---|---|
| row of 8 | 300.5 | 86.5 | 300.8 | 172.5 | 115.2 | 138.3 | 300.1 | 224.1 |
| 120 cores | 313.1 | 178.2 | 312.6 | 354.8 | 237.3 | 284.7 | 413.4 | 460.0 |

On one core both movers read 1.92 to 1.98 times as fast as one. On the full grid
`AAAB` (reads) and `BBBBBAAA` (writes) are best; on a row of 8, `AB` reads at 0.6 of
the classic reader.

**Compute takes blocks in order, so the slower mover sets the pace.** The faster mover
waits for room behind the slower one (`AB` reads on 120 cores: mover A busy 0.15 of
the run, mover B 0.99).

**A shared stream gains when the other NoC is free and loses when the other mover
already carries a stream on it.** The matrix product's K split, whose chunk reads set
its pace while the writer is idle, gains 10.5% under `AB` (it reads its chunks so:
`projects/dense/kernels/split/chunks.hpp`); the resident product, whose
writer already writes Y on NoC 1, gains at most 2% from splitting Y.

**The engine costs 2 to 4% against blocking calls** on one core (pattern `A` against
the reader's blocking loop: -1.8% at 8, 32 and 128 tiles a block; pattern `B` against
the writer's: -4.4% to -2.7%). On the grid, where the routes set the rate, it costs
nothing measurable.

## Limits

- Dynamic NoC mode is not supported by the movers: each issues on its own NoC
  (`static_assert(noc_mode != DM_DYNAMIC_NOC)`).
- Copies through halves (each mover reads its blocks and writes them back, no compute)
  pay on one core only: +58% at 8 tiles a block, but -4% to -36% on 8 and 120 cores.
