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
which issues on NoC 0, and the writer kernel on RISCV_1, on NoC 1; reader and writer
are roles, and a program written with `km::Reader` and `km::Writer` keeps them. Movement
is an addition beneath those roles for a stream that both CPUs, the **movers**, should
carry: mover A is RISCV_0 on NoC 0, mover B is RISCV_1 on NoC 1.

A stream is cut into blocks. A **pattern** of 1 to 24 letters says which mover moves
block k: letter k mod the length, `A` or `B` (`AAAB`: mover B takes every fourth). Each
mover moves its blocks through its own **half** of a **buffer pair**, two circular
buffers, and compute takes the blocks in order, each from the half its owner filled.

## Device calls

```c++
#include "komorebi/device/km_movement.hpp"

constexpr km::BufferPair kInput = {kInputMoverA, kInputMoverB};  // two buffer indices

// One source for both movers (movement.cpp): a km::Reader on mover A, a km::Writer on B.
km::Movement movement;
movement.run([&](const km_command_t& command) {
    const km::Stream read(
        km::kMoveRead,                              // or km::kMoveWrite
        TensorAccessor(arguments, command.a0),      // the tensor
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
| `km::Stream` | direction (`kMoveRead`: tensor pages into this mover's half, `kMoveWrite`: pages of its half to the tensor), tensor accessor, walk, repetitions, pair, pattern. Two constructors: the plain one above and `(direction, tensor, walk, repetitions, pair, pattern)` |
| `km::move(core, streams...)` | runs this mover's share of every stream given, from any data-movement handler (`core`: the handler's `km::Reader`, `km::Writer` or `km::Movement`). False: gave up |
| `km::move(moved, core, streams...)` | the same, adding to `moved` (`km::Moved`: tiles moved, cycles in passes that moved a block) |
| `km::Movement` | for a source compiled for both movers: on mover A a `km::Reader`, on mover B a `km::Writer`, plus `mover()` and `move(streams...)` |

`move` runs a loop: in each pass it issues the next block of every stream whose half has
room (a read) or data (a write), waits once for all of them at the NoC barrier, and
pushes or pops each, until this mover's blocks of every stream are done; writes are
acknowledged before it returns. A pass that moves nothing checks the stop flag and the
wait limit every sixteenth time, as `reserve` and `wait_front` do.

A `km::Reader` handler and a `km::Writer` handler of a three-kernel program can share a
stream: each calls `km::move` with the same stream description, and each moves its own
blocks.

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
pattern.word        # 0x804: a command argument (km::Pattern::of reads it)
pattern.define()    # "0x804": the text of a define, for a pattern fixed at build
pattern.share("B")  # Fraction(1, 4): the part of the blocks mover B moves
pattern.blocks("A", 10)  # 8: how many of blocks 0 to 9 mover A moves

program = km.Program(
    "movement",
    kernels="kernels/",  # movement.cpp (built for both movers) and compute.cpp
    buffers=[km.BufferPair(0, 1, "bfloat16", 2048, pages=64)],  # or pages=(48, 16)
    compile_args={"movement": tensor.accessor_args()},  # both movers
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
`experiments/movement_probe` (one p150a, AICLK 1,350 MHz, 32 tiles a block, 256 tiles
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
its pace while the writer is idle, gains 10.5% under `AB`; the resident product, whose
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
