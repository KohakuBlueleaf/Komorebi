---
title: Programs written as classes
summary: Experimental. A class deriving km.Program whose methods are handlers, compiled when it is made into an ordinary program's files (the movement source, the compute kernel, the handlers header, an overlay's parts); the lines a handler may hold and the line each lowers to, overlays, islands and what may cross a link, the refusals, and the examples measured against the hand programs.
tags:
  - reference
  - host
---

# Programs written as classes

Status: experimental. The lines of this page are those of level 1; six example scripts
run as classes beside their hand programs, on one Blackhole card and on two
([below](#measured)).

A class deriving `km.Program` whose methods are decorated with `km.handler` is a
program written as a class. Made, it is compiled: each handler's body is read with
`ast`, never run in Python, and becomes a case of the movement source (one source for
both movers, [Movement](movement.md)) and of the compute kernel. The made object is an
ordinary `Program` of those files, loaded with `Session.load` and given commands with
`Run.submit`.

```python
import komorebi as km


class MapCollect(km.Program):
    """Each core of a MAP command's group adds its part of `tiles` input tiles from
    `first_tile` into one tile and stores it at output page first_page + rank."""

    @km.handler
    def map(
        self,
        source: km.Tensor,
        tiles: int,
        output: km.Tensor,
        first_page: int,
        first_tile: int,
    ):
        part = self.partition(source.pages(first_tile, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(source[part], block=8):
            total += tile
        output.pages(first_page + self.rank, 1).write(total)
```

The lines move nothing that the source does not say: the stream's line is the read of
the core's part, the last line the write of its tile; the sum is compute's.

## Making and calling

From [02 merge tree](../../examples/02_merge_tree/as_class/merge_class.py):

```python
program = MergeTree(source=inputs[0], output=output, depth=args.depth, log=hand.log)
run = session.load(program, deadline=args.deadline, wait=args.wait)
arguments = (inputs[source % 2], args.tiles, output, job % args.out_pages)
run.submit(
    MergeTree.merge,
    (first, count),
    args=(*arguments, form, source),
    exclusive=True,
)
```

| Made with | What it is |
|---|---|
| a tensor by each tensor parameter's name | an example of the tensor: its format and page size (the buffers', the movers' compile arguments); a tensor of an island session too |
| `plan` | the sizes by name: a module constant the handlers use (`BLOCK`), a field's symbol |
| `l1` | the bytes left for buffers (`Session.l1()[1]`): the buffers must fit, a field's last symbol takes what they leave |
| `rect`, `nocs` | with a broadcast or `self.x`, `self.y`: the rectangle (x, y, columns, rows), the whole grid when not given, and the grid's NoCs, the session's (`Session.load` gives them) when not given; each core's program arguments are made from them at the load |
| `depth`, `log` | the ring slots used, as a `Program`'s; where the build's report goes (`print`) |

The build writes `build/<name>/` beside the class's source: `<name>.h` (the handler
numbers, the buffers, the patterns), `movement.cpp`, `compute.cpp`, an overlay
handler's parts in `<handler>/`, and `program.txt`, the report it also prints: each
buffer with its indices, format, pages, bytes and the Python line that made it, and the
L1 they take. The files are kept; a compile error names their lines, and an inline
body's lines name the Python file's.

`Run.submit(Class.handler, group, args=(...))` makes the command words from the
arguments: a tensor's reference word (`tensor.address`), an int (32 bits), a pattern's
word, in the order of the parameters; a tensor whose `pages` a line uses whole adds a
word, its page count. A handler that returns a word asks for the result. For arrays of
words (`Run.submit_many`), `Class.handler.args(...)` gives one command's words, and
`submit_many(Class.handler, firsts, counts, words)` takes them.

## The lines of a handler

| Line | Movers | Compute |
|---|---|---|
| a parameter `name: km.Tensor`, `name: int`, `name: km.Pattern` | a command word | a command word |
| `n = a + b * 2`, `n += 1`, an `if` on ints | the int, where a kept line uses it | the same |
| `part = self.partition(pages)` | `km::partition(count, rank, size)` | the same |
| `part = b.partition(pages)` | `km::partition` by the core's column of the rectangle (a row's broadcast) or its row (a column's) | the same |
| `total = km.tile.zeros(km.fp32)` | | DST held, tile 0 filled with zeros |
| `for tile in km.Stream(pages, block=8, pattern="AB", repeat=n):` | one read stream in a buffer pair, moved with the write streams the loop pushes to (`movement.move`) | each block from the half its mover filled, each tile of it |
| `km.Stream(pages).drain()` | the read stream, its tiles dropped | each block waited for and freed |
| `total += tile` | | the tile copied and added into the held tile |
| `out = km.Stream(pages, write=True, block=8)` and `out.push(tile)` | one write stream, moved with the read that feeds it | the tile copied into the same half of the write pair |
| `out.push(t)`, `t` a body's tile in the loop | | DST 0 packed into the write pair at the loop tile's place |
| `pages.write(total)` (one page) | mover B: `wait_front`, `km::write_pages` of one page, `noc_async_writes_flushed`, `cb_pop_front` | the held tile packed into a page buffer, DST released |
| `whole = km.reduce(total, form=form)` | `km::reduce_arrive` on mover A, `km::reduce_emit` on mover B | `km::reduce`, the arrivals added in rank order |
| `if self.rank == self.leader:` and `pages.write(whole)` | the leader's write of the sum | |
| `return word` under `if self.rank == self.leader:` | `movement.result(word)` on mover B | |
| `b = km.Broadcast(self, along="row", window=2)` | the broadcast on its mover (rows on mover A, columns on mover B); a core it leaves out skips the handler | the same core skips it |
| `if b.sender():` `v = b.cast(t.pages(first, n).read())` `else:` `v = b.arrive()` | the sender's reserve, read and cast, the others' arrival, block by block of the loop around it; the last block's finish | |
| `for tile in v:` | | the block received, each page of it, then freed; the padding after the last |
| `for i in range(n):` | a loop | a loop |
| `km.cxx("...", takes=[n], on="mover_b")` | the C++ as written, on the CPU `on` names | compute when `on` is not given |
| `w = km.cxx(..., gives=int)` | `uint32_t w = 0;` then the body, which sets it | the same |
| `t = km.cxx(..., gives=km.Tile)` | | `tile_regs_acquire()` then the body, which leaves the tile in DST 0 |
| `self.f.fill(km.Stream(t.pages(first, n)))` | mover A: `reserve` of the field, `km::read_pages` of its n tiles, the barrier, `cb_push_back` | |

`self.rank`, `self.size` and `self.leader` are the core's rank in the command's group,
the group's size and its rank 0; `self.x` and `self.y` its column and row in the
rectangle.

A broadcast's rectangle is fixed at load (a program argument), and `Run.submit`
refuses a broadcast's command whose group is not the rectangle's, naming the
`km.Broadcast` line; a rectangle per command is not in this version.

A body takes the handler's names: an int, a partition, a pattern or a body's word by its
own name; a page that mover B writes (`takes=[whole]` on `"mover_b"`) as its buffer; a
stream's tile on compute as `tile_buffer` and `tile_index` (the loop's name, then
`_buffer` and `_index`); a field (`takes=[self.f]`) as its buffer on compute and its L1
address on mover B, freed after the body. A body's first lines that are `#include`
lines go to the top of its kernel. From
[12 overlays](../../examples/12_overlays/as_class/overlays_class.py):

```python
for tile in km.Stream(source.pages(first, tiles)):
    scaled = km.cxx(
        """
        #include "api/compute/eltwise_unary/binop_with_scalar.h"
        copy_init(tile_buffer);
        copy_tile(tile_buffer, tile_index, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scalar);
        """,
        takes=[tile, scalar],
        gives=km.Tile,
    )
    out.push(scaled)
```

A stream's pattern is the reads' `AAAB` and a write's `B` unless the line gives one. A
read pair holds 4 blocks a half and a write pair 2; a page buffer 2 pages; a reduce's
arrival buffer 118 float32 tiles (one slot for each other core of a 119-core group by
owner); a broadcast's buffer `window` blocks of the pages it casts. Buffer indices are
0 up for reads, 8 up for arrivals, broadcasts and fields, 16 up for writes; 26 to 31
are the runtime's. Compute's tiles are in DST as float32; a float32 buffer is unpacked
to it unrounded.

## Fields

A field is declared on the class with its format and shape; a symbol of the shape is
declared in the class's body, `TILES = km.symbol()`, and used as
`count: km.L1[km.fp32, TILES, 32, 32]`. Each field is a buffer of whole tiles, index 8
up. A symbol is the plan's; one symbol left unknown is solved as the largest for which
the fields and the buffers fit `l1`, and the report prints it.

A field is filled from a stream of a page range, of its own tiles' count, and one
`km.cxx` body in the same block takes it; every core that fills it frees it. `@`,
slices and `.columns` of a field are level 2.

## Overlays

`@km.handler(overlay=True)` makes a handler's code an [overlay](overlays.md): the
build writes its parts (`<handler>/movement.cpp`, built for each mover, and
`<handler>/compute.cpp` when compute has code), declares the overlay
(`kOverlay<Name>` in the header) and a slot for each overlay with a part; the kernels'
case loads it and calls the part with the command. When the session builds the parts
it prints where each handler's code lives. From
[12 overlays](../../examples/12_overlays/as_class/README.md), on the card:

```text
where the handlers' code lives:
  mix: overlay 1: reader 64 B, writer 192 B; slots: reader 3 x 8,640 B, writer 2 x 3,968 B, unpack 1 x 2,624 B, math 1 x 1,664 B, pack 1 x 1,664 B
  scale: overlay 2: reader 3,968 B, writer 3,968 B, unpack 2,624 B, math 1,664 B, pack 1,664 B; slots: reader 3 x 8,640 B, writer 2 x 3,968 B, unpack 1 x 2,624 B, math 1 x 1,664 B, pack 1 x 1,664 B
```

A broadcast's handler is resident.

## Islands

The same class runs on `km.Session(cards=[0, 1])`, made with the island session's
tensors, which are placed by their hints. Each island runs a copy of the program, so
its handlers are named by the class (`MapCollect.map`). Made with island tensors, the
build prints, for each move of each handler, whether a page of it can be another
island's given its tensor's placement, and what it lowers to then. From
[01 map/collect on two islands](../../examples/01_mapcollect/as_class/islands_class.py),
on the cards:

```text
may cross (each move: its tensor's placement; what it lowers to):
  map, line 41, read stream of source (split 0 (island 0 steps 0..117, island 1 steps 118..235)): may cross; km::read_pages_at a row: this island's pages by noc_async_read_page, another island's run by an ask line to its link core (km::ask_line_at); one island: noc_async_read_page a page
  map, line 43, page write of output (split 0 (island 0 steps 0..117, island 1 steps 118..235)): may cross; km::write_pages_at: this island's pages by noc_async_write_page, another island's run by a put line to its link core (km::ask_line_at); one island: noc_async_write_page a page
```

A copy's read never crosses: each island reads its own.

## Refusals

A line the compiler does not take raises `Refused` (a `ProgramError`) when the class is
made, naming the handler, the line and what to write instead (the classes of
[test_classes.py](../../tests/core/test_classes.py)):

```text
Sliced.copy, line 30, `km.Stream(source[1:4]).drain()`: a tensor is in DRAM: a slice is not a value; take a part: source[self.partition(...)], or use a stream or read()
Unled.merge, line 40, `output.pages(0, 1).write(whole)`: whole is the reduce's sum, which only the leader holds; write it under `if self.rank == self.leader:`
Unsolved.count, line 45, `count: km.L1[km.fp32, TILES, 32, 32]`: TILES is solved against the L1 the buffers leave; make the program with l1=session.l1()[1]
```

A `CommandTimeout` of a program made from a class names, for each core that did not
report, the handler and the Python line its CPU was waiting in;
`komorebi.dsl.program.lines_of(program, error)` gives those lines.

## Measured

Each class runs in the same session as its hand program and from the same seed (the
scripts' `--beside`; on two cards, the hand program in the job before). Kernel text is
the packed text of the five CPUs, of 70,656 B.

| Example | Results | Kernel text, class / hand | Time, class / hand |
|---|---|---|---|
| [01 map/collect](../../examples/01_mapcollect/as_class/README.md) | the same: 20 jobs checked, 10 exact | 33,584 / 34,288 B | on the cores 1.77 / 1.91 us a command (120 cores, 1,000 jobs, trace stamps) |
| [01 on two islands](../../examples/01_mapcollect/as_class/README.md) | the same: every tile exact; 354 asks and 708 pages over the links, as the hand's | 60,048 / 61,152 B (mock, islands' build) | own groups 2.96 / 3.08 us a command; island 0 reading island 1's input 637.6 / 617.4 us a round |
| [02 merge tree](../../examples/02_merge_tree/as_class/README.md) | the same: 40 jobs, exact ones bit-equal, 40 result words | 42,240 / 36,176 B | on the cores 15.33 / 15.14 us a command (first seen to sent, median) |
| [03 broadcast](../../examples/03_broadcast/as_class/README.md) | the same: 960 ranks, all exact (level 1a) | 48,736 / 53,424 B (mock) | 109.0 / 99.8 us a job, exclusive (level 1a) |
| [11 movement](../../examples/11_movement/as_class/README.md) | the same: every copy exact (level 1a) | 38,288 / 36,720 B | READ AAAB 308.74 / 310.44 GB/s (level 1a) |
| [12 overlays](../../examples/12_overlays/as_class/README.md) | the same: every mix word and scaled value | 31,088 / 27,324 B, and the parts above | a scale turn 1,379 / 1,717 us from the host; a mix turn 2,794 / 2,123 us (120 commands / 1) |

## Not at level 1

A rectangle per command; dynamic handlers and messages; a class's handlers called from an X280 thread, and a
group switched to a class's program (`kmx_load`); a core's own word written to DRAM;
tile expressions (`tile * factors`, `block @ self.weight`); `@`, slices and `.columns`
of a field.
