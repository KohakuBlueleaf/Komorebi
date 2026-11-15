---
title: Programs written as classes
summary: Experimental. A class deriving km.Program whose methods are handlers, compiled when it is made into an ordinary program's files (the movement source, the compute kernel, the handlers header, an overlay's parts); the lines a handler may hold and the line each lowers to, overlays, islands and what may cross a link, the refusals, and the examples measured against the hand programs.
tags:
  - reference
  - host
---

# Programs written as classes

Status: experimental. The lines of this page are those of levels 1 and 2 (tile
functions); seven example scripts run as classes beside their hand programs, on one
Blackhole card and on two ([below](#measured)).

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
| `word = pages.read()` (one page) | mover B: `km::read_pages` of the page into its word page, the barrier, the L1 cache invalidated, the page's first word | |
| `word += round + page`, `word` a mover's | the line on the word's mover | |
| `pages.write(word)` (one page), `word` a mover's | that mover: its word page (the word first, zeros after), `km::write_pages`, `noc_async_write_barrier` | |
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

## Tile functions

A **tile function** is a Python function of tiles that the tile compiler
([Tile functions](tile.md)) traces and emits for compute: decorated with
`@km.tile_function`, at module level or in the class's body (called `self.f(...)`), and
called in a stream loop on the loop's tiles. Its arithmetic is the tile compiler's
stream form: `+`, `-`, `*`, number constants, `trace.silu`, `trace.sigmoid`,
`trace.rsqrt` of up to three tiles, one result. From
[22 gate](../../examples/22_gate/as_class/gate_class.py):

```python
@km.tile_function(tolerance=TOLERANCE)
def silu_gate(a, b):
    """The gate of two tiles: what the tile compiler emits for compute."""
    return trace.silu(a) * b


class Gate(km.Program):
    """gate: each core's `tiles` pages of a and b from rank x tiles, silu(a) * b into
    the same pages of out, block by block in the command's pattern."""

    @km.handler
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        first = self.rank * tiles
        results = km.Stream(
            out.pages(first, tiles), write=True, block=BLOCK, pattern=pattern
        )
        for x, g in zip(
            km.Stream(a.pages(first, tiles), block=BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=BLOCK, pattern=pattern),
        ):
            results.push(silu_gate(x, g))
```

| Line | Movers | Compute |
|---|---|---|
| `for x, g in zip(km.Stream(...), km.Stream(...)):` | the read streams in one `movement.move` with the write streams the loop pushes to | block k of each stream from the half its mover filled; the streams have one count, block and repeat |
| `results.push(f(x, g))`, `t = f(x, g)` | | once before the blocks `copy_init` and the body's setup (`km_tile::f_init()`); a tile: DST held, each operand copied into DST in the order of the unit program's operands, `km_tile::f(0)` on the math CPU, DST 0 packed into the write stream at the tile's place |

The build traces the function on tiles of the streams' formats (`unit_program(ir,
load_macro=True)`), writes the body as `tile_<f>.hpp` and its unit program as text,
`tile_<f>.txt`, and prints its cost beside the buffers: the instructions and cycles a
tile on the math CPU by the cost model (the vector unit only, the copies and the pack
not priced), the rules that fired, and the tolerance the decorator states
(`tolerance=`, the largest relative error against float64). A loop that calls a tile
function holds no other tile line: the body's setup holds the vector unit's constant
registers for the loop.

`@km.handler(stamps="t")` on a handler that calls a tile function makes the math CPU
read the wall clock around each call between two `tensix_sync()`, as the tile
compiler's measurements do, and writes page `rank` of `t` (uint32 pages of 19 words or
more): the command's sequence, the calls, 16 empty spans, then each call's span. From
the same example on the card: 768 calls, a median span of 1,666 cycles less the empty
span of 10, 1,656 cycles a tile, against the 1,657.6 the build prints.

What the tile compiler does not lower is refused, naming the handler's line, the
function and the compiler's sentence, with "write this body inline: km.cxx(...,
gives=km.Tile)". A matrix product `a @ b` in a tile function is refused: the tile
compiler has no lowering of one yet.

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
  mix: overlay 1: reader 64 B, writer 384 B; slots: reader 3 x 8,640 B, writer 2 x 3,968 B, unpack 1 x 2,624 B, math 1 x 1,664 B, pack 1 x 1,664 B
  scale: overlay 2: reader 3,968 B, writer 3,968 B, unpack 2,624 B, math 1,664 B, pack 1,664 B; slots: reader 3 x 8,640 B, writer 2 x 3,968 B, unpack 1 x 2,624 B, math 1 x 1,664 B, pack 1 x 1,664 B
```

`@km.handler(overlay=True, stamps="stamps")`, `stamps` a tensor parameter of uint32
pages of at least five words, adds to the case on each mover the wall clock around the
load and the call, and writes the command's sequence, the overlay, the load's and the
call's cycles and whether it was held to page 2 x core + mover of `stamps` (mover A 0,
B 1). From the same example, mix on mover B, median cycles (load, call): held 67 /
967, loading 1,779 / 1,007.

A broadcast's handler is resident.

## Dynamic handlers and messages

A handler with `for message in self.arrive():` is [dynamic](dynamic.md): its lines are
compute's alone and the runtime's movers carry its messages (`messages_reader.cpp`,
`messages_writer.cpp`); the program is made dynamic as any program is, with
`.dynamic(name, **options)`. A class has at most one dynamic handler, and a move in one
is refused. A message is a class deriving `km.Message` whose int annotations are its
words, 1 up (word 0 is the runtime's). From
[05 forward](../../examples/05_forward/as_class/forward_class.py):

```python
class Visit(km.Message):
    id: int
    forwards: int
    state: int
    sum: int


class Forward(km.Program):
    @km.handler
    def forward(
        self,
        message_count: int,
        forwards: int,
        mode: int,
        target_rank: int,
        results: int,
    ):
        if self.seeding():
            part = self.partition(message_count)
            for index in range(part.count):
                seed = Visit(id=part.start + index, forwards=forwards, state=1)
                self.send(self.rank, seed)
        for message in self.arrive():
            message.state ^= message.state << 13
            if message.forwards == 0:
                self.consume()
            else:
                message.forwards -= 1
                self.forward(message.state % self.size, message)


program = Forward().dynamic("forward", max_live=1 << 14, queue_messages=1 << 14)
```

| Line | Compute |
|---|---|
| `if self.seeding():` | `messages.seeding()`, its lines, then `messages.seeded()` |
| `seed = Visit(id=..., ...)` | 16 words, the named ones set, the others 0 |
| `self.send(rank, seed)` | `messages.send(rank, seed)` |
| `for message in self.arrive():` | each packet of `messages.next()`, each message of it copied, then `messages.done(packet)` |
| `message.state`, `message.sum += x` | the message's word |
| `self.forward(rank, message)` | `messages.forward(rank, message)` |
| `self.consume()` | `messages.consume()` |

In a dynamic handler `self.rank` and `self.size` are the message loop's
(`messages.rank()`, `messages.count()`): the group without its counter tiles. A core's
totals go where a `km.cxx` body puts them (05 writes them to an L1 area whose address
is a command word).

## X280 threads

Each build also writes `<name>_thread.h`, a C header for a thread's commands: each
handler's number, the place of each of its words (a0 is 0) and their count, and
whether its command asks for the leader's result. A thread that includes it (the
build folder, `program.built.folder`, in its `includes`) submits the class's commands
with `kmx_submit` and waits with `kmx_wait`. From
[07 work cluster](../../examples/07_work_cluster/as_class/README.md):

```c
/* SUM, line 35. */
#define WORKCLUSTER_HANDLER_SUM 2u
#define WORKCLUSTER_SUM_STATE 0u
#define WORKCLUSTER_SUM_CORES 1u
#define WORKCLUSTER_SUM_FIRST 2u
#define WORKCLUSTER_SUM_WORDS 3u
#define WORKCLUSTER_SUM_RESULT 1u /* KM_FLAG_RESULT */
```

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
Product.multiply, line 89, `results.push(product(x, y))`: the tile function product, line 81, has a matrix product (`@`), and the tile compiler has no lowering of one yet; make the product in a km.cxx body (gives=km.Tile) until it has
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
| [01 on two islands](../../examples/01_mapcollect/as_class/README.md) | the same: every tile exact; 2,360 asks and 4,720 pages over the links in 20 rounds, as the hand's | 60,048 / 61,152 B (mock, islands' build) | own groups 2.88 / 3.04 us a command; island 0 reading its own input 198.4 / 190.6 us a round, island 1's 626.0 / 615.5 (20 rounds): the dispatcher's 4.82 / 4.62 and 448.4 / 448.1, the class's submit 8 us longer |
| [02 merge tree](../../examples/02_merge_tree/as_class/README.md) | the same: 40 jobs, exact ones bit-equal, 40 result words | 42,240 / 36,176 B | on the cores 15.33 / 15.14 us a command (first seen to sent, median) |
| [03 broadcast](../../examples/03_broadcast/as_class/README.md) | the same: 960 ranks, all exact (level 1a) | 48,736 / 53,424 B (mock) | 109.0 / 99.8 us a job, exclusive (level 1a) |
| [05 forward](../../examples/05_forward/as_class/README.md) | the same: exact in 3 of 3 runs, 10^6 forwards | 41,600 / 43,040 B | 12,162 / 12,297 us a run (median of 3) |
| [07 work cluster](../../examples/07_work_cluster/as_class/README.md) | the same: 204 rounds, sum 1,004,917, every state word | 21,200 / 21,424 B (mock) | the thread's loop 10.55 / 10.75 ms |
| [11 movement](../../examples/11_movement/as_class/README.md) | the same: every copy exact (level 1a) | 38,288 / 36,720 B | READ AAAB 308.74 / 310.44 GB/s (level 1a) |
| [12 overlays](../../examples/12_overlays/as_class/README.md) | the same: every mix word and scaled value | 32,584 / 27,324 B, and the parts above | a turn from the host, one command each: scale 1,353 / 1,731 us, mix 1,940 / 2,095 us |
| [22 gate](../../examples/22_gate/as_class/README.md) | the hand program's bound: 0 wrong of 786,432; float32 against float64: 2.46e-7 (tolerance 1e-6) | 37,712 / 38,640 B | 86.68 / 76.06 us a command (bfloat16, 12 cores, 64 tiles a core); the emitted body 1,656 cycles a tile on the math CPU, 1,657.6 printed |

## Not in this version

A rectangle per command; tile arithmetic outside a tile function (`tile * factors`
in a handler's line); a matrix product (`block @ self.weight`); slices and `.columns`
of a field; a message that names its handler (several dynamic handlers of one loop).

A group switched to a class's program (`kmx_load`): a load switches between programs
in the runtime's form, parts on the runtime's reader, writer and compute, and a class
is built in the kernel form, its streams moved by `km::Movement`. A class in that form
needs its movement lines to run in parts on the runtime's reader and writer, and the
runtime's switch to give each part its buffers; that is a change of the runtime, not of
the compiler.
