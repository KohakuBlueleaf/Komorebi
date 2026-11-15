---
title: Programs written as classes
summary: Experimental. A class deriving km.Program whose methods are handlers, compiled when it is made into an ordinary program's files (the movement source, the compute kernel, the handlers header); the lines a handler may hold, where each runs, the buffers the build chooses, the refusals, and the four examples measured against the hand programs.
tags:
  - reference
  - host
---

# Programs written as classes

Status: experimental. The lines of this page are those of level 1a; four examples run
as classes on one Blackhole card beside their hand programs ([below](#measured)).

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

```python
program = MapCollect(source=inputs[0], output=outputs[0], depth=4)
run = session.load(program)
words = MapCollect.map.args(inputs[1], 64, outputs[0], 0, 128)
handle = run.submit(program.map, (0, 8), words)
```

| Made with | What it is |
|---|---|
| a tensor by each tensor parameter's name | an example of the tensor: its format and page size (the buffers', the movers' compile arguments) |
| `plan` | the sizes by name: a module constant the handlers use (`BLOCK`), a field's symbol |
| `l1` | the bytes left for buffers (`Session.l1()[1]`): the buffers must fit, a field's last symbol takes what they leave |
| `rect`, `nocs` | with a broadcast or `self.x`, `self.y`: the rectangle (x, y, columns, rows) and `Session.nocs()`; each core's program arguments are made from them |
| `depth`, `log` | the ring slots used, as a `Program`'s; where the build's report goes (`print`) |

The build writes `build/<name>/` beside the class's source: `<name>.h` (the handler
numbers, the buffers, the patterns), `movement.cpp`, `compute.cpp` and `program.txt`,
the report it also prints: each buffer with its indices, format, pages, bytes and the
Python line that made it, and the L1 they take. The files are kept; a compile error
names their lines, and an inline body's lines name the Python file's.

`Class.handler.args(...)` gives the command words of a call: a tensor's reference word
(`tensor.address`), an int (32 bits), a pattern's word, in the order of the
parameters; a tensor whose `pages` a line uses whole adds a word, its page count.
`program.handler` is the handler of that program for `Run.submit`.

## The lines of a handler

| Line | Movers | Compute |
|---|---|---|
| a parameter `name: km.Tensor`, `name: int`, `name: km.Pattern` | a command word | a command word |
| `n = a + b * 2`, `n += 1`, an `if` on ints | the int, where a kept line uses it | the same |
| `part = self.partition(pages)` | `km::partition` of the pages by rank and group size | the same |
| `total = km.tile.zeros(km.fp32)` | | DST held, tile 0 filled with zeros |
| `for tile in km.Stream(pages, block=8, pattern="AB", repeat=n):` | one read stream in a buffer pair, moved with the write streams the loop pushes to | each block from the half its mover filled, each tile of it |
| `total += tile` | | the tile copied and added into the held tile |
| `out = km.Stream(pages, write=True, block=8)` and `out.push(tile)` | one write stream, moved with the read that feeds it | the tile copied into the same half of the write pair |
| `pages.write(total)` (one page) | a write of one page by mover B | the held tile packed, DST released |
| `whole = km.reduce(total, form=form)` | `km::reduce_arrive` on mover A, `km::reduce_emit` on mover B | `km::reduce`, the arrivals added in rank order |
| `if self.rank == self.leader:` and `pages.write(whole)` | the leader's write of the sum | |
| `b = km.Broadcast(self, along="row", window=2)` | the broadcast on its mover (rows on mover A, columns on mover B); a core it leaves out skips the handler | the same core skips it |
| `if b.sender():` `v = b.cast(t.pages(first, n).read())` `else:` `v = b.arrive()` | the sender's reserve, read and cast, the others' arrival, block by block of the loop around it; the last block's finish | |
| `for tile in v:` | | the block received, each page of it, then freed; the padding after the last |
| `for i in range(n):` | a loop | a loop |
| `km.cxx("...", takes=[n], on="mover_b")` | the C++ as written, on the CPU `on` names | compute when `on` is not given |
| `self.result(word)` | mover B's result word | |

`self.rank`, `self.size` and `self.leader` are the core's rank in the command's group,
the group's size and its rank 0; `self.x` and `self.y` its column and row in the
rectangle.

A stream's pattern is the reads' `AAAB` and a write's `B` unless the line gives one. A
read pair holds 4 blocks a half and a write pair 2; a reduce's arrival buffer 118
float32 tiles (one slot for each other core of a 119-core group by owner); a
broadcast's buffer `window` blocks of the pages it casts. Buffer indices are 0 up for
reads, 8 up for arrivals, broadcasts and fields, 16 up for writes; 26 to 31 are the
runtime's. Compute's tiles are in DST as float32; a float32 buffer is unpacked to it
unrounded.

## Fields

A field is declared on the class with its format and shape:

```python
class Sum(km.Program):
    count: km.L1[km.fp32, ("TILES", 32, 32)]
```

Each field is a buffer of whole tiles, index 8 up. A symbol of its shape is the plan's;
one symbol left unknown is solved as the largest for which the fields and the buffers
fit `l1`, and the report prints it. At level 1a no line uses a field: a line that does
is refused, and a `km.cxx` body reads it by its buffer (`kFieldCount`, its pages
`kFieldCountPages`).

## Refusals

A line the compiler does not take raises `Refused` (a `ProgramError`) when the class is
made, naming the handler, the line and what to write instead (two classes of
[test_classes.py](../../tests/core/test_classes.py)):

```text
Sliced.copy, line 48, `for _ in km.Stream(source[1:4]):`: a tensor is in DRAM: a slice is not a value; take a part: source[self.partition(...)], or use a stream or read()
Unled.merge, line 59, `output.pages(0, 1).write(whole)`: whole is the reduce's sum, which only the leader holds; write it under `if self.rank == self.leader:`
```

`komorebi.dsl.program.lines_of(program, error)` names, for a `CommandTimeout` of a
program made from a class, each core that did not report with the handler and the
Python line its CPU was waiting in.

## Measured

Each class runs in the same session as its hand program and from the same seed, on one
Blackhole card (the scripts' `--beside`). Kernel text is the packed text of the five
CPUs, of 70,656 B.

| Example | Results | Kernel text, class / hand | Time, class / hand |
|---|---|---|---|
| [01 map/collect](../../examples/01_mapcollect/as_class/README.md) | the same: 20 jobs checked, 10 exact | 34,256 / 34,288 B | host 28.8 / 34.7 us a job, 8 cores; 6.56 / 2.76 us a command, 120 cores, a tile or none a core |
| [02 merge tree](../../examples/02_merge_tree/as_class/README.md) | the same: 40 jobs, exact ones bit-equal | 43,360 / 36,176 B | 42.0 / 35.4 us a job, exclusive |
| [03 broadcast](../../examples/03_broadcast/as_class/README.md) | the same: 960 ranks, all exact | 51,216 / 53,424 B | 109.0 / 99.8 us a job, exclusive |
| [11 movement](../../examples/11_movement/as_class/README.md) | the same: every copy exact | 38,288 / 36,720 B | READ AAAB 308.74 / 310.44 GB/s |

## Not at level 1a

Handlers chosen at run time, overlays, calls to X280 cores, islands, tile expressions
other than `total += tile` and `out.push(tile)`, a line that uses a field, and a
`km.cxx` body that gives a value (`gives=`).
