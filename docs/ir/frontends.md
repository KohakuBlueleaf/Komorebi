# A frontend's text and its origin

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page says how a frontend (a
    language that produces IR modules, such as Komorebi's Python classes) hands its modules to the
    core.

A frontend prints its module as IR text. The text is checked and compiled like any other module, so
every refusal of [the IR's checks](checks.md) applies to it, by name, before any backend runs.

The text does not say where it came from: a `.kom` text has no syntax for an origin. The frontend
gives its **origin** beside the text, and the core puts it in the module's notes (P in
[the byte-code program](bytecode.md)). There it changes only the full hash.

A `.kom` file is coloured in VS Code by the extension in `packaging/vscode/komorebi-ir/` (copy or
link the folder into `~/.vscode/extensions/`).

## What the origin holds

| Part | What it is |
|---|---|
| files | the frontend's source files, as it resolved them; a place names one by its index |
| lines | the source lines its places cover: (file, line number from 1, the line as written), in (file, number) order, each once |
| map | ranges of the text, each with the place of the source it was printed for: (text's first line, its column, text's last line, its column, file, first line, last line); columns count characters from 1, both ends held, and a range of whole lines runs from column 1 of its first to the end of its last |
| renames | (the name the text writes, the source's name), for values and parameters the source spells otherwise |

## From the text's positions to places

After the module is lowered, each anchored thing whose span starts at line L, column C of the text
takes the place of the **narrowest** range of the map that holds (L, C). An anchor that no range
holds has no place. The spans are then cleared: a frontend's module has an origin and no spans.

Two ranges are nested or apart. Two ranges over the same text, or two that cross, are refused.

A frontend therefore gives these ranges:
- one range over a whole declaration, mapped to the declaration's line. It places the declaration,
  its parameters and the lines nothing narrower holds.
- one range over each statement it prints, mapped to the source statement's lines. It places that
  statement's instructions.
- for a compound statement, the range runs from its first line to its closing brace. It places the
  loop or the branch itself, and its inner statements have narrower ranges of their own.
- for a call inside a statement that starts on a later source line than the statement, a range
  over the call's own columns, mapped to the call's lines. It places that call's instruction at the
  call's first line, the line today's tile trace names for the node a call makes (`caller_line`:
  the call's frame, its first line when the call spans several), however many calls share its
  line of the text.

A diagnostic of the text is placed by the same rule, at its start, so a refusal names the source's
line as well as the text's.

## The writer's layout

The text a frontend hands over is in the writer's own layout, the one `komo fmt` gives.
`komorebi_ir.layout(text, origin)` (`api::origin::layout`) lays a frontend's text out and carries
its origin: each range starts where its first character went and ends where its last went. The
layout moves only whitespace and commas, and takes away the backticks a name does not need (R50),
so the other characters are the same, in order, on both sides. A text where they are not is
refused by name, not guessed. A frontend writes every name it was given between backticks, and
the layout keeps them only where the language has a name of that spelling where the name stands:
where a name needs them is decided once, by the writer.

## Names

A source's name is written as given (R50): one that is a word of the language is written between
backticks where it needs them, not renamed. `renames` gives a value's name in the source where it
is spelled otherwise than in the text: the class frontend's var `h` of the field `self.h`. After
lowering, each name of a value, a slot or a parameter that is a safe name is replaced by its source
name. The renames are applied all at once, so a source's name may be another's safe name. A renamed
name stays in P, which only the full hash covers.

## What is refused

An origin is refused by what is wrong with it, before the module is used:
- a range at line or column 0, or one that starts after it ends;
- a range in a file past the files;
- two ranges over the same text, or two that cross;
- a safe name given twice, or one that names no value and no parameter;
- whatever the module's own rules refuse of an origin (`K954` in [the byte-code program](bytecode.md)),
  such as a place on a line the lines do not hold.

## From Python

`komorebi_ir.check(text, origin)` and `komorebi_ir.compile(text, origin)` take a
`komorebi_ir.Origin(files, lines, map, renames)`. A diagnostic then has a `place`: the file, and its
first and last lines. A refused origin raises `IrError`, and its message says what is wrong.

## Komorebi's classes

The text the frontend writes for each program it takes end to end is kept in `examples/ir/`.

`komorebi.dsl.ir.built.module(cls, **make)` writes a class written as today's DSL writes it (a
`km.Program` with `km.handler` methods) as an IR module. It reads the handlers with `ast`, as the
class compiler does, and gives a `Made`: the text in the writer's layout and `made.origin` for
`komorebi_ir.compile`. The walk itself (`komorebi.dsl.ir.written`, all that `komorebi.dsl.ir`
exports) does not import `komorebi_ir` and gives the text as it writes it; `built.module` lays it
out with `komorebi_ir.layout`. `built` imports the walk, never the other way. Today's make calls
neither.

It takes the lines of the first lesson and of the corpus's programs 1 (`Scalars.line`), 2
(`StepHandlers.sum`), 3 (`StepHandlers.add`), 4 (`Gate.gate`) and 5 (`Movement.read` and
`Movement.copy`):
- an int value (`first = self.rank * TILES`): int parameters and values, numbers, a module's int
  constant as its number, `self.rank`, `self.size` (`group.size`), `+ - * // %`, written
  `let first = worker.rank * 64`;
- a partition of a tensor's pages;
- a write stream and a read loop, each over a partition or over `t.pages(first, count)`, written
  `t.tiles(first, count)`. A view may be named first (`pages = source.pages(a, b)`, written
  `let pages = source.tiles(a, b)`) and streamed by its name;
- a read stream's `repeat=r`, which is no word of the IR: a counted loop around the stream's loop,
  `for i in range(r) { for _ in stream(pages, block = BLOCK) { } }` (the counter renamed when the
  class uses `i`). Only a loop over one stream whose body is `pass` is repeated. A repeat of a loop
  with a body, or of two streams zipped, is refused by name;
- a loop whose body is `pass`, written with an empty body (its blocks moved and dropped);
- a loop over two read streams zipped (`for x, y in zip(km.Stream(...), km.Stream(...)):`),
  written `for x, y in zip(a.tiles(...), b.tiles(...))`. Three streams, a write stream or names
  other than one a stream are refused by name;
- a push of a tile function's result, or of a tile expression (`results.push(x * scale)`), which
  is written as the function `tile_line_<line>` of the names it reads, as today's class compiles
  it; or of the loop's block itself (`out.push(tile)`, a copy);
- a tile of zeros (`total = km.tile.zeros(km.fp32)`), written `var total: tile<f32> = zeros`, and
  the loop's tile added to it (`total += tile`, a stream of one tile a block);
- `whole = km.reduce(total, form="tree")`, written `let whole = reduce(total, with = sum)`. The
  grouping is the backend's, so `form="tree"` writes nothing. `form="owner"` is refused by name
  until a `tensix.*` hint names it;
- `if self.rank == self.leader:` with no `else`, written `if worker.is_leader`, and in it the
  result's page write (`t.pages(0, 1).write(whole)`), written `store t.tiles(0, 1) <- whole`.
  `self.leader` as a number is refused by name.

A handler's docstring, as a tile function's, writes nothing.

A stream with no `block=` is of one tile a block, as today's. A class's int constant
(`BLOCK = km.constant(8)`) is written as a module parameter fixed at build,
`param BLOCK: u32 = 8 @bind(build)`, its value the make's (`BLOCK=4` at the make writes 4). It
reaches `compile` in the text: a parameter's value is a compile input. `block=self.BLOCK` is
written `block = BLOCK`. A module's int (`block=CT`, `CT = 4` at the module's top) is written as
its number, `block = 4`, as a word's is. A constant of another kind is refused by name.

A `km.Pattern` parameter is written as a `u32` parameter. A stream's `pattern=` naming it is
written as the hint `@tensix.pattern(p)`: on the loop for its read streams (which must give the
same one), and on the sink's `let` for the write stream. On the path through the IR, `args(...)`
takes a `km.Pattern` (or its letters) for it and gives today's word. Any other value is refused
before anything is sent. A `pattern=` of anything but such a parameter is refused by name.

A tile function is one `return` (a docstring before it writes nothing) of `+ - * /` of its
parameters and numbers, the tile compiler's functions of one tile by their names (`km.exp(x)` is
written `exp(x)`; `exp`, `exp2`, `sigmoid`, `silu`, `tanh`, `erf`, `log`, `sqrt`, `rsqrt`,
`reciprocal`, `relu`, `square`, `floor`), and calls of a plain function of its module on its
parameters (a helper), which is written as a `func` of its own, its lines its own. A call with a
keyword (`exp(x, error=...)`) and a helper called on other than parameters are refused by name.
A tile function's `tolerance=` is taken and writes nothing. The IR states no precision, and
`tolerance=` changes none of today's files: it is a check of the trace, in `program.txt` alone.
`fidelity=` is refused by name.

A stream of blocks of n tiles gives the loop `tile<F>[n]`, so its tile function is declared on
blocks (`tile<F>[n]` for the block and the result). Every other line is refused by name, with the
class's line, and so is a make keyword that would change the module. The make's tensors
give the module's formats; a tensor's pages must be one tile each, since an IR view is in tiles.
`l1=`, `depth=` and `log=` are not part of the module.

A partition the class applies to a second tensor (`y[src]`) is written as the same range of that
tensor, partitioned alike. The Tensix backend prints one partition for the two.

### A class's program through the IR

`komorebi.dsl.ir.built.program(cls, machine=None, **make)` builds the class's `km_program.Program`
(a `Built`) through the IR, apart from today's make, which it neither calls nor changes. It imports the
komorebi-ir wheel, which `komorebi.dsl.ir` does not.

1. `module` gives the module, which is compiled with its origin.
2. The Tensix backend emits the module's files and kernels.
3. `komorebi_ir.load` takes the make's tensors by parameter name. Each tensor gives its pages,
   its layout words, its page in DRAM and its shape in tiles. The load is on `machine` (a `komorebi_ir.Machine`; None:
   one island). It gives the thread header, the program's header as the machine sizes it (in
   place of the compile's), the kernels with each tensor's layout in its place, the buffers,
   their formats by the loader's names, and the files the output names as made at load (a
   movement source that walks a rectangle in rows of its tensor's width).
4. The files go to `build/<name>-<text_key>` beside the class's source, where today's make
   writes the same files.
5. The Program is made of that output:
   - a kernel's source for each processor;
   - each pair of buffers as a `BufferPair`;
   - the movers' compile arguments as one `movement` list;
   - the compute kernel's options;
   - the handlers' header;
   - the make's `depth=`.
6. Each handler is made from its entry point: `entry(name)` is a handler `Run.submit` takes,
   and its `args(...)` gives the command words of a call, as today's make gives them. A range
   that reads the rank (program 1's `a.tiles(worker.rank * 64, 64)`) is checked for every rank
   of the group, `args(..., group=(first, count))` as the submit's, and refused at the rank
   that does not fit. A rectangle's rows are checked against its tensor's rows of tiles and its
   columns against its width, and a tensor of another width than the load's is refused, in
   today's words (program 6).

A field (`self.out: km.L1[km.fp32, 1, WIDTH, 32, 32]`) is placed by today's own rule
(`compiler.field_users`, `kept_field`). A field today does not keep between commands is the
handler's `var` of a block of one row from `zeros`, named as the field (`self.out` is `out`,
renamed back in the origin), its tiles the field's columns. A field today keeps (filled by one
handler and read by another) is a state member, which the IR module does not write yet, and is
refused by name, as is a field with a value. A stream's block of B tiles over a rectangle of several rows
and C columns, both known at the make, is written `[A, C]` when B is A * C, A above 1 (A as its
declaration names it, `W_BLOCK = CHUNK * WIDTH` is `[CHUNK, WIDTH]`, else its number), and as B
when B divides C, the rectangle then walked a row after another; any other is refused by name.
A side `r + k : r + k + 1` is one row, as `r : r + 1` is. A loop zips two streams or more, and
`t.tiles[r0:r1, c0:c1].write(f(self.h, self.g))`, f a tile function of fields, stores `f(h, g)`
of the vars (program 7, `SwiGLU.up` and `SwiGLU.down`). A name is the name the class gives it
(R50): the class, a handler, a constant, a field, a parameter, a local or a tile function named
by a word of the language is written as given, between backticks where it needs them (a field
`store` is `` `store` ``, a handler `zip` is `handler zip`), never renamed and never refused. A
name the walk makes itself (a `repeat=` loop's counter, a partition restated on a second tensor,
a tile expression's function `tile_line_<line>`) takes `_` until it is no name the handler uses.

A `km.chain` method is written as a `chain` of the program, its calls in order, each
`self.h(names..., cores=(first, count))` a `call h(names...)`: R47's form, with nothing of how
Tensix keeps the order (R49). The chain's group is the span of its calls' cores, from the lowest
first core to the highest end. A call over the whole span has no `@on`; any other is
`@on(group[a:b])`, its ranks counted from the span's first core. Program 7's `forward`, `up` on
`cores=(0, 12)` and `down` on `cores=(0, 4)`, is `call up(x, w1, w3, hidden)` and
`call down(hidden, w2, y) @on(group[0:4])`. A parameter's type is the one the handlers give it.
`made.chains` keeps each chain's span, (first core, count), for its entry. A call without
`cores=`, or with cores the make does not know, a keyword other than `cores=`, an argument that is
not a parameter of the chain (`km.past`, `km.direct`), and a line other than such a call (a loop,
`km.all_gather`) are refused by name with the chain's line.

`chain(name)` of the `Built` is the chain's entry: its `calls(...)`, given the chain's arguments,
are today's `kc.Call`s of the chain on the session's cores of its span (a call's `@on` counted
from the span's first core), each call's words its handler's with the mask word after them.
`built.chain(run, entry, ...)` gives them to `run.chain`, which proves them and writes their
records as today (`kc.build`, `kc.records`), and runs the command on the chain's entry. For
program 7 the records are today's word for word, on a span from core 0 or core 4. An outside read
carries no range (today's gives `x` one); a record does not hold the reads.

For the first lesson, each of these fields is today's, and the folder is the same. The path
through the IR checks each range a handler's entry point states at submit, refusing a range past
its tensor's end before anything is sent; today's make checks only the ranges of `items(...)`.

### On islands, not yet

What depends on the machine is sized at load from its islands ([backends](backends.md)): a tree
reduce's arrival slots, today's `tree_slots`. Every other place today's make reads a session's
islands (`compiled.islands`) is in a construct the IR path refuses by name until a program needs
it:

| Today | What it sizes | On the IR path |
|---|---|---|
| `dsl/compile/tiles/reduce.py:907` | a reduce over a group of `size` (`island_reduce.group_slots`) | a reduce's `group=` is refused |
| `dsl/compile/tiles/reduce.py:904` | `reduce_local=True`, every group on one island | a reduce's `reduce_local=` is refused |
| `dsl/program.py:277` | a chain's K shared by island (`KM_CHAIN_K_ISLANDS`, `KM_CHAIN_BY_ISLAND`) | `chain=` is refused; a class's chain is written as one group (below) |
| `dsl/program.py:1729` | the link cores' buffer index kept from a direct input (`KM_LINK_CB`) | `direct=` is refused |
| `dsl/program_sources.py:100` | the island table after a rectangle's grid words | `rect=`, `nocs=` and a rectangle parameter are refused |
| `dsl/compile/handler/handler_compile.py:365`, `:369` | a rectangle's grid and place on islands (`rect_glue`) | a rectangle parameter is refused |
| `dsl/compile/casts/broadcasts.py:738` | a broadcast's place on islands | `km.broadcast` is a line the walk does not take |
| `dsl/compile/pages/ranges.py:52` | DRAM items not of whole 64 B refused on islands | unreachable while a page is one tile: every tile size is a whole number of 64 B; a page of other items needs this check |
| `dsl/first_use.py:112`, `:161` | `km.ANY` handlers compiled on first use | `handlers=` is refused |
| `dsl/merged.py:609` | `km.merge` | a path of its own the IR does not take |
