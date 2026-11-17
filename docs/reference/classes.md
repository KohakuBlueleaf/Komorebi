---
title: Programs written as classes
summary: Experimental. A class deriving km.Program whose methods are handlers, compiled when it is made into an ordinary program's files (the movement source, the compute kernel, the handlers header, an overlay's parts); the lines a handler may hold and the line each lowers to, overlays, islands and what may cross a link, the refusals, and the examples measured against the hand programs.
tags:
  - reference
  - host
---

# Programs written as classes

Status: experimental. The lines of this page are those of levels 1 and 2 (tile
functions); the scripts [below](#measured) run as classes beside their hand programs
or the tile compiler's measurements, on one Blackhole card and on two.

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
| a constant of the class by its name (`K_TILES=256`) | its value for this program ([Constants](#constants)) |
| `plan` | the sizes by name: a constant of the class, a module constant the handlers use (`BLOCK`), a field's symbol |
| `handlers` | by handler, the sets of its constant parameters to build, each a handler of its own (`handlers={"gate": [{"block": 4}, {"block": 8}]}`) |
| `session`, `trace` | the session it is loaded on and the trace pages its load is given: the build checks the load's L1 table of a core (`Session.l1_rows`: the buffers with the token and trace pages, the control block and overlay area, the host transfer buffer, what the session keeps) with the service overlay's slot at its most (`overlay.SERVICE_SLOT_MOST`), so a class that builds loads; a class's own overlays' slots are the load's; a field's last symbol takes what is left |
| `l1` | without a session, the bytes left for buffers (`Session.l1()[1]`): the buffers alone must fit, a field's last symbol takes what they leave |
| `rect`, `nocs` | with a broadcast or `self.x`, `self.y`: the rectangle (x, y, columns, rows), the whole grid when not given, and the grid's NoCs, the session's (`Session.load` gives them) when not given; each core's program arguments are made from them at the load |
| `depth`, `log` | the ring slots used, as a `Program`'s; where the build's report goes (`print`) |
| `broadcast_wait` | as a `Program`'s ([host](host.md#program)), `False` by default: `True` builds the kernels with a word broadcast's wait at `finish` (the guard), in a build folder of its own |
| `guard` | as a `Program`'s ([host](host.md#program)), `True` by default: `False` builds the kernels without a command's deadline, give-up and inspect ([hazards](hazards.md#a-wait-that-never-completes)), in a build folder of its own |
| `form` | `"kernel"` (the default): the class owns its kernels; `"runtime"`: the class is a program in the runtime's form, in parts the runtime's kernels call, which loads in a list or a dict and which a group switches to ([The runtime's form](#the-runtimes-form)); each form has its build folder |
| `beside` | with `form="runtime"`, a program or a list of them that share the run's cores: the class leaves their buffer indices and takes their unpack modes ([The runtime's form](#the-runtimes-form)) |

The service overlay's slot is reserved at `overlay.SERVICE_SLOT_MOST`, 9,216 B: the
service's reader part built for every example and project that builds without a card
is 8,640 B (with the islands defines too), and 8,840 B in the build that checks the
core-table indices (`KM_CHECK_NOC_SOURCES`); rounded up to 9 KiB. The load does not
refuse on it: it refuses only by its own L1 table, made from the real sizes, and logs a
service part over the constant. The build's table shows the reserve, the load's the real
slot (8,640 B), so a class has the 576 B between them to spare. A class made with an
island session (a card split, or two cards) reserves `overlay.ISLAND_SERVICE_SLOT_MOST`,
43,008 B, and checks the first island's table: the island loads' service part has the
links' and the tensor table's services built in, at most 41,856 B logged (two cards;
39,512 B on a card split in three, 39,552 B in two), and 1,152 B of margin. The load
logs a part over the reserve. The host transfer row is the load's own: its chunk counts
the page of the dispatcher's state area (40,960 B), which every load makes before the
transfer buffer. On card 0 split in two, example 01's class: the build's table and the
load's are equal in every row but the slot (43,008 B reserved, 39,552 B loaded): host
transfer buffer 82,000 B in both (the build counted 32,848 B before).

The build writes `build/<name>-<hash>/` beside the class's source, `<hash>` sixteen hex
digits of the generated files' names and text (not of the make's inputs: the plan,
the tensors, the constants and the sets that give a file its text are in the text): two
instances whose files are equal share one folder, whoever made them and whatever else
differed (a mock device's make and a card's), and files that differ are never one
folder. A part's cache key and the kernels' own hash carry the folder's path, so a
class made on a mock device is the card's build: a job that loads it compiles no
kernel and no overlay part. A file the folder already holds equal is not written again. In it: `<name>.h` (the
handler numbers, the buffers, the patterns), `movement.cpp`, `compute.cpp`, an overlay
handler's parts in `<handler>/`, and `program.txt`, the report it also prints: each
buffer with its indices, format, pages, bytes and the Python line that made it, and the
L1 they take. The files are kept; a compile error names their lines, and an inline
body's lines name the Python file's.

`Run.submit(Class.handler, group, args=(...))` makes the command words from the
arguments: a tensor's reference word (`tensor.address`), an int (32 bits), a float
(`scale: float`: the value's float32 bits; a number that is no float32 or not a number
is refused, naming the parameter), a pattern's word, in the order of the parameters (a constant parameter's value, at its place, is
no word: it picks the handler built for it); a tensor whose `pages` a line uses whole adds a
word, its page count. A handler that returns a word asks for the result. For arrays of
words (`Run.submit_many`), `Class.handler.args(...)` gives one command's words, and
`submit_many(Class.handler, firsts, counts, words)` takes them. A tensor whose
rectangles (`t.tiles[...]`) the build wrote with the example's tile width is refused at
submit when its own width is another, naming both.

The words of a handler (its int parameters, the ints it computes, a page's words) are
unsigned 32-bit, as the C++ they become: a subtraction below 0 wraps to a large word,
so `a - b > 0` is refused (write `a > b`), and `a or b`, `a and b` of words, a word in
Python and a truth in C++, are refused (write `max(a, 1)`, or an `if`). A parameter is
not assigned again, and a name first assigned inside an `if` or a loop ends with it, as
in C++ (assign it before the `if`). The generated code's own names start with `km_`,
and a name of the handler's with that prefix is refused.

## Constants

A constant is a number fixed when the program is built: a stream's block, a field's
size, a loop's count, a rectangle's most. It is declared in one of two places: on the
class, or as a handler's parameter.

### A constant of the class

```python
class Linear(km.Program):
    K_TILES = km.constant(128)  # K = 4096
    WIDTH = km.constant(16)
    CHUNK = km.constant(2)
    W_BLOCK = CHUNK * WIDTH  # derived: follows an override

    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        first = self.rank // self.SPLIT * self.WIDTH
        ...


wide = Linear(x=x, w=w, y=y, K_TILES=256, session=session)
```

`km.constant(default)` in the class's body is a declaration: its name, its default and
its type (an int, a float or a bool). Arithmetic on declarations in the body (`+`,
`-`, `*`, `//`, `%`, `/`, a negation, with numbers or other declarations) is a derived
declaration. The class holds the declarations only, and making a program never changes
one: `Linear.WIDTH` is the declaration on the class, and `wide.WIDTH` (and `self.WIDTH`
in Python) that program's number.

The make collects the declarations of the class and of its parents, takes an override
by a keyword of the declared name (`K_TILES=256`) or by name in `plan` (one of the two
for one name), evaluates each derived constant from this program's own values, and
keeps the values on the program, frozen: assigning one on a program is refused, and two
programs of one class with different values do not touch each other or the class. A
derived constant's names are resolved through the program, so a subclass that declares
`WIDTH = km.constant(32)` changes the `W_BLOCK` its parent derived; the subclass has
its parent's handlers, fields and options, and a tile function of the class's body it
declares again is the one its parent's handlers call (`self.f(...)`). A name a parent declares, rebound to a plain
value in a subclass, is refused when the subclass is made.

In a handler, `self.K_TILES` is the constant: the compiler writes its value where it
writes a module constant's, so a block, a count or a most may be one. A field's shape
may hold declarations (`km.L1[km.fp32, 1, WIDTH, 32, 32]`); the shape keeps them on
the class and each program's build takes its values. `km.constant()` without a
default is a symbol, as `km.symbol()` is: the plan (or the make's keyword) gives it, or
the build solves it as the largest that fits, and `self.TILES` of one the build solves
is refused in a handler's lines. Module constants and the plan's names stay as they
were: a module constant a handler names is the plan's when the plan gives it.

The build key holds the class's constants by name and value: two programs with
different values are two builds, side by side under `build/`; equal values share the
folder.

A line that needs a constant and is given a word of the command (a stream's block, a
field's size, a rectangle's stated most `r:r + n` or `r:min(r + n, e)`) is refused at
its line, naming the word and the tier it needs:

```text
Worded.walk, line 197, `km.Stream(x.pages(0, tiles), block=tiles).drain()`: tiles is a word of the command, known only when the command runs, and a size the build needs is fixed when the program is built: it needs a constant of the class; declare TILES = km.constant(n) in the class and write self.TILES
```

With two programs of one class in one run, the class's handler (`Linear.split`) does
not say which program a command is for: it is refused, naming both, and the handler is
named through the program (`wide.split`). With one program of the class in the run the
class's handler names it, as before.

```text
Linear.split: the run holds 2 programs of the class (Linear(K_TILES=4, WIDTH=2, CHUNK=1, W_BLOCK=2) at field 0; Linear(K_TILES=8, WIDTH=2, CHUNK=1, W_BLOCK=2) at field 1), and the class's handler does not say which; name it through the program (program.split)
```

Refused at the make, besides: a keyword that is a number and no constant of the class
(`K_TILE=8 is not a constant of the class (its constants: ...)`), a value of another
type (`K_TILES=2.5: the constant is declared int`; an int is a float's value, a bool
only a bool's), one name given by keyword and by the plan, and an override of a
derived constant (`W_BLOCK is derived from the class's constants; give the constants
it is made of`). A tuple is not a constant (`km.constant((1, 2))`): the compiler emits
no table of constants.

### A constant parameter of a handler

```python
class GateBlocks(km.Program):
    @km.handler
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
        block: km.constant[int] = 8,
    ): ...  # block=block in its streams


program = GateBlocks(a=a, b=b, out=out, handlers={"gate": [{"block": 8}, {"block": 4}]})
run.submit(program.gate, (0, cores), args=(a, b, out, tiles, pattern, 4))
```

A parameter annotated `km.constant[int]` (`float`, `bool`) with a default is not a
word of the command: the handler's lines read it as a module constant. With no set
listed the handler is built for its defaults. The make's `handlers=` lists, by
handler, the sets of its constant parameters (a parameter a set leaves out takes its
default); each set is a handler of its own in the kernels, named by its values
(`gate_block8`, `gate_block4`), the first at the handler's number and the others after
the class's handlers. A submit gives the constant's value in `args` at its place, as
the other parameters are, or leaves it out for its default; the host runs the handler
built for those values. The words `program.gate.args(a, b, out, tiles, pattern, 4)`
gives are a list as before that also carries the values stated, so `submit_many`,
`Graph.command`, a stream's `submit` and `enqueue`, and a chain's call
(`kc.Call(program.gate, words, ...)`) run the handler built for them. A slice, a
`copy()` and a plain list added on either side keep the set; words of two sets added
together, or repeated, are refused naming the handler; plain words (a list rebuilt
with `list(words)` too) mean the defaults. The rows of one
`submit_many` state one set. A graph and a stream have no run to find the class's
program in: they take such a handler through its program (`program.gate`), and
`Class.gate` is refused there. A set that was not built is refused, naming the sets
that were:

```text
Gate.gate with size=2: no handler was built for it (built: size=4; size=8); list it at the make, handlers={'gate': [{...}]} (a set is not compiled on first use)
```

Refused besides: a constant parameter without a default (at its line), `handlers=`
naming no handler of the class or one without constant parameters, a parameter a set
names that is not one, a value of another type, and a set listed twice.

### A set compiled on its first use

```python
sets = [{"block": 8}, km.ANY(block=16)]  # block 8 built; any other at its first use
program = GateBlocks(a=a, b=b, out=out, handlers={"gate": sets})
run = session.load(program)
words = (a, b, out, tiles, pattern, 4)  # block 4: compiled, added, submitted
run.submit(program.gate, (0, cores), args=words)
```

`km.ANY` among a handler's sets (`{"gate": km.ANY}` alone, or after listed sets) lets
`run.submit` take any set of that handler's constant parameters. A set that was built
runs as above. A set that was not is compiled at the submit by the class compiler,
from the make's own inputs (plan, tensors, constants, the L1 left), with the wanted
set in place of the handler's first set, its **template** (the first listed set, or
the defaults). So it gets the template's number and its buffers' indices. Its code
becomes the handler's overlay parts (`<set>/movement.cpp`, `<set>/compute.cpp`), which
are added to the run by `run.addon(..., handler=True)`
([overlays](overlays.md#code-added-to-a-live-run)). Then the command is submitted
with the added number. Later submits of the set find it added. The files go into
`build/<class>-use<key>/`: the key holds the load's folder and make key, the handler, the set's
values, the class's source files and a hash of the package's Python source (the class
compiler that wrote them: a changed compiler compiles the set again). A later process that makes the same program finds
them there (their digest, kept in `first_use.json`, is compared with the files) and reads the set's words from it, so nothing is compiled.
Its parts are then found in the parts' cache, so nothing is built either.
`program.ahead(Gate.gate, size=4)` compiles a set before the run and has the load
build its parts ([overlays](overlays.md#code-added-to-a-live-run), built ahead).

What the opt-in costs and requires, checked at the make:

- The kernels get the added handlers' path (`KM_ADDED_HANDLERS`, with
  `Program.slots(bytes=)`). On the gate's kernels it costs 5,072 B of text
  (BRISC 408, NCRISC 1,064, TRISC0 1,160, TRISC1 1,240, TRISC2 1,228; 2,560 B on the
  decode program). The kernels' sizes are checked when they are built, at the load,
  before any submit.
- Each kernel part gets a slot of 16 KiB (`first_use.SLOT_BYTES`; the reader's beside
  the service part, the larger), 80 KiB of L1. A make with `session=` whose buffers
  and those slots do not fit is refused there (`L1BudgetError`, with the table).
- `km.ANY(p=n)` also sizes the template's buffers for the set with `p=n`: each buffer
  takes the larger page count of the two compiles. `km.ANY` alone sizes them for the
  template only. Buffers that no longer fit the L1 left are refused at the make, with
  their bytes. A program pays for the reserve even if it never submits a larger set.
  The gate's buffers take 327,680 B of L1 for block 8 alone and 655,360 B with
  `km.ANY(block=16)`: 327,680 B more on every core.

A set compiled this way may use only what the load has: the buffers, fields and
streams of the class's L1 layout. It is refused at its first submit, naming the buffer
or field and why, when its compile needs any of these:

- a buffer the load does not have;
- a buffer of another format or page size;
- more pages than the load's buffer holds;
- a page count that the load's pages are not a multiple of (a block must not wrap);
- a field of another shape.

The message says to list the set or to name it in `km.ANY(...)`:

```text
Gate.gate_size16: buffer kWrite0 (index 16): it would hold 32 + 32 pages, the load's holds 16 + 16; the L1 layout is the load's: list the set at the make, or make it with km.ANY(...) naming the largest set
```

A handler with `stamps=` or a broadcast is resident only, so a set of one is refused.
So are: `km.ANY` twice for one handler; `km.ANY(p=n)` naming no constant parameter;
and the opt-in on a handler without constant parameters. A first use is reached from
`run.submit` only. `submit_many`, `Graph.command`, a stream's `submit` and a chain's
call take command words today and cannot name a set, so they run the template. Once
they take a handler's constants, an unbuilt set reaching them is refused with:

```text
Gate.gate with size=4: no handler was built for it, and a first use (km.ANY) is reached from run.submit only, not from this submit; submit it once with run.submit first, or list the set
```

Without `km.ANY` the refusal of a set not built is the one above, unchanged.

In the runtime's form (`form="runtime"`) a set is compiled at the buffer indices the
class was made to leave (`beside=`) and added with the program's compile arguments and
its args' place, as its own parts are built; it runs in a list and in a group that
switches program, on the cores that hold its program. A set added in a load of
`{program: group}` is refused on a group that switched to another program ("does not
hold").

The gate ([22_gate/as_class](../../examples/22_gate/as_class/README.md)) on 120 cores,
64 tiles a core, made with block 8 listed and `km.ANY(block=16)`. Blocks 4, 16 and 2
ran by first use, 0 wrong. The time of a command at each step, beside the same block
listed in a load of its own:

| Step | Block 4 | Block 16 | Block 2 |
|---|---|---|---|
| the class compile (a fresh process; found built: 0.3 ms) | 13.9 ms | 13.9 ms | 14.1 ms |
| the parts' build (fresh; found in the cache: 49 to 52 ms) | 7.25 s | 7.07 s | 7.31 s |
| the parts' write, the entry commands | 0.6, 3.4 ms | 0.6, 3.5 ms | 0.5, 3.3 ms |
| the first submit, warm cache (compile look, add) | 54.6 ms | 56.8 ms | 53.6 ms |
| the first command (loads the parts) | 174.86 µs | 169.48 µs | 171.66 µs |
| later commands, median | 169.70 µs | 158.62 µs | 169.74 µs |
| the block listed, later commands, median | 171.80 µs | 156.78 µs | 167.02 µs |

The later commands run as the listed block's, within 1.6 %. Block 16's row is from 20
commands of each kind in one process, and the same block resident in kernels with the
added handlers' path took 158.08 µs. Five commands of each, in another process, gave
166.36 against 155.41 µs, so five is too few to tell 16's difference from the card's
spread (150 to 169 µs over the 20). The first-use load's block 8 took 161.68 µs.

Sets with different blocks share the template's ring. A block never wraps the ring's
end, so a ring left at a position that is not a multiple of the next set's block makes
that command give up ([hazards](hazards.md#sets-of-a-handler-on-one-ring-with-blocks-that-do-not-divide-each-others)).

### A size given at each submit

```python
sets = [km.UPTO(block=8)]  # one build serves block 1, 2, 4 and 8
program = GateBlocks(a=a, b=b, out=out, handlers={"gate": sets})
run.submit(program.gate, (0, cores), args=(a, b, out, tiles, pattern, 4))
```

`km.UPTO` among a handler's sets makes one build serve every value up to a most. The
handler's text does not change. Each key names one of the handler's parameters, and
the parameter's kind says what the key means:

- **A constant int parameter** (`block: km.constant[int] = 8`): `km.UPTO(block=8)`.
  The value is a command word, given in `args` at the constant's place. The buffers
  are sized for the most, and the streams that read `block=block` take the word as
  their block. The host refuses, by name, a value below 1, above the most, or one
  that does not divide the most (a block never wraps a buffer's end):
  `GateBlocks.gate: block=3; a size given at each submit, at most 8: an int that
  divides 8 (the buffers hold whole blocks)`.
- **A tensor parameter** (`x: km.Tensor`): `km.UPTO(x=64)` means x's tile columns,
  up to 64. The width comes from the tensor given at the submit, and its rows' pitch
  is a word the handler's rectangles of x read. The make's example tensor is only
  the template. A wider tensor is refused at the submit, by name.
- **`rect`**, when the handler has no parameter of that name: `km.UPTO(rect=(4, 4))`
  means any rectangle of up to 4 columns and 4 rows. The rectangle is given at the
  submit as one more argument after the handler's parameters, `args=(x, out, (x0,
  y0, columns, rows))`, and the command goes to the rectangle's group (its first
  core and the span to its last). Each core computes its `self.x`, `self.y`,
  whether the rectangle holds it, and each broadcast's rank, receivers, sender and
  multicast corners from those four words and the grid's NoC coordinates
  (`km_rect.hpp`). These are the arguments `rect=` at the make gives today. The
  host refuses, by name, a rectangle larger than its most, one off the grid, a
  group that is not the rectangle's, and `rect=` at the make; the load refuses a grid
  past 16 x 16 cores or 7 panels ("a grid past 16 x 16 or 7 panels") and one whose
  cores' NoC coordinates are not their column's and their row's. On islands the
  rectangle counts from the island of the group's first rank and may run on to the
  next ([islands](islands.md#classes-on-islands)), `self.x` and `self.y` its places
  in the whole.

One `km.UPTO` per handler, naming as many keys as it gives:
`km.UPTO(block=8, x=64, rect=(4, 4))`. It stands beside the other sets in one list,
and what it names stays a word in each of them:

- With listed sets, `[km.UPTO(block=8), {"size": 4}, {"size": 2}]` is two builds
  (size 4 and size 2), each taking block at each submit.
- With `km.ANY`, `[km.UPTO(block=8), km.ANY]` compiles sets of the other constants at
  their first use, again with block given at each submit.

Refused by name at the make:

- a listed set or `km.ANY(p=n)` that names a key of `km.UPTO`
  (`block is given at each submit (km.UPTO); a set does not name it`);
- `km.UPTO` twice for one handler, or `km.UPTO` naming nothing;
- a key that is no parameter of the handler (other than `rect`), or one that names
  an int command word, a float or bool constant, or a pattern;
- a most that is not a positive int, or not (columns, rows) for `rect`;
- `rect` on a handler that reads no `self.x`, `self.y` or broadcast;
- a class where one handler takes the make's rectangle and another a rectangle
  given at each submit.

A handler's words are those of its class's last make, as its tile widths are. A
program made with other `km.UPTO` than the class's last make is refused at its next
submit (`made with km.UPTO {'block': 8}, and the class was made since with none ...:
make this program again`). A chain call's `km.at_submit` cannot give the value
(refused by name). Not in this version: two rectangles in one handler, and `km.UPTO`
in a merged program.

What it costs, on one p150a card at 1,350 MHz
(`experiments/language/sized_block/sized_block.py --form all`), every word exact
against numpy:

| Size | Given at each submit | Built in |
|---|---|---|
| block over 20 tiles a core, 8 cores: block 8, 4, 2, 1 | 8.30, 7.92, 9.08, 13.26 µs | block 8: 8.32 µs |
| kernel text, block | 34,880 B | 34,768 B |
| width: tensors of 16 and then 8 tiles, one load | 0 wrong each | |
| a row broadcast at 4 x 4, five rectangles on one load (one across the panels, one a single column, one a single row) | 4.54 µs | 4.40 µs (`rect=` at the make) |
| kernel text, rectangle | 37,744 B (BRISC +944, NCRISC +80, TRISCs +68, +68, +88) | 36,496 B |

A class without `km.UPTO` is built as before, to the byte.

### Another class added to a live run

A handler of **another** class can be added to a live run whose groups switch program
(a run loaded with a dict): `run.addon(Other(..., form="runtime"))` makes the class one
more program of the run and returns its program field; `run.load(other, group)` gives a
group its buffers and its parts, and `run.submit(Other.handler, group, args=...)` runs
it ([host](host.md#a-class-added-to-a-live-run)). The five things it needs are met so:

1. **Its buffers and fields** lie in the switch area above the runtime's buffers, which
   the load keeps on every core: the largest program's layout, grown by default into
   what the load's L1 table leaves free less 64 KiB (`session.load(..., buffers=N)`
   states or caps it). Its record (its buffers' four words and its parts' table
   entries) is written to DRAM; the load command sets its buffers up as for a program
   given at the start.
2. **Its buffer indices** are any the loaded kernels' buffers do not use with another
   format: make it with `beside=` the loaded class to leave that class's indices.
3. **Its tensors** are command words, as every runtime-form class's are.
4. **Its formats.** The kernel build bakes a table of each buffer index's formats
   (unpack source and destination, pack source and destination, the tile sizes) into
   every part as constants. The added class's parts are compiled against a copy of the
   loaded kernels' table with its own buffers' entries, recomputed by tt-metal's
   functions over the load's formats and its own; the copy must equal the kernels' at
   every other index (an added format that would change the shared Dst format is
   refused). Its fp32 fields' `UnpackToDestFp32`, per buffer index, comes with it. The
   loaded kernels are not rebuilt.
5. **Its compute options** that are one a kernel (fidelity, `fp32_dest_acc_en`,
   `dst_full_sync_en`, `math_approx_mode`, `bfp8_pack_precise`) equal the load's;
   refused by name with both values.
6. **Its per-core args** (a rectangle, a broadcast, `self.x`, `self.y`). A run's
   per-core args are set at the launch, so an added class's go in a DRAM table, one
   row per core (the reader's words, then the writer's, then compute's, padded to
   64 B), and its record holds the table's address, a row's bytes and where the row
   goes in L1: right after the class's buffers in the switch area. The load command
   copies the core's row there, and the class's parts (only they: built with
   `KM_ARGS_L1`) read their args from it; the loaded kernels and the other classes
   still read theirs from the launch. Why no command of the class runs on a core
   before its row is in L1: the load command runs on the reader with the four other
   CPUs (the writer and the three compute CPUs) parked in it until the reader's go;
   the reader reads the row and waits for the read to land before it sets its
   buffers up and gives the go; each parked CPU clears its L1 cache when it sees the
   go; and a core takes its commands in order, so its next command, on any of the five
   CPUs, starts after the load command. A thread's `kmx_load` is the same load
   command and returns when it is done. Where that order is not given, by name: the
   host refuses a command to a group whose cores do not hold the class
   (`does not hold`), and `run.load` refuses a group that does not contain the
   class's rectangle; the dispatcher refuses a command of the class, from any source (a
   thread, a stream, a graph), to a group with a core that does not hold it, by its
   table of the field each core holds (`CommandRejected`, reason 13, `program`), so no
   CPU of such a core takes it and the run goes on. The row costs the
   load command little: 6.80 and 4.78 µs (first load, second) with a 64 B row and 7.86
   and 4.60 µs with a 128 B row, against 8.02 and 4.34 µs for a class without args and
   7.84 and 4.38 µs for the loaded gate. The row read is built into the load's kernels
   only when a class with per-core args is named with `Program.ahead(other)` before
   the load (412 B of the reader's text, about 40 B of each other CPU's); a load
   without one pays nothing, and an add of a class with per-core args to it is refused
   by name. A word field (`km.L1[km.u32, n]`) is one more per-core arg: its tensor's
   address, compute's word, goes in the row, so a class with word fields is added as
   one with args (named ahead); its tensor, one row in each core's L1, may be made
   before the load or while the run is live (it takes the L1 the area leaves, 64 KiB
   by default).
7. **Its own overlays** (`@km.handler(overlay=True)`): numbered by the add, each
   declared number plus the lowest shift that leaves every number free (the run's
   overlays, added code, earlier added classes' own) and below every program field's
   overlay; a later field whose number they took is refused by name. Their parts are
   built as the class's (its formats, its args' defines) and go into the room the load
   reserved; each group's cores write their table entries (each part's address and
   bytes) by the entry command of the service overlay. A class named ahead has its own
   overlays' parts built at the load, the slots sized by them (a slot more in a kernel
   where its part loads its own overlay while it runs) and the entry command's code
   built into the service overlay (`KM_ADDED_ENTRY`, 448 B of its reader part); a
   load naming no such class pays nothing, and a class with its own overlays added
   to it is refused by name (unless the program names slot bytes, which builds the
   entry command too). Without ahead, a kernel with one slot is refused by name.
8. **On an island run** (`Session(cards=[0, 1])`, loaded as a dict): `run.addon(other)`
   on the run the session's load returned adds it on every island, all or none:
   every island's checks and parts first (built against that island's kernels, with
   the defines its load gave its programs), then on each island its parts in its
   room, its record and its per-core rows in its own DRAM (a row a core of its card,
   as its load would give them at the start) and its own overlays' entries; one
   program field and one overlay numbering on every island. `run.load(other, group)`
   takes flat ranks as before; the link cores take no part. Its tensors are ones made
   before the load (the islands' tensor table is written at the load): a class made
   with one made after is refused by name. `run.addon` on one island's run is refused
   by name, and so is a card split into islands (one launch, the kernel form). A
   part takes whole rows of the room, a row the load's widest part: 41,280 B on an
   island (the service overlay with the link code; 9,088 B on one card), so a load
   naming a class ahead reserves at least a row for each of its parts and its own
   overlays' (a class with two own overlays: 15 rows, 619,200 B of DRAM a bank copy);
   the room is written while the run lives in pieces within the run's transfer chunk
   (host.md). Parts packed several to a row are not built.

Refused besides, by name: a class in the kernel form, a run not loaded as a dict, other
`opt` levels or NoCs, a dynamic handler, buffers
past the switch area or a part past its slot, a traced run and a merged
program. The slots cannot grow under a live run: the load sizes them by the parts of the
classes named with `Program.ahead(other)`, which the load also builds, or by
`Program.slots(bytes=)`.

What an added class cannot be, refused by name at the add:

| An added class that is | The message names | What adding it would take |
|---|---|---|
| a class with a dynamic handler | `a dynamic handler (its queues and tables are set at the launch)` | the dynamic run's queues in DRAM and its tables on every core reserved at the load for a class named ahead, or the dynamic service loaded again at the add |
| added to a traced run | `a traced run (the trace buffer is placed at launch)` | trace room and numbers for its handlers given at the add; today the trace's layout is fixed at the launch |
| a merged program | `a merged program (merge its sources with the load's)` | merged into the loaded merged program's dispatch table at the add: its merged parts built again against the live table |

On the card (cores 0 to 59 the gate, 60 to 119 the added class, a row of 4 tiles times
4 x 4 tiles with an fp32 field and a bfloat8_b weight stream that the gate does not
have): its words exact; the add took 7.84 s (its parts linked and checked on its own
tables) and 72 ms with the class named ahead; its command 9.42 and 9.40 µs against 9.26
and 9.42 µs loaded at the start; the gate 22.84 µs before the add, 22.40 after and
22.58 after its group switched back, 0 wrong, the same largest error each time. With
per-core args, on a rectangle of cores 60 to 119: a class reading x page `self.x`, and
a class whose rows' first cores broadcast a page along the row, each exact added live
(3.80 and 4.88 µs a command) and loaded at the start (3.78 and 4.94 µs); the broadcast
class loaded and called four times by a thread's `kmx_load` and commands (the load
10.2 µs, a call 6.0 to 6.8 µs), exact.

## The lines of a handler

| Line | Movers | Compute |
|---|---|---|
| a parameter `name: km.Tensor`, `name: int`, `name: km.Pattern` | a command word | a command word |
| `n = a + b * 2`, `n += 1`, an `if` on ints | the int, where a kept line uses it | the same |
| `part = self.partition(pages)` | `km::partition(count, rank, size)` | the same |
| `part = b.partition(pages)` | `km::partition` by the core's column of the rectangle (a row's broadcast) or its row (a column's) | the same |
| `total = km.tile.zeros(km.fp32)` | | DST held, tile 0 filled with zeros |
| `for tile in km.Stream(pages, block=8, pattern="AB", repeat=n):` | one read stream in a buffer pair, moved with the write streams the loop pushes to (`movement.move`) | each block from the half its mover filled, each tile of it |
| `q = km.Stream(pages, block=b)` (read, not iterated) | its one block moved at its line | the block waited for at the line, taken whole by bodies in the loops inside its block of lines, freed when that block's pass ends (A read stream bodies take) |
| `km.Stream(pages).drain()` | the read stream, its tiles dropped | each block waited for and freed |
| `km.Stream(t.items(first, n, size=B))`, read or `write=True` | the same stream of items: item i is B bytes at page i / P of t, byte (i % P) x B, P = t's page bytes / B (`km::items`, an address generator over t's accessor); the pair's pages are B bytes; t on one island | each item a page of the pair (a body takes it: Inline bodies); an item is not a tile of compute: a push of it, into it, an add of it or a tile function or product of it is refused by name (write the tensor's tiles, `t.tiles[r : r + 1, c0:c1]`, or move the items in a mover's body) |
| `first, count = t.pages(i, 1).read()[0:2]` | mover A: page i read into a page it pushes to compute; mover B: page i read into its word page; both take words 0 and 1 | unpack waits for mover A's page, takes the words, hands them to math and pack by mailbox, frees it |
| `m0, m1 = t.pages(i, 1).read(km.fp32)[0:2]`, t float32 | the same, each word a C++ `float` of its bits | the same, handed as bits: float words, a command's float words' equal (A float word in tile arithmetic) |
| `for e in t.pages(i, 1).entries(Job, first=8, size=8, count=0):` | page i read once on each mover (mover A's pushed to compute, mover B's its own), then the loop over its entries, `e.word` an int of the loop (Tables of entries) | the page waited for as pushed, the same loop, the page freed after it |
| `jobs = t.pages(i, 1).entries(...)`, then `for e in jobs[a:b]:` and `jobs.count` | the same read at the line, at the handler's top; the slice's ends clamped to the count | the same; the page freed at the handler's end |
| `total += tile` | | the tile copied and added into the held tile |
| `out = km.Stream(pages, write=True, block=8)` and `out.push(tile)` | one write stream, moved with the read that feeds it | the tile copied into the same half of the write pair |
| `out.push(t)`, `t` a body's tile in the loop | | DST 0 packed into the write pair at the loop tile's place |
| `pages.write(total)` (one page) | mover B: `wait_front`, `km::write_pages` of one page, `noc_async_writes_flushed`, `cb_pop_front` | the held tile packed into a page buffer, DST released |
| a second `pages.write(total)` (or of a reduce's sum) in the same statement list, to a tensor of the first's format | mover B: `km::write_pages` of the same pages once more; the flush and the `cb_pop_front` move from the first write to the last | nothing more: packed once |
| `if word:` / `else:` each writing `total` (or the leader's `whole`), the condition a command word or rank | each branch its own write, as above, on compute and mover B alike | each branch's pack |
| `if word:` writing `total`, no `else` (and `total` not used after the if) | on the other path mover B frees the sum's pages (`// not written on this branch: freed`) | DST released on the other path |
| `whole = km.reduce(total, form=form)` | `km::reduce_arrive` on mover A, `km::reduce_emit` on mover B; in a range loop a reduce a turn, and so is a handler's second reduce: a sender waiting from turn 2 for its receiver's credit ([ports](ports.md#reduce)). A handler's reduces share one form and group size (another is refused, [not in this version](#not-in-this-version)); a chain describes such a handler by its first. A second pass that needs the first's sum at every rank (a layer norm's (x - mean)^2) is written today as a chain of three calls: the first reduces and writes the sum to a tensor m, the second streams x beside m's page and pushes `f(x, m)` to a tensor d, the third reduces d ([two_reduces.py](../../experiments/language/two_reduces/two_reduces.py)'s `two_pass`) | `km::reduce`, the arrivals added in rank order |
| `whole = km.reduce(acc, form="tree", group=S)`, `acc` n tiles | the same in `km::digit_group(command, me, 1, S)`: S consecutive ranks a group, n tiles a slot | each of the n tiles added through DST tile n (n + 1 tiles of DST) |
| `whole = km.reduce(self.part, form="tree", group=S)`, `self.part` a float32 field of sums | `km::reduce_arrive` on mover A into slots of the field's tiles; `km::reduce_emit` of the field's pages on mover B (a sender) | each arrival copied into DST and packed onto the field with the packer's L1 accumulation; the field handed on (Products) |
| `whole = km.all_reduce(total, form=form)` | `km::all_reduce_arrive` on mover A, `km::all_reduce_emit` on mover B: rank 0 casts the sum into every other rank's slot 0 ([ports](ports.md#all-reduce-the-sum-at-every-rank)) | as `km.reduce`'s, then `km::all_reduce_take`: a rank but 0 packs the sum cast to it; `whole` is every rank's, written without the leader's if to a float32 tensor (written in another format it is packed again by compute, refused by name at a rank but the leader: that buffer's front there is the part it sent) |
| `km.all_reduce(self.part)`, `self.part` a float32 field compute wrote | the same, of an out buffer of the field's tiles | the field into DST, the arrivals added, the sum packed for mover B and (rank 0) into the field; every other rank packs the sum cast to it into the field, which a later loop's tile function takes |
| `km.reduce(self.state, form="tree", group=S, combine=MERGE)`, `self.state` an own float32 field written in place | the same, of the state's pages | the MERGE body on each arrival, which writes the state in place; a sender's state pushed for mover B; the root's stays in place (A state merged, not added) |
| `t.tiles[r0:r1, c0:c1]`, where a page range is taken | the first page `r0 x T + c0` (T the tensor's tile columns); one row, or all T columns, is a page range; more rows a `km::Walk` of whole rows, `T` apart; r and c any ints of the handler (words at run time) | |
| `t.tiles[r0:r1:s, c0:c1]` | its rows s rows of t apart, `(r1 - r0 + s - 1) / s` of them: a `km::Walk` of whole rows `s x T` apart (s an int or a word) | |
| `n = min(a, b)`, `max(a, b)`, a and b ints | `(a < b ? a : b)`, `(a > b ? a : b)` | the same |
| `if self.rank == self.leader:` and `pages.write(whole)` | the leader's write of the sum | |
| `return word` under `if self.rank == self.leader:` | `movement.result(word)` on mover B | |
| `word = pages.read()` (one page) | mover B: `km::read_pages` of the page into its word page, the barrier, the L1 cache invalidated, the page's first word | |
| `word += round + page`, `word` a mover's | the line on the word's mover | |
| `pages.write(word)` (one page), `word` a mover's | that mover: its word page (the word first, zeros after), `km::write_pages`, `noc_async_write_barrier` | |
| `b = km.Broadcast(self, along="row", window=2)` | the broadcast on its mover (rows on mover A, columns on mover B); a core it leaves out skips the handler, on the other mover too | the same core skips it |
| `b = km.Broadcast(self, along="column", length=L, mover="A")` | bands of L cores of each column (or row), each band's first core its sender, on the mover given (A: NoC 0, the band's top or left; B: NoC 1, its bottom or right) | |
| `km.Broadcast(self, ..., unread=word)` | when the word is not 0, each core's mover takes each block's slot and pushes it as it is (no read, cast or arrival), padding to the window at the end | the blocks taken as cast ones |
| `if b.sender():` `v = b.cast(t.pages(first, n).read())` `else:` `v = b.arrive()` | the sender's reserve, read and cast, the others' arrival, block by block of the loop around it; the last block's finish | |
| the same inside more than one loop, or a loop not from 0 | the blocks counted over every loop around the cast (`km_blocks_b`, on the mover and compute), the finish after the outermost loop by the count | the padding by the count |
| the same outside every loop | one block (block 0), its finish at the cast (at the handler's end when a stream, a page write or a fill on its mover follows a product's block) | the padding (the window's other pages) after the block's last use, at the end of compute's handler |
| `v = b.cast(k.tiles[k0:k1, 0:D].T, w.tiles[k0:k1, 0:E])` | several parts into one block, in order, each at its most's place (Several ranges into one block) | |
| `for tile in v:` | | the block received, each page of it, then freed; the padding after the last |
| `km.cxx("...", takes=[v])`, v a broadcast's block | | the block received before the body, `v_buffer` its buffer and `v` its first page there (0), freed after the body: the body indexes the block's pages itself |
| `for i in range(n):` | a loop | a loop |
| `km.cxx("...", takes=[n], on="mover_b")` | the C++ as written, on the CPU `on` names (Inline bodies) | compute's three CPUs when `on` is not given; `on="unpack"`, `"math"`, `"pack"`: that CPU's alone |
| `class P(km.Program, compute={...})` | | the compute kernel's options, by the host API's names, over the ones the class's lines set |
| `class P(km.Program, defines={...})`, and `P(..., defines={...})` | the kernels' defines (the make's over the class's) | the same |
| `km.cxx(..., at="start")`, `at="end"` | the body at the kernel's start (after `km::Movement`) or after its run | after the start-up, before the commands; or after the run |
| `class P(km.Program, startup=km.Startup("a", "b", "out", order="reverse"))` | | `compute_kernel_hw_startup<SrcOrder::Reverse>` of those fields' buffers, in place of the default |
| `w = km.cxx(..., gives=int)` | `uint32_t w = 0;` then the body, which sets it | the same |
| `t = km.cxx(..., gives=km.Tile)` | | `tile_regs_acquire()` then the body, which leaves the tile in DST 0 |
| `km.cxx(..., takes=[report], on="mover_b")` after pack's body over `report` | mover B: the page waited for, the body, then the movers' write of it | (A page from several CPUs' words) |
| `self.f.fill(km.Stream(t.pages(first, n)))` | mover A: `reserve` of the field, `km::read_pages` of its n tiles, the barrier, `cb_push_back` | |
| `self.f.fill(km.Stream(t.tiles[r0:r1, c0:c1]))` | the same, a read a row; refused on the card when the rows are more than the field holds | |
| `self.f.zero()` | compute: a zero tile in DST 0 packed into each of the field's pages, `cb_push_back` of them (no DRAM read); then read as a filled field | a field that only needs zeros (a step's states, rows of running sums) |
| `p: km.L1[km.bf16, 3, 32, 32] = km.own()` (a field) | | a buffer of its format and pages that only bodies use: no line fills, waits for or frees it (A body's own buffer) |
| `for xb, wb in zip(km.Stream(x...), km.Stream(w...)):` `acc += self.g(xb, wb)` | the two read streams in one move, a tile of x's a row of w's | block k of each as it arrives: each tile t of xb by row t of wb into acc's DST tiles, both blocks then freed (Products) |
| `for xb in km.Stream(x...):` `acc += self.g(xb, ws)`, `ws` a broadcast's arrived block | | the block received before the loop: each tile t of block k of xb by K row k x b + t of the block's part (a window `ws[:, c0:c0 + n]`, a part `kv[i]`) into acc's DST tiles; the block freed after the loop (Products) |
| `for wb in km.Stream(w...):` `acc += self.g(xs, wb)`, `xs` a broadcast's arrived row of K tiles, `wb` K rows of acc's width | the stream on its pattern (both movers on AB); the cast's finish at the handler's end when its mover moves the stream | the row received before the loop: tile k of xs by row k of the stream into acc's DST tiles; the row freed after the loop (Products). Its cost today, one card, a decode gate (X 168 tiles cast down each column, 8 columns a core on 84 cores, W in pages of 2 on AB): 321.5 us (bfp8_b) and 197.5 us (bfp4_b) against the stream form's 313.3 and 187.3 (X and W streamed in step, groups of 2 cores and a tree), the same words; the cast alone takes 49.7 us before the first product |
| `acc += f(xs[r], w[i][:, c0:c0 + n])`, xs a held stream, w a broadcast's block | as before | k products of row r's tiles by the block's rows, between the block's wait at its cast and its free at the end of its lines (A held stream's row by a broadcast's block) |
| `acc += f(self.h, w[i][r0:r0 + n, :])`, w a broadcast's block | the cast as before | the block waited for at the cast; the field waited for, n products by rows of the block; the block freed at the end of the cast's lines (A field by a broadcast's block) |
| `s = g(self.m[i], self.s[r])`, then `pages.write(f(s, self.s[r]))` | as before | f run on state r where s's start put it, no second take of its token; packed once at s's write (Sums started from a field) |
| `acc = f(self.o[i], self.s[r])`; `self.o[j].write(acc)`, self.o a filled field of rows | the fill | row i copied into DST in place after the field's token; acc packed into row j by index, the token pushed, then the state (Sums started from a field) |
| `acc = f(self.o, self.s[r])`, f a step function, self.o an fp32 field compute writes | none | the field's tiles into DST 0 .., state r after them, f's body; acc's write packs the row, then the state (Sums started from a field) |
| `self.h.write(acc)` (or `f(acc)`, a step's row); `out += f(self.h, wb)` | none | the last write's pages freed, acc packed into the field's; before the loop the field waited for, a row of products a stream row (A product whose left side is a field) |
| `pages.write(g(acc, self.s[r]))`, g a step function, `self.s` an fp32 field of states | as `pages.write(acc)`; `t.pages(first, n).write(self.s)`: mover B a page at a time | the token taken, state r copied into DST after acc, g's body, the row packed, the state packed into page r and the token pushed (A step function on DST sums) |
| `self.f += self.g(xs, ws)`, f an fp32 field, `xs`, `ws` two broadcasts' blocks or two streams' | | the blocks' product packed into the field with L1 accumulation (Products) |
| `y.tiles[r0:r1, c0:c1].write(self.f)`, f that field | mover B: a row at a time, `wait_front` of its tiles and `km::write_pages` | the field's tiles copied into DST, packed into Y's format, the sums' count reset |
| `y.tiles[r0:r1, c0:c1].write(gate(self.h, self.g))`, gate a tile function, h and g fields of sums of one shape | as the field's write | tile by tile each field's tile copied into the DST tile of its operand, gate's body on the math CPU, DST 0 packed into Y's format; both fields freed, both counts reset |
| `y.pages(first, n).write(act(acc))`, act a tile function of one tile, acc DST sums | as acc's write | act's setup, then act's body on each of acc's DST tiles in place on the math CPU; the tiles packed as acc's write packs them |
| `self.f[i] = x`, `self.f[i] += x`, f a field of words | | the unpack CPU's write of word i of this core's row of f (`UNPACK((field_f[i] += x))`) |

An items stream reads a tensor its island holds; a kernel reads an item of a tensor on
another island with `km::read_bytes`, at the far rate ([Islands](islands.md)' table).

`self.rank`, `self.size` and `self.leader` are the core's rank in the command's group,
the group's size and its rank 0; `self.x` and `self.y` its column and row in the
rectangle; `self.core` its place in the grid, whatever the group (`me()` of the
runtime: on islands the flat rank, its island's first rank plus its index there).

A broadcast's rectangle is fixed at load (a program argument), and `Run.submit`
refuses a broadcast's command whose group is not the rectangle's, naming the
`km.Broadcast` line; a rectangle per command is not in this version. The group is
the rectangle's core indices from its first core to its last, so on a rectangle
narrower than the grid `self.rank` is not the place in the rectangle: a core is placed
by `self.x` and `self.y`, and the cores between are in the group but skip the handler.
On `rect=(x0, y0, W, H)` of a grid G columns wide, the core at column x and row y of the
rectangle has `self.rank` = y * G + x, which runs from 0 to (H - 1) * G + W - 1 with
gaps. `self.x` and `self.y` are already counted from the rectangle's origin, so the
dense index 0 .. W H - 1 is `self.y * W + self.x`, with W the rectangle's width as the
class was given it (a constant or a command word). Rows or pages picked by `self.rank`
on such a rectangle run past the tensor
([hazards](hazards.md#selfrank-on-a-rectangle-narrower-than-the-grid)).

### A broadcast's mover, bands and count

`km.Broadcast(self, along="column", length=L, mover="A")` cuts each column of the
rectangle into bands of L cores (the last band the rest), each a broadcast of its own
whose sender is the band's first core in its NoC's direction: mover A runs on NoC 0
(a column's top, a row's left), mover B on NoC 1 (bottom, right). Without them a row's
broadcast is mover A's and a column's mover B's, the whole side of the rectangle a
broadcast. L is an int the build knows (a constant or a plan value). A handler runs
one broadcast a mover; two handlers' broadcasts on one mover each have their own
arguments (a buffer and a number), the second's after the first's in the mover's
(`kArgBroadcast + km::kBroadcastArgs * i`). A core has `KM_BROADCASTS` (2) broadcast
areas, one a broadcast of the class: a third is refused at the make, naming the
class's broadcasts and their lines. Make it a class of its own.

A cast inside one loop from 0 numbers its blocks by the loop's name and finishes after
that loop, as before. Inside more than one loop (a job loop around a chunk loop whose
count is the job's), or in a loop not from 0, the blocks are counted over every loop
around it: a word `km_blocks_b` on the broadcast's mover and on compute, from 0 at the
handler's top, the block's number at each cast or arrival and one more after it; the
finish and compute's padding (`km::broadcast_end`) come after the outermost loop, by
the count. A cast outside every loop is one block: the mover's finish stands at the
cast (it waits for compute to pop the block) and compute's padding pop stands at the
end of the handler, after the block's last use; a second cast of the same broadcast
outside a loop, or in a loop after one outside, is refused. Casts in two loops at the
handler's top are refused.

`unread=word` (a word of the handler) runs the same command with no broadcast: when
the word is not 0, each core's mover takes each block's slot and pushes it as it is,
and after the last block pads the slots to the window as the finish does; compute
takes the blocks as it takes cast ones. A block compute takes and drops unused is
`for tile in v: pass` (waited for, freed). In the
[table of jobs](../../experiments/language/class_lines/README.md), BandCasts casts 5
blocks over two loops down bands of 2 cores on mover A, 0 of 416 filled pages
differing on the card (13.98 us), and runs the command again unread (9.84 us).

A body takes the handler's names and runs on one CPU or on compute's three: the section
Inline bodies says what it takes where. From
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
read pair holds 4 blocks a half and a write pair 2; a page buffer 2 pages. Handlers whose
read streams differ only in block share one read pair, its halves the largest block's
(a make without `km.ANY`; one handler's two blocks, or a block given at each submit,
still take pairs of their own): an item that does not fit before a half's limit is
pushed after padding to it, which the consumer pops (`KM_RING_PADDED`, built only into
such programs), and the build report names the pair `shared by <handlers>`. A reduce's
arrival buffer 118 float32 tiles by owner (a slot for each other core of a 119-core
group), by tree a slot of n tiles for each round (7, or the bits of S - 1 in groups of
S); a broadcast's buffer `window` blocks of the pages it casts. Buffer indices are
0 up for reads, 8 up for arrivals, broadcasts and fields, 16 up for writes; 26 to 31
are the runtime's. Compute's tiles are in DST as float32; a float32 buffer is unpacked
to it unrounded.

## Inline bodies

A `km.cxx` body is its C++ as written, at its line's place in its handler, on the CPU
`on=` names: `"mover_a"` or `"mover_b"` (the movement source, under that mover's
`#if`), `"unpack"`, `"math"` or `"pack"` (the compute source, under that CPU's
`#if defined(TRISC_...)`), or compute's three CPUs when `on=` is not given. A body
that binds no name is not braced, so the names it declares are seen by the bodies after
it on its CPU (a walk's state on pack, set up before a loop and read in it). Its first
lines that are `#include` lines go to the top of its kernel for every CPU of it;
`include=["path"]` puts a header under the body's CPU's `#if` alone (a header only
the pack CPU compiles). A `#line` before the body points the compiler's messages at
the Python line, in the handler's own file (in a `km.merge` of classes from several
files too). The text is a string, a string constant of the class's module, or a
sum of them (`km.cxx(HEADERS + GIVE_UP + "...")`: one body in several handlers); for a
constant the `#line` is the call's. Any other text is refused at its line.

What a body takes, by where it runs:

| `takes=` of | compute's three | one compute CPU | a mover |
|---|---|---|---|
| an int, a partition, a pattern, a page's words (`.read()[0:2]`) | by its name | by its name | by its name |
| a constant of the class (`self.K`, [Constants](#constants)) | `K`, a `constexpr uint32_t` of this program's value | the same | the same |
| an entry `e` of a table (`for e in ...entries(...)`), or a table | `e_word`, each of its words (a table: `name_count`) | the same | the same |
| a body's word (`w = km.cxx(..., gives=int)`) | a word of compute's three | a word of that CPU, or of compute's three | a word of that mover |
| a stream's tile or item (the loop's name) | `name_buffer` and `name_index`, its buffer and place | `name`, its L1 address: unpack's own; math and pack are handed it by mailbox before the body and hand a release back after it, and unpack frees the block only then | after the loop: `name_tiles` and `name_cycles`, what this mover moved in the loop's move (`km::Moved`: its tiles, the cycles of its passes that moved a block) |
| a write stream | `name` and `name_buffer` as on pack (a body that unpacks and packs itself: the reserve and push are the line's; `name` is 0 only on pack, when it gave up) | on pack: `name`, the L1 address of the page it fills (0 when compute gave up), and `name_buffer`, the half's buffer for `pack_tile`; in a stream loop the loop's page, reserved with the block and pushed after it (block= the loop's); outside one, one block reserved, pushed after the body and moved by the movers at that line; made outside the loops around the body, one block each time the body runs (A write stream bodies fill) | mover B, outside a loop, after pack's body: `name`, the L1 address of the page pack pushed, waited for; the movers write it after this body, not at pack's line (A page from several CPUs' words) |
| a read stream made outside a for line (`q = km.Stream(...)`) | `name_buffer` its block's buffer, `name` its first page there (0); no wait or free of the body's own | | |
| a field filled above in the handler | its buffer, freed after the body | its L1 address (unpack's own, handed to math or pack), freed after the body | mover B: its L1 address, freed after the body |
| a field another handler fills (kept) | its buffer, waited for, never freed (the filler frees its last fill before the next) | its L1 address, waited for once, handed to math or pack | mover B: its L1 address |
| an own field (`km.own()`) | `name`, its buffer, no wait, no free | the same | the same |
| a tensor parameter | | | `name`, a `km::Tensor` of its command word (its accessor, `.word`) for `km::read_pages`, `km::write_pages` and the dataflow calls |
| a tile written to a page (`pages.write(t)` below) | | | mover B: its buffer, waited for once |
| a reduce's sum of a field (`whole = km.reduce(self.sums, ...)`), under `if self.rank == self.leader:` | `whole_buffer` and `whole_index` (0): the group's sum, fp32 tiles of the field, waited for before the body and freed after it, once | | |

The build's report names each value handed between compute CPUs, and each kept field:

```text
  field tree: filled by load_tree, read by trace_rays (it stays in L1 between commands)
  hand-over: trace_rays, line 36: first, count from unpack to math and pack by mailbox
  hand-over: trace_rays, line 38: tree's L1 address from unpack to pack by mailbox
  hand-over: trace_rays, line 58: rays's L1 address from unpack to pack by mailbox
  hand-over: trace_rays, line 58: the release of rays from pack to unpack by mailbox
```

What each line costs, beside the hand-written form of the same move (measured on the
ray tracer's trace pass, [raytrace_class](../../experiments/language/raytrace_class/README.md)):

| Line | Lowering's cost |
|---|---|
| `t.items(first, n, size=B)` | none over a page stream: one NoC read or write of B bytes an item, its address a divide and remainder by P (the hand's own arithmetic); the pair's pages B bytes |
| `.read()[0:2]` | one read of the page on each mover, mover A's pushed to compute (one page of t's page bytes on each, 64 B for the ray tracer's directory); on compute one timed wait and two mailbox words a word |
| a stream's item on one compute CPU | two mailbox words an item (address, release) besides the loop's own lines (the wait's answer to math and pack, the reserve and push of a write stream's page): on the ray tracer's slowest core 97 cycles a packet more than the hand's loop, 20.7 us over its 288 packets |
| a kept field | one timed wait a command (and one mailbox word to one CPU); no copy: the ray tracer's tree is read once by its load handler (23.7 us) instead of 21.2 us of every command |
| `compute={...}`, `on=`, `include=` | none: options of the build, an `#if` |
| a tensor | a `km::Tensor` made from its word (its accessor) |

### A body's own buffer

A field whose value is `km.own()` is a buffer that only bodies use, as a hand kernel's
scratch space: a FIFO between compute's CPUs, a tile written in place, a constant
built once. From [own_buffer](../../experiments/language/own_buffer/README.md):

```python
total: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)  # written in place
step: km.L1[km.bf16, 1, 32, 32] = km.own()  # pack's token: total is written
delay: km.L1[km.bf16, 2, 32, 32] = km.own()  # pack to unpack, one step late
```

It is a buffer of its format and pages (8 up, with the other fields), in the build's
L1 table, and the report says "own: used by bodies, not filled by a line". A body
takes it by name (`takes=[self.total]` binds `total`, its buffer index); the bodies do
their own reserve, push, wait and free. A float32 one is unpacked to DST as float32. A
buffer has one producer: the build finds `cb_push_back(name, ...)` in the bodies that
take the field and refuses a push on a second CPU. `km.own(push="pack")` names the one
CPU that pushes, and `km.own(push=None)` says the field is written in place and never
pushed. A line that fills an own field, and a field of words declared own, are
refused. The pack and unpack CPUs share no variable, so state written in place needs a
token from the writer to the reader (`step` above), as the attention pass's has. A
body that copies from buffers of two formats sets the unpacker's format before each
copy (`reconfig_data_format_srca(total)`): `copy_init` keeps the one it finds.

### A body on a reduce's sum

The hand's K split runs an operation on the group's sum at its root, where it would
otherwise pack it for the write (the decode layer's residual add, its scale by 1 /
rms). A body under the leader's `if` takes the reduce's result as that operation:

```python
whole = km.reduce(self.sums, form="tree", group=self.SPLIT)
if self.rank == self.leader:
    residual = km.Stream(r.tiles[0:1, first : first + self.WIDTH], block=self.WIDTH)
    out = km.Stream(
        y.tiles[0:1, first : first + self.WIDTH], write=True, block=self.WIDTH
    )
    km.cxx(ADD, takes=[whole, residual, out])
```

`whole_buffer` is the field's buffer and `whole_index` its first tile (0): the group's
sum in fp32, `kField<Name>Pages` tiles. Compute waits for it before the body and frees
it after; the body unpacks it, adds what it takes and packs into `out`
([test_class_body_takes_sum.py](../../tests/core/dsl/compile/tiles/bodies/test_class_body_takes_sum.py)). A body
that takes the sum outside the leader's `if`, on one compute CPU or a mover, or after
the sum was written or taken, is refused at its line.

### A state merged, not added

Attention's partial results (each row's maximum, its sum, its output rows) go up a
group's tree merged, not summed: at each round the arrival and the core's own state make
a new state. `km.reduce(self.state, form="tree", group=S, combine=MERGE)` does that for
an own float32 field written in place (`km.own(push=None)`):

```python
state: km.L1[km.fp32, 3, 32, 32] = km.own(push=None)
token: km.L1[km.bf16, 1, 32, 32] = km.own(push="pack")  # MERGE's, pack to unpack

...
km.cxx(PARTIAL, takes=[self.state])  # this core's state
km.reduce(self.state, form="tree", group=4, combine=MERGE)
if self.rank == self.leader:
    km.cxx(FINISH, takes=[self.state])  # the group's merged state
```

Mover A takes each round's arrival into slots of the state's tiles. MERGE, a body of
compute's three, runs once for each arrival slot with `arrival_buffer`,
`arrival_index` (the slot's first tile) and the field by its name bound; it writes the
merged state in place. On a sender compute then pushes the state and mover B sends it to
its receiver; at the root the merged state stays in place for the bodies after. The
line gives no value. A merge of a field that is not own and written in place, of a
field other than float32 (the arrival's slots are float32), one assigned to a name, one
by owner without `group=`, and `combine=` on a tile in DST are refused at the line
([test_class_merge_reduce.py](../../tests/core/dsl/compile/tiles/reduce/test_class_merge_reduce.py)).

Pack writes the state and unpack reads it, and they share no variable: the next
arrival's run, or the root's body after the merge, could read a state pack has not
written yet. So MERGE hands a token from its pack to its unpack: an own field pack
pushes (`kFieldToken` above), pushed after the body's packs and waited for and freed on
unpack before it reads the state. A MERGE body that names no own field compute pushes
is refused at the merge's line, naming the first later line that takes the state.

A merged group may span islands as a summed one does: on two cards (2026-11-17) a
float32 tile a rank merged by an elementwise maximum (`binary_max_tile` in the combine
body) over one group of all 236 ranks gave the exact maximum (integers of -127 to 127),
28 to 30 us a command; each sender takes its state's token after the reduce, as the
root's body does, or the next command's first pack of it finds no room.

### A body's reads and writes stated

A body that takes a tensor reads or writes its pages with its own C++, which the
build does not read. For a command that is enough; a chain must know what each call
reads and writes, so the body states it: `reads=[t]` and `writes=[t]` name tensors the
body takes.

```python
km.cxx(ROTARY, takes=[qk, position], reads=[qk], on="mover_a")
km.cxx(APPEND, takes=[keys, position], writes=[keys], on="mover_b")
```

In a chain, a tensor in `reads=` is an outside input of the call: no record waits for
it, and a chain in which a call other than the one before writes it is refused; when
the call before writes it, the body waits for every block of that call. A tensor in
`writes=` is written by the call: it is in the chain's written set (no read of it goes
early). `writes=[(t, first, pages)]` states the pages each core's body writes (an
expression of the words, constants, `self.rank` and `self.size`, as a stream's): a
write like a stream's, block g of the call's output, which any later call reads as it
reads a stream's. `writes=[t]` writes anywhere in t: the next call that reads t waits
for every block of the call (the call's output, or t in blocks of one tile when the
call has none), in a call of groups of one; a later call's read of it is refused. A
body that writes runs on mover B, whose marks wait for its writes. `writes=[(t, first,
pages)]` on mover A is refused in a chain; `writes=[t]` on mover A, in groups of more
than one, or of a tensor the call also writes by pages, builds, and only the next
call's read of t is refused (a decode layer's cache append, which no call of the
layer reads, is one); a read that depends on its writes across a direct record
(`km.direct`) is refused, as a direct record's marks tell nothing of DRAM. A read whose first page or count is a
word of a table's entry (`lists.pages(e.first, 1)`) waits for every block of the call
before. A write at a table's words (`zbuf.pages(e.first, e.pages)` in an entries loop,
a frame's raster) is a write anywhere in its tensor, as `writes=[t]` on mover B: the
next call that reads it waits for every block. A read of it further on, in groups of
more than one, or of a tensor the call also writes by pages is refused where it is
read. A body that takes a tensor and states neither is refused by `run.chain`, and
still runs as a command. At the build, `reads=` or `writes=` that is not a list of
tensor parameters the body takes is refused at its line. A line that writes rows inside
a tile tensor's tiles, such as `keys.rows(position, rows).write(k)` for the cache's row,
is not built: it could be added as a write line of its own, which a chain would then
describe.

### A page from several CPUs' words

A page written once can hold the words of pack, mover A and mover B, as the ray
tracer's profile page does. Pack's body fills the page of a write stream outside a
loop. A body on mover B then takes the same stream: mover B waits for the page, the
body gets its L1 address and adds its words, and the movers write the page after that
body. Mover A's words reach mover B through an own field that mover A pushes
(`km.own(push="mover_a")`). The time words come from the loop's move: a mover's body
after a stream loop takes the loop's name and gets `name_tiles` and `name_cycles`. From
[raytrace_class](../../experiments/language/raytrace_class/trace_class.py):

```python
handoff: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_a")  # the reader's words
...
km.cxx(
    "...wait, read, total, pages into handoff...",
    takes=[rays, self.handoff],
    on="mover_a",
)
report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
km.cxx("...the walk's words...", takes=[report], on="pack")
km.cxx(
    "...the reader's words from handoff, then the writer's...",
    takes=[report, rays, self.handoff],
    on="mover_b",
)
```

The page is mover B's: a body on mover A that takes it, a body on mover B before
pack's, and a stream whose block is not mover B's (pattern other than `"B"`) are
refused.

### A write stream bodies fill

A write stream made before a loop and taken by a body inside it (on pack or on
compute's three) is filled one block each time the body runs: compute counts the
blocks (`km_fed_name`), reserves the block's half before the body (a timed reserve,
`name` 0 on pack when it gave up), the body packs into `name_buffer` by `pack_tile`, and
the line pushes the block after the body. The movers move the whole stream once,
after the outermost of those loops. A loop that also has movers' lines is refused
(the movers would wait on each other: compute's packs wait for room the write would
make only after the loop's reads). The stream's blocks are the body's runs: the
build does not check their count against the stream's pages. Each finished row's
`DV` tiles, a block a row:

```python
out = km.Stream(o.pages(first, rows * DV), write=True, block=DV)
for row in range(rows):
    km.cxx("finalize(compute, row, out, out_buffer);", takes=[out, row])
```

### A read stream bodies take

A read stream that is not iterated is one block, the read twin of a write stream bodies
fill: `q = km.Stream(pages, block=b)`, made on a line of its own and not in a for line.

- The movers move it at its line: mover A reads its block before the lines after it.
- Compute waits for the block at the line.
- A body on compute's three CPUs takes it (`takes=[q]`): `q_buffer` is the half mover A
  filled, `q` its first page there (0). The body may be in any loop inside the block
  of lines where the stream was made, and those loops may hold movers' lines too (a
  broadcast's casts).
- Compute frees the block when that block of lines' pass ends: made in a loop, at the
  end of each pass; made at the handler's top level, at the handler's end.
- A stream of no pages is one block too, pushed by mover A unread (compute waits for
  one block a stream). The pair holds one block a half: no block is read ahead into
  it while one is held.

The attention pass holds each job's Q rows through its K / V blocks
([attention_class](../../experiments/language/attention_class/README.md)):

```python
for job in jobs[first_job:end]:
    r0 = job.q // DT
    qs = km.Stream(
        q.tiles[
            r0 : min(r0 + R * job.stride, r0 + job.rows * job.stride) : job.stride, 0:DT
        ],
        block=R * DT,
    )
    for b in range((job.keys + KB - 1) // KB):
        ...  # the cast or the arrival of kv
        km.cxx("...matmul_block_no_mop(qs_buffer, kv_buffer, ...)...", takes=[qs, kv])
    # qs's block is freed here, at the end of each job's pass
```

Refused at the line:

- a stream both iterated and taken (`for t in q:` after it);
- a tile function that takes it: a tile function's operands are a stream loop's
  tiles, and a held block has no loop of its tiles;
- a block the build cannot bound: its most pages unknown (a side `r:min(r + n, e)`,
  rows `r:min(r + n * s, e):s`, or a known count says it) or more than its block;
- `row=`, `repeat=` and items, and a pattern whose block 0 is not mover A's;
- a take on one compute CPU or on a mover;
- a take outside the block of lines where the stream was made.

### Tensors in pages of several tiles

A tile tensor made with `page_tiles=k` ([host](host.md#tensors)) holds k tiles of a row
in one DRAM page. A read stream of its tiles is written as at k = 1
(`km.Stream(w.tiles[r0:r1, c0:c1], block=b)`). Its block is still in tiles, a tile a
page of the buffer, and the movers read the block one NoC read a page of k tiles. The
handler is built for its example's k, and a submit of a tensor of another k is refused,
naming both. Refused at the build, by name:
- a block, or a rectangle's first column or width the build knows, that is not a
  multiple of k (a page is never split; a first column or a width given at each submit
  that is not gives up at the stream's start, as a bad walk);
- a stream of its pages (`w.pages(...)`, a part) rather than its tiles;
- every other line over it: a fill, a write stream or a page write, page words, items,
  entries, a broadcast's cast;
- a chain's direct input in pages of k tiles.

At k = 1 the class's text is what it was. A read a page costs about 156 cycles beside
0.22 cycles a byte (experiments/language/bounds), so pages of 2 tiles in bfp8_b and 4 to
7 in bfp4_b read W near the card's DRAM rate.

### Items with a step

`t.items(first=f, count=n, size=B, step=s)` is items f, f + s, ..., n of them, of B
bytes in t's pages (the keywords of `.entries`; `step=` left out is 1, the run of
`t.items(first, n, size=B)`). A stream of them is a `km::Walk` of one item a row and
a row pitch of s items over `km::items`: one item of each of several pages, as a slice
of every core's counters page. From the rasterizer's SUM
([region_class](../../experiments/language/raster_class/region_class.py)), slice
`part` of each counting core's page of 128 items:

```python
entries = km.Stream(counters.items(first=part, count=counting, size=64, step=128))
```

`count=` and `step=` are each an int or a word of the handler (a parameter, an entry's
word); their most is n where n is an int, and none otherwise (refused where a most is
needed, as a rectangle's rows are). Refused at the line: a missing `first=`, `count=`
or `size=`, another keyword, an item size that does not divide the page, a count or
step that is an expression (`n * 2`: the build bounds an int or a word, not this), and
a count or step under 1.

### An item put, a word bumped

`t.items(size=B).put(self.f, at=i)` writes B bytes of own field f (from its start, or
item k of it: `self.f[k]`, at k x B) to item i of t, i / P of t's pages at byte (i % P)
x B, P = page bytes / B: one `km::put_bytes` by a mover (`on="mover_a"`, mover B when
not given). The put is not waited for at its line: each mover that put waits for its
puts once, at the handler's end (`km::put_wait`). A body that rewrites a source it put
waits for the write first (`noc_async_writes_flushed()`). `slot = self.f.bump(i)` is
word i of field f, then one more: on the mover an own field's `push=` names (its words
in L1), or on the unpack CPU for a field of `km.u32` words; `slot` is a word of that
CPU. The two lines stand alone or together (SETUP's record to its list's next slot):

```python
slot = self.next_slots.bump(page[1])
lists.items(size=128).put(page, at=slot)
```

Refused: a put on a compute CPU; a source that is not an own field (or a drained
stream's page, below); a size that is not whole words or does not divide the page;
`at=` a word of another CPU; a bump of a field whose words are no one CPU's (an own
field of `push=None` or found, a filled field).

### A word handed between the movers

A word a body gives on one mover, taken by a body on the other, is handed over by the
line: both CPUs are written (`on=` of each), never inferred.

```python
pages = km.cxx("...pages = ...;", gives=int, on="mover_a")
km.cxx("...use(pages);", takes=[pages], on="mover_b")
```

The giver writes the taking body's words into one 64 B page of a buffer of its own
(`kHandover`) and pushes it before the taker's place; the taker waits for the page,
gets the words by name and frees the page after its body. The build's report names
each: `hand-over: count, line 16: pages from mover A to mover B by the hand-over
buffer`. Its cost is one 64 B page of that added buffer, and the taker waits for the
giver: mover B's body does not start before mover A has pushed the page. A word given
and taken on the same mover is the plain take, with no hand-over. Refused: more than
16 words of the other mover in one body (the most a page holds), words of both movers
handed into one body, a giver or a taker with no `on=` (a compute word taken on a
mover, or a mover's word taken on compute).

### A stream compute fills and a mover drains

`out = km.Stream(self.f, mover="B")`, f an own field, is pages compute's bodies fill
and the mover named drains, a page a turn. A body on compute (on pack, or compute's
three) that takes `out` gets `out`, the L1 address of a page reserved before it (0
when compute gave up), and `out_buffer`; the line pushes the page after the body and
counts it. `out.end()` is compute's line saying no page follows: pack hands the count
of pages pushed to the mover in one page of the stream's end buffer (`kEndOut`).
Leaving a `with` block is exactly an `.end()` at that place:

```python
with km.Stream(self.out, mover="B") as out:
    for tri in range(count):
        km.cxx("bin(tri, out);", takes=[out, tri], on="pack")
for page in out:  # mover B: until the end's count
    slot = self.next_slots.bump(page[1])
    lists.items(size=128).put(page, at=slot)
```

`for page in out:` is the mover's loop: it waits for a page or the end's count (a
timed poll), stops when it has taken as many pages as the count says, and frees each
page once the writes that read it have read it. `page` is the page's L1 address (a
put's source, a body's on that mover) and `page[i]` its word i, an int of that mover.
The loop holds what a mover's lines may be: puts, bumps, writes, bodies on that mover.
The end costs 0.1 us a command: a command that fills no page, ends and drains takes
2.20 us beside 2.10 us of one with a pack body alone, and 64 pages a core filled,
drained and put take 14.24 us, on 120 cores
([drain_cost](../../experiments/language/raster_class/drain_cost.py)).

Refused, each at its line: a drain of a stream no line ends above (neither `.end()`
nor a `with`); `.end()` twice, or inside its own `with`; a body that takes the stream
after its end, or on a mover; no `mover=`; a field a mover pushes, or not an own field;
a drain of a stream no compute body fills; lines of another CPU in the drain loop.

## Tables of entries

A page of words can be a table: `t.pages(i, 1).entries(Job, first=F, size=S,
count=C)` is page i of t read on the CPUs of the handler that use it, its entries S words each
from word F, as many as word C says; `fixed=N` in place of `count=` says N entries,
known at build. `Job` is a class of int annotations, `class Job(km.Entry)`, its words
0, 1, ... of an entry in order (an entry may have more words than it names).

Where the page is read follows from the lines inside the loop: when only one mover's
lines use the entries (a body on that mover, a put, a bump), that mover reads the page
alone and compute and the other mover skip the loop; when compute's lines use them,
mover A reads it for compute and mover B its own copy, as below. The rasterizer's HIST,
whose tile list only mover B's body reads, went from 33.06 us (the page read on all
CPUs) to 24.80 us, beside the hand's 23.96 us
([raster_class](../../experiments/language/raster_class/README.md)).

```python
class Job(km.Entry):
    source: int
    count: int
    target: int
    add: int


for job in tables.pages(self.rank, 1).entries(Job, first=4, size=4, count=0):
    out = km.Stream(target.pages(job.target, job.count), write=True)
    for tile in km.Stream(source.pages(job.source, job.count)):
        km.cxx("...", takes=[tile, job, out, self.bias])
```

Each CPU has the same loop and the same words: mover A reads the page into a page it
pushes to compute, mover B reads its own copy, compute waits for mover A's page and
reads it in place (each of its three CPUs, the L1 cache invalidated), and frees it
after the loop. Inside the loop `job.count` is an int of every CPU, so the streams,
ranges and casts the loop holds take their pages from the entry, and a body that takes
`job` gets `job_source`, `job_count`, ... A count over the page's room ((page words -
F) / S entries) is taken as none, as the hand programs' tables do. Named,
`jobs = t.pages(i, 1).entries(...)` is read at the handler's top and kept to its end
(compute frees the page there): `jobs.count` is the count, `for e in jobs[a:b]:` a part
(ends clamped to the count, as a Python slice's). One entry known at build
(`fixed=1`) is a block of no loop: the ray tracer's directory page is

```python
class Span(km.Entry):
    first: int
    count: int


for span in directory.pages(self.rank, 1).entries(Span, size=2, fixed=1):
    hit = km.Stream(hits.items(span.first, span.count, size=HIT), write=True)
    ...
```

Against the line it replaced (`first, count = pages.read()[0:2]`, the words handed from
unpack to math and pack by mailbox), the ray tracer's class is 80 B of kernel text
smaller (49,712 B against 49,792 B) and 0.6 us slower in 3,290 us a command (0.02 %),
its hit pages equal word for word.

Refused at the line: an entries call without its class, its `size=`, or with both
`count=` and `fixed=` (or neither); a count's word inside the entries; a size under
the class's words; a page that holds no entry; a word the class does not name; a
named table below the handler's top; a slice with a step; a table that is not one page.

A record read by address, as the decode project's chain is (16 words, 64 B, at a DRAM
address of one bank, its next record's address a word of it), is not this line's: the
line reads one whole page of a tensor parameter by its page index once, at its line.
`entries(Record, size=16, fixed=1)` reads such a record when it is the first 16 words of
a tensor's page. What a chain needs besides: a read by DRAM address and bank (not by a
tensor's page index), of 64 B (not a page), and a loop whose next entry's address is a
word of the entry before (a read at each turn, not one at the line).

## Defines, the start-up, a kernel's start and end

`class P(km.Program, defines={"FLASH_CAUSAL": 1})` gives the kernels (the movement
source and the compute kernel, and the class's overlays) those defines, each an int or
a string; `P(..., defines={...})` adds or replaces some at the make (a causal call
beside a full one). A name of the runtime's (`KM_...`) is refused, and so is a value
that is not an int or a string. The build's report prints them.

`km.cxx(..., at="start")` puts the body at its kernel's start, before the first
command: on compute after the start-up, on the movers after their runtime is made; at
`at="end"` after the run (the kernel's last lines). It takes the class's own fields
alone (no handler's value is made there), gives nothing, and runs once a kernel, not
once a command; the names it declares are seen by the handlers' bodies (a constant
tile built once into an own field, an `init` of the vector unit). `on=` names its CPU
as for any body. compute's start-up is `compute_kernel_hw_startup` of the first buffer
the class reads and the first it writes; `startup=km.Startup("a", "b", "out",
order="reverse")` makes it of the fields a, b and out, in that source order
(`SrcOrder::Reverse`, the matrix unit's), as a hand kernel's start-up names its
operands. The start-up sets the unpacker, the math and the packer by those buffers'
formats and tiles.

## Tile functions

A **tile function** is a Python function of tiles that the tile compiler
([Tile functions](tile.md)) traces and emits for compute: decorated with
`@km.tile_function`, at module level or in the class's body (called `self.f(...)`), and
called in a stream loop on the loop's tiles. Its arithmetic is the tile compiler's
stream form: `+`, `-`, `*`, `/`, `abs`, number constants, `km.silu`, `km.sigmoid`,
`km.rsqrt`, `km.sqrt`, `km.reciprocal`, `km.maximum`, `km.minimum`, `km.relu`
(`maximum(x, 0)`), `km.square` (`x * x`), `km.exp`, `km.tanh`, `km.erf`, `km.log`
(`error=`, `over=`), `km.where(a > b, x, y)` (any of `>`, `<`, `>=`, `<=`, `==`, `!=`
of tiles and numbers, as its first argument only) and `km.floor` of up to four tiles,
one result ([select and floor](tile.md#what-a-tile-function-may-contain)). From
[22 gate](../../examples/22_gate/as_class/gate_class.py):

```python
@km.tile_function(tolerance=TOLERANCE)
def silu_gate(a, b):
    """The gate of two tiles: what the tile compiler emits for compute."""
    return km.silu(a) * b


class Gate(km.Program):
    """gate: each core's `tiles` pages of a and b from rank x tiles, silu(a) * b into
    the same pages of out, block by block in the command's pattern."""

    BLOCK = km.constant(8)  # tiles a block, as the hand program's kBlock

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
            out.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for x, g in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=self.BLOCK, pattern=pattern),
        ):
            results.push(silu_gate(x, g))
```

| Line | Movers | Compute |
|---|---|---|
| `for x, g in zip(km.Stream(...), km.Stream(...)):` | the read streams in one `movement.move` with the write streams the loop pushes to | block k of each stream from the half its mover filled; the streams have one count, block and repeat |
| `results.push(f(x, g))`, `t = f(x, g)` | | once before the blocks `copy_init` and the body's setup (`km_tile::f_init()`); DST held for as many tiles as half of DST takes (4 float32 tiles over the operands: 2 tiles of a gate an acquire), each tile's operands copied into its DST tiles in the order of the unit program's operands and `km_tile::f(d)` on the math CPU, then each tile's first DST tile packed into the write stream at its place |
| `for row in km.Stream(pages, block=4, row=2):` | one read stream, as without `row=` | each item a row of 2 tiles of the block; the block a multiple of the row |
| `results.push(self.f(row, self.w))`, `self.w` a field | | the row form: the field filled above the loop waited for before its blocks and freed after them; a row: its tiles copied into DST 0 and 1, the field's tile into DST 4, `km_tile::f(0)`, DST 0 and 1 packed; DST in full-sync mode |

**A tile expression in a handler's line.** `results.push(x / y)`, `t = km.silu(x) * y *
SCALE`: an expression of the loop's tiles, numbers, module constants and the tile
arithmetic above is the tile function of the tiles it names, in the order it names
them, and is emitted as that function's call is (`tile_line_N` in the build's list,
N the line): its unit program is the same, word for word, as the one a
`@km.tile_function` of the same expression gives. A field or `self` in such an
expression is refused at its line; a call that is no tile arithmetic (`km.cxx(...)`)
leaves the line to its own form. On the card
([tile lines](../../experiments/language/tile_lines/README.md), bf16 tiles, 768 a
command on 12 cores): `results.push(km.silu(x) * y)` equal datum for datum to the tile
function's call, 82.1 us both; `x / y` within 1.6e-7 of float64 at 30.3 us against
75.5 us for the inline body of tt-metal's `div_binary_tile`; `km.sqrt(x)` within 9.5e-8
at 40.2 us against 53.9 us for `sqrt_tile`; `km.minimum(abs(x), y)` exact; four tiles
`x * y + z * w` within 6.0e-8.

**A tile function's result as another's operand.** `results.push(scaled(larger(x, y),
z))`, `t = larger(x, y)` with `results.push(scaled(t, z))` on a later line of the same
loop, and `results.push(km.silu(larger(x, y)) * z)` are one tile expression: the calls
are traced into ONE unit program (`tile_line_N`), called once a tile, with no extra
`Dst` tile; inside a tile function a tile function's call is traced in place. The
composed function keeps a tile function's limits (at most 4 operand tiles, one tile
function a loop, the formats it takes); a name bound to a result is held, not computed,
when the loop's later lines read it only as an operand. Refused at the line: a result
both pushed (or written) and read by a later tile line ("t is pushed and read by a later
tile line": push the later line's result only, or compute it again there), and a field
or `self` inside the composed expression. A tile function called outside a handler's
tile line or a tile function raises by name.

**A float word in tile arithmetic.** A handler parameter `scale: float` is one command
word, the value's float32 bits (the host packs it at submit and in a chain's calls). In
a tile expression (`results.push(x * scale)`) or as a tile function's argument
(`results.push(affine(x, scale, shift))`, `def affine(x, scale, shift): return x *
scale + shift`) it is a run-time scalar: the call passes its bits
(`km_tile::affine(dst, scale_bits, shift_bits)`), and the unit program loads each into
a work register it holds across the rows, by two `TT_SFPLOADI` whose halves the core
builds from the argument, before the rows; the rows and their replay are the same
words as for a number. Each command takes its own value, with no build. A page's words
are float words when read as float32, `m0, m1, m2, m3 = cam.pages(r,
1).read(km.fp32)[0:4]` (a float32 tensor; `read()` gives ints): every kernel holds each
as a C++ `float` of the word's bits, so a camera row of a tensor page is a tile
function's scalars, more than a command's 10 words give. On the card ([page
scalars](../../experiments/language/tile_page_scalars/README.md), 12 cores, 4 tiles a
core): `clip(x, y, z, m0, m1, m2, m3)` with a row's words equal lane for lane to the
same command with float words, at 9.8 to 10.0 us against 9.6 on float32 tiles; on two
cards with the group across the edge 294 to 296 us against 276. An inline body
takes it as a C++ `float` of its name (`takes=[scale]`; tt-metal's `mul_unary_tile`
takes the bits: `__builtin_bit_cast(uint32_t, scale)`). Refused at the line: an `int`
word in a tile expression or a tile function's call (annotate it `float`, or read the
page's words with `read(km.fp32)`), `read()` with another argument than `km.fp32`,
`read(km.fp32)` of a tensor that is not float32 or of one word alone (`w =
pages.read(km.fp32)` is mover B's int: read a slice, `(w,) = t.pages(i,
1).read(km.fp32)[0:1]`), a float word in the handler's integer arithmetic (a slice bound, a range, a count), a division
by a float word (pass its reciprocal and multiply), and a float word of a function
called on a row (the row form's numbers are immediates). On the card
([run-time scalars](../../experiments/language/tile_lines/README.md#run-time-scalars-in-a-tile-function)):
bf16 tiles, 768 a command on 12 cores, four scales one command each:
`results.push(x * scale)` equal datum for datum to the inline body of tt-metal's
`mul_unary_tile` at each, 15.5 to 16.3 us against 33.6 to 34.0 us; `affine(x, scale,
shift)` within 5.9e-8 of float64 at the same time.

A function called on a row of a stream of `row=` tiles is traced in the tile
compiler's row form (`row_program(ir, 2, 2, 4)`): its first parameters are rows of the
streams, its last the fields it holds as resident rows, and its arithmetic the row
form's, the sum of a row (`km.reduce_sum`), its largest or smallest value
(`km.reduce_max(..., exact=True)`, `km.reduce_min`) and a value a row (`.col()`) or a
column (`.row()`) among it. From
[tile rules](../../experiments/language/tile_rules/as_class/norm_class.py), the row norm:

```python
class RowNorm(km.Program):
    """norm: each core's `rows` rows of WIDTH tiles of x from rank x rows, y = x
    rsqrt(mean(x^2) + eps) g into the same pages of out; g's tile filled from w."""

    BLOCK = km.constant(2 * WIDTH)  # tiles a block: two rows
    weight: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(tolerance=TOLERANCE)
    def norm(row, weight):
        ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * WIDTH))
        r = km.rsqrt(ms + rules.EPS)
        return row * r.col() * weight.row()

    @km.handler(stamps="stamps")
    def norm_rows(
        self, x: km.Tensor, w: km.Tensor, out: km.Tensor, stamps: km.Tensor, rows: int
    ):
        first = self.rank * rows * WIDTH
        self.weight.fill(km.Stream(w.pages(0, 1)))
        results = km.Stream(
            out.pages(first, rows * WIDTH), write=True, block=self.BLOCK
        )
        for row in km.Stream(x.pages(first, rows * WIDTH), block=self.BLOCK, row=WIDTH):
            results.push(self.norm(row, self.weight))
```

Several sums in turn and several fields take the row form in levels
([tile compiler](tile.md)): the layer norm, from
[tile levels](../../experiments/language/tile_levels/tile_levels.py), a weight and a bias
field filled as the weight above (field k copied into DST 2 width + k), at most 4 values
a row kept by `.col()` (the rows tile's slots):

```python
@km.tile_function
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDTH))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDTH))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()
```

A row of more than 2 tiles (`row=` 3 or more, the stream's block the whole row) takes the
chunk form ([tile compiler](tile.md)): in one acquire the sums' passes over the row's
chunks of 2 tiles and the per-row values, packed to a page of their own
(`kChunkRows<F>`); then an acquire a chunk, the chunk, those values and each field's
chunk in DST, the chunk packed to its place in the block. A field holds a tile a chunk.
The call is pushed (`results.push(layer_norm(row, self.weight, self.bias))`); assigning
it to a name, and stamps= on its handler, are refused. From
[tile wide](../../experiments/language/tile_wide/tile_wide.py), rows of 736 values:

```python
WIDE = 23  # tiles a row: 736 values (11 chunks of 2, a last of 1)
CHUNKS = 12  # chunks a row: the fields' tiles


class Wide(km.Program):
    BLOCK = km.constant(WIDE)  # tiles a block: one row
    weight: km.L1[km.fp32, CHUNKS, 32, 32]

    @km.handler
    def rms_rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, rows: int):
        first = self.rank * rows * WIDE
        self.weight.fill(km.Stream(w.pages(0, CHUNKS)))
        results = km.Stream(out.pages(first, rows * WIDE), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, rows * WIDE), block=self.BLOCK, row=WIDE):
            results.push(rms_norm(row, self.weight))
```

A row longer than its stream's block (`block=` an even part of the row, its rows a
constant count or a stated most) moves through L1 a block at a time: the range is read
once a level of sums, then once more for the writes (the read stream's repeat). A sum
pass holds DST for the row across its blocks and packs the per-row values' and the
sums' tiles to a page pair of the row's (`kChunkRows<F>`, a pair a row; one page, the
values', where the function has one level of sums), unpacked again at the next pass;
the last pass writes each block. A row that fits its block takes the chunk form
above unchanged. A field of a wide row may be bf16 (half the L1): its tile is unpacked
in its own setup, then the values' page's again. From
[tile long](../../experiments/language/tile_long/tile_long.py), rows of 4,096 values in
blocks of 8 tiles (L1: 933,888 B, the two fields 524,288 B of it):

```python
class Long4096(km.Program):
    BLOCK = km.constant(8)
    weight: km.L1[km.fp32, 64, 32, 32]

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        first = self.rank * 128
        self.weight.fill(km.Stream(w.pages(0, 64)))
        results = km.Stream(out.pages(first, 128), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, 128), block=self.BLOCK, row=128):
            results.push(rms_4096(row, self.weight))
```

A load whose buffers do not fit names its four largest with their bytes.

The rows may be a command word: a rectangle of whole rows whose rows side states their
most (`r:min(r + MOST, r + rows)`, as in the table of rectangles' rows) sizes
`kChunkRows<F>` at the most, and a command reads and writes its `rows` alone, the rows
past them untouched. A constant count pays nothing for it. From
[tile count](../../experiments/language/tile_count/tile_count.py), up to 64 rows of
2,048 values a core (the RMS norm's one page a row: 256 KiB at the most):

```python
class Counted(km.Program):
    weight: km.L1[km.fp32, 32, 32, 32]

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, rows: int):
        r0 = self.rank * MOST
        self.weight.fill(km.Stream(w.pages(0, 32)))
        results = km.Stream(
            out.tiles[r0 : min(r0 + MOST, r0 + rows), 0:64], write=True, block=8
        )
        for row in km.Stream(
            x.tiles[r0 : min(r0 + MOST, r0 + rows), 0:64], block=8, row=64
        ):
            results.push(rms_2048(row, self.weight))
```

A row function whose result is a per-row value alone (`return
km.reduce_sum(row * row, exact=True).col()`) may write it to a 1-tile float32 field of
compute's own, `self.part.write(square_sums(row))`: its sum passes run over the row,
then its part `value` puts each row's value in every column of one tile, packed into
the field (no write pass). A later loop of the handler reads the field by `.col()`; the
next write frees it. The decode's norm of one row over 16 cores, each core reading the
whole row for its sums and writing its own 4 tiles
([tile split](../../experiments/language/tile_split/README.md): 15.7 to 16.6 us a
command on bf16 tiles). A function whose result reads the row is refused there.

In a call on a stream's tiles, a field may be read by `.col()` alone: its column 0,
each row's value, is unpacked to every column (`unary_bcast<BroadcastType::COL>`) and
the call reads it as a per-row operand, as `.row()` reads row 0. From the same
experiment, the hand decode norm's last step on 16 cores, `s` each row's sum of squares:

```python
@km.tile_function
def rms_apply_2048(x, w, s):
    return x * km.rsqrt(s.col() * (1.0 / 2048) + EPS_RMS) * w.row()
```

The weight's tile is the row form's per-column operand, laid out by the host
(`p_tile` of [rules.py](../../experiments/language/tile_rules/rules.py)): the tile compiler's
row form reads it so. This layout is temporary: a plain weight row, laid out on the
card, is the tile compiler's to take.

The build traces the function on tiles of the streams' formats (`unit_program(ir,
load_macro=True)`), writes the body as `tile_<f>.hpp` and its unit program as text,
`tile_<f>.txt`, and prints its cost beside the buffers: the instructions and cycles a
tile on the math CPU by the cost model (the vector unit only, the copies and the pack
not priced), the rules that fired, and the tolerance the decorator states
(`tolerance=`, the largest relative error against float64). A loop that calls a tile
function holds no other tile line: the body's setup holds the vector unit's constant
registers for the loop.

A tile function may return several tiles (a state and an output): the call names each,
and each is pushed into its own write stream in the loop. Result k is DST k of the call;
the commit and the packs run at the last push of the call's results. From
[tile levels](../../experiments/language/tile_levels/tile_levels.py):

```python
@km.tile_function
def decay(x, h):
    h = DECAY * h + x
    return h, km.silu(h)


for x, s in zip(
    km.Stream(a.pages(first, TILES), block=8),
    km.Stream(h.pages(first, TILES), block=8),
):
    s2, y = decay(x, s)
    states.push(s2)
    results.push(y)
```

A field of one plain row (row 0 of its tile: 32 values, a bias or a scale a column)
passed to a call on a stream's tiles is read by `.row()`: the unpacker copies row 0 to
every row of its DST tile (`unary_bcast<BroadcastType::ROW>`). This is not the row
form's field (a call on a row of a `row=` stream, above). There `.row()` reads a field
of one tile for each chunk of 2 tiles, in `row_operand`'s layout ([tile](tile.md)), and
row 0 of a tile would give wrong values. From
[tile columns](../../experiments/language/tile_columns/tile_columns.py):

```python
@km.tile_function
def affine(x, scale, shift):
    return x * scale.row() + shift.row()


for x in km.Stream(a.pages(first, TILES), block=8):
    results.push(affine(x, self.scale, self.shift))
```

Format: a float32 field is unpacked to DST (exact), a bfloat16 one through SrcB (exact).
A field read whole in that call (`x * scale`) is refused, naming `.row()`.

A call holds 4 DST tiles (half of the fp32 DST), so a function returns at most 4 tiles.
Refused by name: the results pushed without a name (`out.push(decay(x, s))`), a result
left unpushed in the loop, several results on a row of 2 tiles.

A row function on a wide row (over 2 tiles, in one block or read in parts) may return
several rows, each named and pushed into its own write stream: result 0 is written over
the chunk (DST 0, 1), each other into 2 DST tiles of its own past the fields' chunks,
and each chunk's results are packed after its one call. With one field that is 2 results
(DST 0 and 5), with none 3; more are refused with the count it holds:

```python
@km.tile_function
def normed(row, weight):
    n = row * km.rsqrt(km.reduce_sum(row * row, exact=True) * (1.0 / 2048) + 1e-6).col()
    return n * weight.row(), n


for row in km.Stream(x.pages(first, rows * 64), block=8, row=64):
    y, n = normed(row, self.weight)
    ys.push(y)
    ns.push(n)
```

`@km.handler(stamps="t")` on a handler that calls a tile function makes the math CPU
read the wall clock around each call between two `tensix_sync()`, as the tile
compiler's measurements do, and writes page `rank` of `t` (uint32 pages of 19 words or
more): the command's sequence, the calls, 16 empty spans, then each call's span. From
the same example on the card: 768 calls, a median span of 1,606 cycles less the empty
span of 10, 1,596 cycles a tile, against the 1,597.6 the build prints. In the row form
the cost and the span are a row's.

A step's call (A step function on DST sums, Sums started from a field) is stamped as
two calls, its setup and its body, each between two `tensix_sync()`: the page holds a
setup's span then a body's for each step in turn. In the decayed rows below, with
`@km.handler(stamps="t")`, a command's page holds 2 n spans after the empty ones: the
cost of loading `decay`'s constants and the cost of its arithmetic, row by row.
A handler that calls steps also writes, in the page's last 9 words, each compute
CPU's (unpack, math, pack) wall clock from the handler's start to its end, the time it
waited for inputs (a wait for pages, a token or a block; pack's wait for math's DST)
and the time it waited for room (a reserve; math's wait for pack to free DST); its
spans then end 9 words earlier, and a page of fewer than 28 words is refused. What a
CPU did not wait is its own work: on math that includes the stalls of its unit on
unpack's sources, which no wait of its thread shows.

What the tile compiler does not lower is refused, naming the handler's line, the
function and the compiler's sentence, with "write this body inline: km.cxx(...,
gives=km.Tile)". A matrix product `a @ b` is taken in three forms: a sum over K of the
products of a stream's tiles by a field and a sum of two broadcasts' blocks in a
float32 field (Products, below), and a product of one K step inside a tile function,
`km.silu(a @ w1) * (a @ w3)` of three streams' bfloat16 tiles. There the unpack CPU
unpacks each product's two tiles (`llk_unpack_AB_matmul`, a into SrcB, w into SrcA)
after `mm_no_mop_init_short` once a loop, the body issues the products' words before
its vector code, nothing is copied into a product's `Dst` tile, and the compute
kernel's fidelity is the function's (`fidelity=`, HiFi4 if not given). A product of one
32 x 32 tile a side is lowered; the function called on a row (a K loop) is refused.
From experiments/language/tile_rules/as_class (`mm_gate_class.py`, card 0, 12 cores, 64 tiles a
core): within 5.6e-4 of float64 (2^-7 stated), 1,706 cycles a call less the empty span
against the tile experiment's BODY 29 at 1,707 and the build's 1,678.6.

From the 22 gate on the card (12 cores, 64 tiles a core, bfloat16), 2 tiles an acquire
against one: 83.70 us a command against 86.68, the hand program's 76.24. A tile takes
1,765 cycles of the command at 1,350 MHz (the hand's 1,608): the body's 1,597 and 168
for the copies into DST, the acquire and the pack. What is left of the difference is
the body: the emitted silu and product in float32 on the vector unit against the hand
program's library silu and its product by b on the matrix unit, which copies no b.

### A step function on DST sums

A tile function of two parameters, a row and its state, that returns the row and the
state's per-row values, `return p, (m, l)` (or a third, `km.word_max(...)`'s word), is
traced in the tile compiler's step form ([tile](tile.md)). Written as
`pages.write(f(acc, self.s[r]))`, acc DST sums of 1 to 3 tiles and self.s a float32
field of states (`km.L1[km.fp32, n, 32, 32]`, a state tile a page, filled before the
first step), state r is copied into the DST tile after the row, f's body runs on the
row and the state in place, the row is written as `pages.write(acc)` writes it and
the state is packed back into page r. A token buffer orders a state's pack before the
next step's copy: pack pushes after each state and unpack takes before the next; one
is pushed at the handler's start and the last taken at its end.
`t.pages(first, n).write(self.s)` writes the states after the last step. A word index
r is checked at run time (a give-up past n). The step's row sums stay on the vector
unit (`step_program(..., matrix_sum=False)`: a class makes no products with a tile of
ones). From [step lines](../../experiments/language/step_lines/README.md), a row
softmax over chunks of a product:

```python
@km.tile_function
def softmax_step(row, state):
    x = row * LOG2E
    m_old, l_old = state[0], state[1]
    m_new = km.lagging_max(m_old, km.reduce_max(x, exact=True), 0.0)
    alpha = km.exp2(m_old - m_new, error=2e-3, over=(-math.inf, 0.0))
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    l_new = alpha * l_old + km.reduce_sum(p, exact=True)
    return p, (m_new, l_new)
```

`p.pages(self.rank * C * N + c * N, N).write(softmax_step(acc, self.state[0]))` in a
loop over C chunks, acc the chunk's sums. Refused at the line: a row past 3 tiles or
not from DST 0, a field not filled above in the handler, a field not of float32
pages, a constant index past its pages, and a function that does not return the row
and its state.

### A product whose left side is a field

`self.h.write(acc)` packs acc's DST sums into a field compute writes (one row of K
tiles, `km.L1[km.bf16, K, 32, 32]`, K acc's width), in the field's format; so do
`self.h.write(f(acc))`, a function of the sums, and `self.h.write(g(acc, self.s[r]))`,
a step's row (its state packed after it). `out += f(self.h, wb)`, f `return a @ b`,
in `for wb in km.Stream(w.tiles[0:K, c:c + n], block=b n)`, then takes the field as
the product's left side: row k of the stream by tile k of the field, a row of n
products (the library's `matmul_block`, ct_dim n) a step into out's DST tiles. The
field is waited for before the loop and stays for any later product until the next
write frees it (its products are done by then); the handler's end frees the last.
A hidden layer kept in L1, from [step lines](../../experiments/language/step_lines/README.md):

```python
h: km.L1[km.bf16, KH, 32, 32]

for r in range(R):
    acc = km.tile.zeros(km.fp32, KH)
    for xb, wb in zip(
        km.Stream(x.tiles[row : row + 1, 0:K], block=2),
        km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
    ):
        acc += project(xb, wb)
    self.h.write(act(acc))
    out = km.tile.zeros(km.fp32, N)
    for wb in km.Stream(w2.tiles[0:KH, 0:N], block=O_BLOCK):
        out += project(self.h, wb)
    y.pages(row * N, N).write(out)
```

The left field may instead be a filled one: filled above in the handler, or by
another handler (kept between commands). It is waited for once in the handler, as a
product's right field is, and freed at the handler's end unless another handler fills
it. A tile function `return b @ a` is the same product with its operands swapped at
the call: `acc += f(xb, self.v)` is `self.v @ xb`. A vertical 3-tap filter, the taps
V (one row of 3 tiles) on the left of a column's 3 tiles, from
[denoise](../../experiments/forms/render/denoise.py):

```python
vtaps: km.L1[km.bf16, 1, 3, 32, 32]


@km.tile_function(fidelity="HiFi4")
def left(a, b):
    return b @ a


@km.handler
def load_taps(self, h: km.Tensor, v: km.Tensor):
    self.vtaps.fill(km.Stream(v.tiles[0:1, 0:3]))


@km.handler
def vertical_left(self, img: km.Tensor, out: km.Tensor, cols: int):
    acc = km.tile.zeros(km.fp32, 1)
    for xb in km.Stream(img.tiles[r : r + 3, c : c + 1], block=3):
        acc += self.left(xb, self.vtaps)
    out.pages(self.rank, 1).write(acc)
```

Refused at the line: a product of a field neither written above nor filled, a field
filled in a loop (the product frees it once), a write to a field that is filled,
summed into or a body's, a width not the field's tiles, a stream whose rows are not the
field's tiles or whose block is not whole rows, a tile function that is neither `return
a @ b` nor `return b @ a`, and a `return b @ a` whose arguments are not the loop's
block and a field.

### A field by a broadcast's block

`acc += f(self.h, w[i][r0:r0 + n, :])` (or `w[r0:r0 + n, :]`, or `w[a:b, :]` of ints
the build knows), f `return a @ b`, self.h a row of n tiles compute wrote and w a block
a broadcast brought, takes tile k of the field by row r0 + k of w's part (acc's width
of columns): n products of the library's, outside a stream loop. Any number of them
take a block: it is waited for at its cast and freed at the end of the cast's lines
(a body does not take it then). A hidden layer whose second weights come down the
column once (on mover A, beside mover B's page writes):

```python
column = km.Broadcast(self, along="column", window=1, mover="A")
if column.sender():
    ws = column.cast(w2.tiles[0:W2_ROWS, 0:N].read())
else:
    ws = column.arrive()
for r in range(R):
    acc = km.tile.zeros(km.fp32, KH)
    for xb, wb in zip(km.Stream(..., pattern="B"), km.Stream(..., pattern="B")):
        acc += project(xb, wb)
    self.h.write(acc)
    out = km.tile.zeros(km.fp32, N)
    out += project(self.h, ws[r * KH : r * KH + KH, :])
    y.pages(row * N, N).write(out)
```

Refused at the line: a field not written above, rows that are not the field's
tiles, a part not of acc's columns or cast turned, and a product inside a stream loop.
For a block cast outside a loop, a stream, a page write or a fill on the cast's mover
after it moves the cast's finish (the wait for compute to free the block) to the
handler's end, after the mover's last line.

### A held stream's row by a broadcast's block

`acc += f(xs[r], w[i][:, c0:c0 + n])` (or `w`), f `return a @ b`, xs a held stream
(`xs = km.Stream(x.tiles[r0:r0 + R, 0:k], block=R k)`, waited for at its line and
freed at the end of its lines) and w a block a broadcast brought, takes tile j of
xs's row r by row j of w's part (acc's width of columns, transposed where the cast
turned it): k products of the library's, outside a stream loop. The block is waited
for at its cast and freed at the end of the cast's lines, as for a field's products.
Rows of X kept in L1 while blocks of W^T come down the column:

```python
xs = km.Stream(x.tiles[self.y * R : self.y * R + R, 0:K], block=X_BLOCK)
for b in range(B):
    if column.sender():
        wb = column.cast(wt.tiles[b * N : b * N + N, 0:K].T)
    else:
        wb = column.arrive()
    for r in range(R):
        acc = km.tile.zeros(km.fp32, N)
        acc += project(xs[r], wb)
        y.pages(...).write(acc)
```

Refused at the line: a part not of acc's columns, a constant row past the held
block's (a word is checked at run time), a stream whose rows' tiles the build does
not know, and a product inside a stream loop.

A cast whose rectangle may be shorter than its slot (a side `r:min(r + m, e)`) leaves
the slot's tail holding an earlier block. A window into such a part whose end the
build does not know, here or in a field's product by the block's rows, is checked at
run time against the rectangle, and a window past it gives up. It is not checked
where the sums started from a row of a field (`s = start(self.mask[i], self.s[r])`),
whose -inf columns make the tail's scores vanish, nor in a product whose left field
was written from such sums. So a row of X can take W^T's blocks three columns at a
time with the last step part empty: the mask row for that step holds -inf past the
block's columns, and P is zero there.

### Sums started from a field

`acc = f(self.o, self.s[r])`, f a step function (above) and self.o a float32 field
compute writes (1 to 3 tiles), starts DST sums from the field: its tiles are copied
into DST 0 .., state r into the DST tile after them, and f's body runs on both; acc's
products then add into the tiles f left (`acc += project(xb, wb)`), and acc's write,
back into the field (`self.o.write(acc)`) or to pages, packs the row and then the
state into page r with the token. So a field is updated in place from itself, a
step's per-row values and a product's sums: `o = o d + X_c W_c` over chunks, d a
per-row value in a state slot.

```python
@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


zero = km.tile.zeros(km.fp32, N)
self.o.write(zero)
for c in range(C):
    acc = decay(self.o, self.state[0])
    for xb, wb in zip(km.Stream(...), km.Stream(...)):
        acc += project(xb, wb)
    self.o.write(acc)
```

A field of rows (`km.L1[km.fp32, n, w, 32, 32]`, filled above, w 1 to 3) is read and
written a row at a time, in place, as a step's states are: `acc = f(self.o[i],
self.s[r])` copies row i into DST 0 .. w - 1 after the field's token, and
`self.o[j].write(acc)` packs acc into row j by index and pushes the token, then
acc's state. A row is written only from sums started from one of the field's rows
(the token orders each read after the last write); `t.pages(first, w).write(acc)`
writes acc elsewhere and gives the token back, the row unchanged; `t.pages(first,
n w).write(self.o)` writes the rows after the last. So each of n rows keeps its own
running sum, `for r in range(n): acc = decay(self.o[r], self.state[0]); ...;
self.o[r].write(acc)`.

A field that starts at zero need not be read from DRAM: `self.f.zero()` packs a zero
tile into each of its pages on compute and pushes them as a fill does, so step states
and rows of running sums are read from it as from a filled field. The decayed rows
above start so, with no tensor of zeros:

```python
self.state.fill(km.Stream(init.pages(self.rank, 1)))  # the decays: data
self.o.zero()  # n rows of w running sums, zero
for r in range(n):
    acc = decay(self.o[r], self.state[0])
    ...
    self.o[r].write(acc)
```

Refused at the line: arguments, a field filled (or zeroed) above and not taken, sums
still in DST, an own field.

Sums started from a row with state r carry state r in DST after their tiles. A step
on them with the same state, `pages.write(f(s, self.s[r]))` (or a field's write),
runs f on it there: its token is not taken again, and s's write packs it once. So a
row of a filled field of constants can be the start of sums that a step then reads,
such as an offset added before a product's sums go through a row's step:

```python
s = start(self.offsets[i], self.state[r])  # start: return row, (state[0],)
s += project(xs[r], wb)
y.pages(...).write(shift(s, self.state[r]))
```

A step on another state than the one the sums started with is refused.

The same line with no products after it divides a field's rows by a per-row value
at their write: `last = norm(self.o, self.state[r])`, norm `return row /
state[1].col(), (state[0], state[1])`, then `y.pages(...).write(last)` (a division a
value on the vector unit; a reciprocal kept a row in a free slot is not lowered yet).

Refused at the line: a field not written above in the handler, a field not 1 to 3
float32 tiles, and DST still held by other sums.

## Products

A tile function whose body is `return a @ b` of its two parameters is a matrix
product, the library's `matmul_block`. Its operands are tiles of any two matrix
formats of one exponent family, each side its own (bfloat16 and the B family's block
formats, or float16 and the A family's; see [tile formats](formats.md)); its sums
float32. A float32 stream of two streams in step is read as tf32 (`km::tf32_formats`
sets that source itself, as a hand kernel does; [unpack and pack](formats.md#unpack-and-pack)).
A float32 operand of any other product (a field, a cast, a held stream's row) or of a
turned product is refused at its line ("a product's left operand is float32: ..."):
stream a bfloat16 copy of the tensor or declare the field `km.bf16`. A class of A-family buffers computes in a
16-bit DST. The left operand goes into SrcB, unless the tile function places its
operands with `km.srcb(...)` or `km.srca(...)` ([placement](formats.md#placement)):
`return x @ km.srcb(w)` builds the turned product, its W tensor stored with transposed
tiles (`session.tensor(..., transposed_tiles=True)`). Either way the build prints both
placements with the significand bits each operand keeps.

A product's weight is in one of two forms:

- **A field.** `acc += self.f(block, self.w)` in a loop over X's stream adds the product
  of each tile of the stream by the field's row of `acc`'s width of tiles into `acc`.
  The field is `km.L1[km.bf16, K, n, 32, 32]` (or another matrix format, `km.bfp8_b`
  a block-float weight at 1,088 B a tile), K rows of n tiles, filled whole before
  the first product: the form for a weight that is kept (filled by one handler, read by
  others, or read by several loops).
- **A stream.** `for xb, wb in zip(km.Stream(x.tiles[0:1, k0:k1], block=b),
  km.Stream(w.tiles[k0:k1, c:c + n], block=b * n)):` reads X's tiles and W's rows in
  step, a tile of X's stream a row of W's (b K tiles a block of each), in one move of
  the movers; each W block is used as it arrives and freed, so W's read and the
  products overlap. X's stream is the zip's first: the loop counts its tiles, and the
  last block holds the rest of K.
- **A broadcast's block.** `acc += self.f(xb, ws)` in a loop over X's stream, `ws` a
  block `b.cast(w.tiles[k0:k1, c0:c1].read())` or `b.arrive()` brought before the
  loop: the block is a weight one core reads and every core of its row or column
  shares. Each tile t of block k of the stream takes K row k x b + t of the block's
  part. `ws[:, c0:c0 + n]` takes a window of n of the part's columns; a cast of
  several parts names one, `kv[i]` or `kv[i][:, c0:c0 + n]`; a part cast transposed
  (`t.tiles[...].T`) is multiplied with the library's transpose flag. The block is
  waited for before the loop and freed after it. Refused at the line: a cast of
  several parts named whole, a window past the part's columns, a width other than the
  sums', and a window whose width the build does not know. A cast outside a loop
  whose block a stream on the cast's mover follows finishes at the handler's end
  (compute frees the block after that stream's last block), so the stream may use
  both movers.

A product's sum is in one of two places:

- **DST.** `acc += self.f(...)`, `acc` a `km.tile.zeros(km.fp32, n)`: n float32 tiles of
  DST, each W block's product added tile by tile.
- **A field of sums.** `self.part += self.f(xb, wb)`, `self.part` a float32 field: the
  blocks' product by subblocks of DST (h x w tiles, at most 4), each packed onto the
  field's tiles with the packer's L1 accumulation (overwriting at the first sum). The
  blocks are two streams' (the field one row of n tiles, `km.L1[km.fp32, 1, n, 32,
  32]`) or two broadcasts' arrived blocks (rows x k and k x columns, `b.cast(...)` or
  `b.arrive()`), as the dense project's grid form.

A field of sums starts from zero at each command: its first block packs without the L1
accumulation, and its count of sums is 0 at the handler's start; nothing of one
command's sums is kept for the next. One loop may add two products into two fields
(`self.h += self.p(xb, w1b)` and `self.g += self.p(xb, w3b)`, X's block read once for
both): each is counted on mover B for its write, and a product whose setup differs from
the loop's first (W's format, the block's shape, a turn) sets its own before its K
tiles and the first's after them. A tile function of a product's sums is written in
the same handler: of DST sums, `y.pages(first, n).write(act(acc))` runs act's body on
each of acc's tiles in place before the pack (on the card silu over 4 tiles adds 1.1 us
a tile to the product's command, and the result is the product's own error:
[a product and a tile function](../../experiments/language/tile_lines/README.md#a-product-and-a-tile-function-in-one-handler));
of fields of sums, as one rectangle (the table above): the README's SwiGLU layer,
[31 swiglu](../../examples/31_swiglu/README.md), whose `up` handler writes
`gate(self.h, self.g)`.

**Two sums in DST and a function of both.** A second `km.tile.zeros` while the first
sums are held and unwritten, in the same loops, zeroes its tiles after the first's
(no second acquire), and `pages.write(f(a, b))` runs f's body on each tile t of the two
with the second operand read its distance apart in DST, then writes the first's tiles
as `pages.write(a)` writes them:

```python
h = km.tile.zeros(km.fp32, 4)  # DST 0 to 3
for xb, wb in zip(km.Stream(x.tiles[r : r + 1, 0:32], block=2), km.Stream(...)):
    h += self.project(xb, wb)
g = km.tile.zeros(km.fp32, 4)  # DST 4 to 7
for xb, wb in zip(km.Stream(x.tiles[r : r + 1, 0:32], block=2), km.Stream(...)):
    g += self.project(xb, wb)
hidden.tiles[r : r + 1, c : c + 4].write(gate(h, g))  # silu(h) * g, no fields
```

The two are one width and together at most DST's 8 fp32 tiles ("h (5 tiles) and g (4
tiles) take 9 of DST's 8 fp32 tiles; make each 4 or fewer"); a write or a reduce of one
of them alone is refused (its pack would release the other's sums), and so is a function
of two widths. The tests are
[test_class_dst_pair.py](../../tests/core/dsl/compile/dst/test_class_dst_pair.py); the layer over 128
rows with its gate so, [31 swiglu, the gate in up](../../examples/31_swiglu/README.md#the-gate-inside-up).

The size rule: a sum in DST is at most 8 float32 tiles (4 in the half-sync DST, 8 in
full sync), two sums held together the same 8; a larger sum is a float32 field, 4 KiB
a tile of L1.

`whole = km.reduce(acc, form="tree", group=S)` adds the group's sums at its first
rank, and so does `whole = km.reduce(self.part, form="tree", group=S)` of a field of
sums: mover A takes each arrival into a slot of the field's tiles, compute copies it
into DST and packs it onto the field with the packer's L1 accumulation, then hands the
field on, to mover B, which sends a sender's field to its receiver (`km::reduce_emit`),
or to the leader's write. The reduce's size rule: of a sum in DST, its n tiles and one
for the arrivals, at most 8 (n at most 7); of a field, the field and its slots in L1, a
slot of the field's n float32 tiles for each round by tree (the bits of S - 1, 7 when
the group is the command's) or for each other core of the group by owner (which takes
`group=`). At 16 tiles in groups of 14: 4 slots, 256 KiB. The handlers' reduces share
one arrival buffer, the size of the largest one reduce takes (its slots times its
tiles), each reduce counting its own slots in it: a tree of 2 over 16 tiles and one of
14 over 1 tile take 16 tiles, not 4 slots of 16.

A rectangle `t.tiles[r0:r1, c0:c1]` takes any ints of the handler as its bounds
(command words, expressions of `self.rank`, `self.x`, `min` and `max`), its widths read
at run time, so a last column of cores narrower than the others is the same line:
`w.tiles[k0:k0 + KC, left:min(left + OW, N_TILES)]`. Where the line holds the
rectangle's pages, the build needs a side's most: a side `c:c + n` is n, and
`c:min(c + n, e)` at most n.

- A cast's slot holds the most rows and columns, the rows at the most columns' pitch,
  each row's words read; a side whose most the build cannot tell is refused.
- A field's fill and a write of a field of sums take the field's pitch, a row of the
  field a row of the rectangle; a side that may be more than the field's is refused.
- A stream's blocks are whole rows: a stream of more than one row takes constant
  columns (its rows may be words).
- A step on the rows (`r0:r1:s`) takes rows s apart; a step on the columns is refused.

### Rows with a step

`t.tiles[r0:r1:s, c0:c1]` is the rectangle's rows r0, r0 + s, ... below r1, the
`(r1 - r0 + s - 1) / s` rows a `km::Walk` with a row pitch of `s x T` pages; s is an
int or a word of the handler. A stream of it moves whole rows a block, and a block
may be partly read: `km.Stream(q.tiles[r0:min(r0 + R * s, r0 + rows * s):s, 0:D],
block=R * D)` reserves R rows whole, reads `rows` of them (at most R) and hands the
block on; compute takes the read tiles alone. Where the line holds the rows (a cast's
slot, a field), their most is stated by the slice:

| Rows side | Rows | Their most |
|---|---|---|
| `a:b:s`, all ints | `(b - a + s - 1) / s` | the same |
| `r:r + n * s:s` (n an int) | n | n |
| `r:min(r + n * s, e):s` | `(min(...) - r + s - 1) / s` | n |
| `r:r + m:s`, `r:min(r + m, e):s`, s an int | as above | `(m + s - 1) / s` |
| anything else | as above | none: refused where a most is needed |

### Several ranges into one block

A cast takes one part or more, each a page range, `.read()` or not: `t.pages(first,
n)` (n an int), `t.tiles[...]` (a rectangle, a step on its rows allowed) or
`t.tiles[...].T`, the rectangle transposed: its columns in turn, a column a row of
the slot. Each part takes the slot pages of its most (a side's most as above), the
next part starting after them, so a short part leaves its unread pages in place and
a body finds every part at the same page:

```python
for chunk in range(chunks):
    k0 = chunk * KB
    if down.sender():
        kv = down.cast(
            k.tiles[k0 : min(k0 + KB, keys), 0:DT].T,  # slot h x KB + key
            v.tiles[k0 : min(k0 + KB, keys), 0:DV],  # slot KB x DT + key x DV + c
        )
    else:
        kv = down.arrive()
```

The sender reads a transposed part a page at a time (`km::read_pages` of page
`first + key x T + h` to slot page `h x KB + key`), a rectangle a row at a time at its
most columns' pitch, a run of pages in one call; the parts' reads are joined by `&&`
before the cast. The block is the parts' mosts summed (here `KB x DT + KB x DV`), in
the slot `window` times. Refused at the line: a part that is not a page range of a
tensor, a part with a side whose most the build cannot tell, and parts of tensors of
different formats or page sizes. `.T` is a cast's: a stream, a fill or a write of a
transposed rectangle is not a line.

A body on compute's three takes the block whole, `takes=[kv]`: each CPU waits for it
(`km::broadcast_receive`) before the body, which reads `kv_buffer` from page `kv` (0)
on, at the parts' places, and frees it after (`km::broadcast_done`). A block is taken
once, by a body or by `for tile in kv:`; a second take, or a body on one CPU
(`on="pack"`), is refused at its line. The [table of jobs](../../experiments/language/class_lines/README.md)
casts K's chunks transposed and V's into one block and packs it whole to a write
stream: 0 of 112 filled slot pages differ on the card, on the sending and the
receiving row.

`@km.tile_function(fidelity="HiFi2")` sets the matrix unit's fidelity, tt-metal's word
(LoFi, HiFi2, HiFi3, HiFi4): the compute kernel's `math_fidelity`, so a class's
products take one; HiFi4 when no product gives one. From
[the dense project's linear layer as a class](../../experiments/language/dense_class/linear_class.py),
the dense project's two plans, picked by the shape (the K split for one row tile of X,
the grid form for more):

```python
class Linear(km.Program):
    K_TILES = km.constant(128)  # K = 4096
    N_TILES = km.constant(128)  # N = 4096
    SPLIT = km.constant(14)
    WIDTH = km.constant(16)
    SPLIT_CORES = N_TILES // WIDTH * SPLIT  # 112
    CHUNK = km.constant(2)
    W_BLOCK = CHUNK * WIDTH  # W's tiles a block: CHUNK rows of the group's columns
    COLUMNS = km.constant(12)
    KC = km.constant(2)
    OW = -(-N_TILES // COLUMNS)  # 11
    OH = km.symbol()  # the plan's
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]
    part: km.L1[km.fp32, OH, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        share = self.rank % self.SPLIT  # this core's part of K
        first = self.rank // self.SPLIT * self.WIDTH  # the group's columns of Y
        k0 = share * self.K_TILES // self.SPLIT
        k1 = (share + 1) * self.K_TILES // self.SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w.tiles[k0:k1, first : first + self.WIDTH], block=self.W_BLOCK),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=self.SPLIT)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.WIDTH].write(whole)

    @km.handler
    def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        along = km.Broadcast(self, along="row", window=2)
        down = km.Broadcast(self, along="column", window=2)
        top = self.y * self.OH
        left = self.x * self.OW
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if along.sender():
                xs = along.cast(x.tiles[top : top + self.OH, k0 : k0 + self.KC].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(
                    w.tiles[
                        k0 : k0 + self.KC, left : min(left + self.OW, self.N_TILES)
                    ].read()
                )
            else:
                ws = down.arrive()
            self.part += self.project(xs, ws)
        y.tiles[top : top + self.OH, left : min(left + self.OW, self.N_TILES)].write(
            self.part
        )
```

`split` is the dense project's K split for one row tile (its plan 14,1,16): 112 cores
in groups of 14 consecutive ranks, each group 16 columns of Y, each core a fourteenth
of K (9 or 10 tiles); X's tiles and W's rows stream in step, 2 K tiles a block, each
pair's product packed onto the 16 tiles of `sums`; the group's sums are added by a tree
of 4 rounds, the first rank its owner. `grid` is the grid form on the hand's 12 x 8
rectangle: core (x, y) makes Y's block of OH x 11 tiles, the last column 7; X's chunks
of 2 K tiles are cast along the row and W's down the column, and each core adds the two
blocks' product into its field.

| Line | Movers | Compute | Cost |
|---|---|---|---|
| `for xb, wb in zip(km.Stream(x.tiles[0:1, k0:k1], block=2), km.Stream(w.tiles[k0:k1, c0:c1], block=32)):` | two read streams in one move: X's run of K tiles, W's rows of c1 - c0 tiles (a `km::Walk`), 2 rows a block | block k of each from the half its mover filled; the rest of K in the last | the reads of X's tiles and of W's rows |
| `self.sums += self.project(xb, wb)` | | before the blocks: `matmul_block_init(x, w, 0, 4, 1, 2)`, the pack's format; a block: the field reserved at the first, `pack_reconfig_l1_acc` on after it; for each subblock of 1 x 4 tiles the products over the block's K tiles, packed onto the field; both blocks freed | 32 tile products a pair; the pack and its L1 accumulation not priced |
| `acc += self.project(xb, wb)`, `acc` n tiles of DST | | tile t of xb: `matmul_block(x, w, t, t x n, dst, 0, n, 1, b)` | n tile products a tile of X |
| `self.f.fill(km.Stream(w.tiles[k0:k1, c0:c1]))`, a kept weight | mover A reads the field's rows, one read of c1 - c0 pages a row | waited for once a handler, before the first product; freed at the handler's end (or at the next fill, when another handler reads it) | the field's bytes, once a command |
| `acc += self.project(xb, ws[:, c0:c0 + n])`, `ws` a broadcast's block of a part p columns wide from page o | | before the blocks: `km::broadcast_receive`, a stall of the math CPU for the vector unit's fills, `matmul_block_init(x, ws, t, n, 1, b)` (t the part's transpose); a tile t of block k: `matmul_block(x, ws, t, o + (k x b + t) x p + c0, dst, t, n, 1, b)`; after the blocks: `km::broadcast_done` | n tile products a stream tile |
| `acc += self.project(block, self.w)` | | before the blocks: the field's wait, a stall of the math CPU for the vector unit's fills, `matmul_block_init(in0, field, 0, n, 1, 8)`; a tile t of block k: `matmul_block(in0, field, t, (k x 8 + t) x n, 0, 0, n, 1, 8)` | n tile products a stream tile, 16 MVMUL words a phase (HiFi2 2 phases, HiFi4 4) |
| `whole = km.reduce(self.sums, form="tree", group=14)` | `km::reduce_arrive` of 16 tiles a slot on mover A; `km::reduce_emit` of the field's 16 pages on mover B, in `km::digit_group(command, me, 1, 14)` | each arrival's 16 tiles copied into DST and packed onto the field; the field pushed | 4 rounds; 16 float32 tiles (64 KiB) sent a round, 16 copies and packs |
| `if self.rank == self.leader:` `y.tiles[0:1, c0:c1].write(whole)` | the owner (`rank - rank % 14`) writes 16 pages | the field's sums copied into DST 4 at a time and packed in Y's bfloat16 | the pack of 16 tiles |
| `xs = along.cast(x.tiles[top:top + OH, k0:k0 + KC].read())`, `xs = along.arrive()` | the sender reads the block a row at a time and casts it; the others receive it | | the block's bytes from DRAM once a row (or column) of the rectangle |
| `ws = down.cast(w.tiles[k0:k0 + KC, left:min(left + OW, N_TILES)].read())` | the slot holds KC x 11 tiles; the sender reads each row's `min(left + 11, 128) - left` tiles at the slot's pitch of 11 | | the columns' bytes |
| `self.part += self.project(xs, ws)` | | both blocks received; the field reserved at the first pair; `matmul_block_init(xs, ws, 0, w, h, k)`, `pack_reconfig_l1_acc` on after the first pair; for each subblock of h x w tiles (4 x 1 here: 11 has no divisor up to 4) the products over k, packed onto the field's tiles; both blocks freed | rows x columns x k tile products a pair; the pack and its L1 accumulation not priced |
| `y.tiles[top:top + OH, left:min(left + OW, N_TILES)].write(self.part)` | mover B writes each row's `min(left + 11, 128) - left` pages of the field's 11 | the field pushed, its tiles copied and packed into Y's format, the sums' count reset | the pack of rows x columns tiles; Y's bytes |

A class that mixes tile formats (bfloat16 operands, float32 sums) reconfigures the
unpack and the pack before each copy, product and pack (`reconfig_data_format`,
`pack_reconfig_data_format`); one that does not has none. A field of sums counts as a
format compute packs.

The build prints each product line's cost: the words the library issues, priced as the
tile compiler's matrix unit runs them (`matrix_cost`), from blocks rotating to a block
alone, as the library's order is not known to it:

```text
products (the library's matmul_block, priced by the tile compiler's matrix_cost; the library's order, rotation not known):
  split, line 74, `self.sums += self.project(xb, wb)`: project, a pair of blocks into the field's 16 tiles over k = 2: 32 tile products, 1024 MVMUL words (HiFi2), 1031.0 to 6146.0 cycles a pair; the unpack, the pack and its L1 accumulation not priced
  grid, line 97, `self.part += self.project(xs, ws)`: project, a pair of blocks into the field's 44 tiles over k = 2: 88 tile products, 2816 MVMUL words (HiFi2), 2823.0 to 16898.0 cycles a pair; the unpack, the pack and its L1 accumulation not priced
```

On card 1 (a Blackhole p150a), seed 1, HiFi2 as the dense project's rows, one load for
both shapes, against its hand kernels at the same shapes (their text by a build on a
mock device):

| Y = X W, M x K x N | Plan | Values | Class | Hand (the dense project) | The first version of the class |
|---|---|---|---|---|---|
| 32 x 4096 x 4096 | K split, groups of 14, 16 columns | integers: exact, 0 wrong; normal bfloat16: relative RMS 4.38e-3 against float64, the SwiGLU of two products and the gate 8.03e-3 | 94.2 us a command (median of 3); 90,192 B of kernel text | 92.4 us; 61,584 B | 111.9 us (groups of 3, 4 columns, the weight a field) |
| 1024 x 4096 x 4096 | grid form, 12 x 8 | the same: exact; 4.38e-3, 8.03e-3 | 358.6 us; 90,192 B | 382.4 us; 69,296 B | 434.3 us (8 x 8) |

The account beside the first version's, whose terms were measured by the plans' parts
run as commands of their own
([linear_parts.py](../../experiments/language/dense_class/linear_parts.py), the first version's
plans), against the card's DRAM ceiling of 495.6 GB/s:

| K split, 32 x 4096 x 4096 | First version | Now | Whose |
|---|---|---|---|
| W into the field before the first product, 33.55 MB | 87.5 us (383 GB/s), then X's stream and the products, 24.4 | W read under the products: the whole command 94.2 us, W at 356 GB/s over it | the lowering's, gone: each W block used as it arrives (a product over two streams) |
| X read by each group of 4 columns, 8.45 MB (6.2 MB more than the hand's) | in the 24.4 | X read once a group of 16 columns, 2.1 MB, as the hand's | the form's, gone: a group as wide as the hand's (a reduce of a field of sums) |
| left: 1.8 us over the hand's (2 %) | | not split by term: no part was run. Its candidates: W in blocks of 2 K rows (64 KiB), three blocks of four on mover A (the reads' AAAB), where the hand reads chunks of 5 (160 KiB) on both movers in turn; after the last block its products, the tree's 4 rounds and Y's write, the hand's in form | |

| Grid form, 1024 x 4096 x 4096 | First version | Now | Whose |
|---|---|---|---|
| W's casts by the column senders, 33.55 MB | 402.6 us by 8 senders (10.4 GB/s a sender) | 12 senders: at that rate a sender, 269 us | the form's, gone: 12 columns of 11, the last 7 (a rectangle whose bounds are words) |
| the rest: X's casts, the products, the last pair, Y's write | 31.7 us after the casts | 358.6 us in all, 24 us under the hand's | not split by term: no part was run. K chunks of 2 (L1: at 4 the casts' slots and the K split's buffers are over a core's L1 with the runtime's), 64 casts a column where the hand casts 8 of 16 |

The gate after the products, as the Tile functions section's: 7.3 us and 92.7 us.

## Fields

A field is declared on the class with its format and shape; a symbol of the shape is
declared in the class's body, `TILES = km.symbol()` (or `km.constant()`), and used as
`count: km.L1[km.fp32, TILES, 32, 32]`; a size may be a constant of the class too,
each program's value ([Constants](#constants)). Each field is a buffer of whole tiles, index 8
up. A symbol is the plan's; one symbol left unknown is solved as the largest for which
the fields and the buffers fit `l1`, and the report prints it.

A field is filled from a stream of a page range, of its own tiles' count, or of a
rectangle of tiles, `t.tiles[r0:r1, c0:c1]`, whose columns are the field's (a read a
row; the card refuses rows past the field's). One `km.cxx` body in the same block takes
it, or a product reads it; every core that fills it frees it. `@` of a field is the
Products section's; slices and `.columns` of a field are not built.

A field one handler fills and another reads (a product, or a body that takes it) stays
in L1 between commands, with no other line: the reader waits for it once and does not
free it, and the filler frees its last fill before the next. The build prints it:

```text
  field weight: filled by load, read by apply (it stays in L1 between commands)
```

A field that no handler fills is still refused where it is read. A field is declared
on the class itself: a subclass does not inherit its parent's fields (declare them
again).

A field of fewer than 1,024 `km.u32` words, `km.L1[km.u32, n]`, is words the host
reads after the run: the class is made with a tensor of the field's name, one row of
at least n words in each core's L1 (`shard_grid`), whose address is compute's program
argument. A line sets a word, `self.f[i] = x` or `self.f[i] += x`, and is the unpack
CPU's alone (one write, not one a compute CPU); its value may read the field's words,
and no other line reads them. From
[14 message handlers](../../examples/14_message_handlers/as_class/handlers_class.py):

```python
totals = session.tensor(np.zeros((cores, WORDS), np.uint32), "uint32", shard_grid=grid)
program = MessageHandlers(totals=totals, log=log).dynamic(
    "lookup", "value", "report", **hand.options(args)
)
```

and after the run `totals.read()`, a row a core. On islands the session's own
`session.tensor(..., shard_grid=session.grid)` makes it on every island at one address,
`[islands x cores, words]` ([islands](islands.md)): a tensor of one island
(`session.island(i)`) is refused at the load, naming the cause.

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

A broadcast's handler is resident. `@km.handler(overlay=False)` keeps a handler
resident also where a merged program's build would move it (below).

**Resident and overlay handlers in one class.** A class may keep some handlers resident
and make others overlays, tile code on both sides (the decode layer's chain with `o`
and `down` resident and `qkv`, `attend`, `gate_up` as overlays). A part's compute
copies of the build's 22 constant tables of the buffers pass when their bytes are the
resident kernel's (a part built against other buffers is refused, naming the table);
a math part that defines matmul.h's `throttled_mop_status` is built with its copy and
the resident kernel's set unknown around the call
([hazards](hazards.md#a-part-that-keeps-something-from-one-call-to-the-next)). On the
card that layer's chain gave the seven tensors word for word equal to the hand's, in
342.11 us a layer against 342.25 us with all five as overlays
([decode_class](../../experiments/language/decode_class/README.md)).

## Classes merged into one

Several classes become ONE program in the kernel form: one program field, one handler
numbering, one L1 layout. Two ways give the same program:

```python
merged = km.merge(gate, linear, session=session)  # programs already made
run = session.load(merged)
run.submit(gate.gate, (0, 12), args=(a, b, out, 64, pattern))  # a source's handler


class Layer(Gate, Linear):  # the classes themselves
    pass


layer = Layer(x=x, w=w, y=y, a=a, b=b, out=out, session=session)
run.submit(Layer.split, (0, 112), args=(x, w, y))
```

`km.merge(a, b, ..., overlay=[...], resident=[...], name=None, session=None, l1=None,
log=print)` takes made class programs: each handler is compiled again with what its
own program was made with (its constants, plan and example tensors). A class deriving
two or more classes with handlers, adding no handler, constant or field of its own, is
made as one class is (`Layer(x=x, ..., session=session)`): the make's keywords go to
each class, a constant to each class that declares it. Its sets are not: `handlers=`
of such a class is refused ("`handlers=` of a merged class is not built (its handlers
take their defaults); make each class with its sets and merge the made programs
(km.merge)").

- **Numbering.** The first class keeps its numbers; each next one takes the next
  numbers in its own order. The build prints them (`4: linear.gate as linear_gate
  (linear's 3)`). A name two classes have is named `<class>_<name>` in the kernels and
  the header; `merged.handler("gate")` is refused naming both, `merged.handler(gate,
  "gate")`, `layer.handler(Gate, "gate")` or the merged name pick one, and on a merged
  class `Layer.gate` is refused the same way. A source's own handler (`gate.gate`,
  `linear.split`) given to a run of the merged program is mapped to its number there.
  More handlers than a program numbers (239) is refused with the count.
- **Constants.** Each class keeps its own: two classes' `K_TILES` are two values, each
  handler's `self.K_TILES` its own class's. On a merged class a keyword for a name two
  classes declare differently is refused naming both (merge made programs to give each
  its own value); declared alike, the one value goes to both. The plan's names are
  shared: one name with two values is refused.
- **Tensors.** A tensor parameter is a compile argument of the movement kernel, by its
  name. Two classes whose parameter of one name has two different example tensors (the
  gate's `a` of 768 pages, Linear's `a` of 1,024 x 4,096 tiles) each keep theirs: the
  second class's is the kernel's `<class>_<name>`, and its handlers' lines name it in
  a block of their own (`using A = LinearA;`).
- **Fields and buffers.** A field of one name in two classes is one field when declared
  alike (format and shape, by each class's constants), else refused naming both:
  `field sums is Field(...) in Sums and Field(...) in Wider`. The streams' buffers are
  one class's: a pair is shared by handlers of one role, format, page and block, as
  between the handlers of one class, so the L1 they take is at least each class's.
  The build's L1 check refuses what does not fit, with the bytes: the gate class's
  `BLOCK = 8` merged with Linear asks 1,642,496 B of the 1,414,272 B left; at
  `BLOCK = 2` the gate's streams share Linear's pairs and add nothing.
- **Where the code lives.** Resident unless marked: `overlay=[...]` and
  `resident=[...]` name handlers (a name, a merged name, a source's handler) or whole
  sources (a made program, a class), and `@km.handler(overlay=True / False)` is a
  handler's own marker; the merge's markers override it. When the kernels do not fit
  the session's kernel buffer, the load places the largest resident handler that may
  move (unmarked, not a broadcast's; by its generated C++: its lines in both kernels
  and the tile functions only it calls) as an overlay, prints its choice and loads
  again, until they fit; a handler whose part the overlay builder refuses stays
  resident and the next is tried; with none left, the load's `KernelSizeError` gives
  the bytes. Each such pick makes the program again, so a field set on it after the
  make goes back to the class's (its `opt`, `noc`, threads, ports), except the defines
  the load itself added (an island load's), which are put back. At every load the session prints where each handler's code lives,
  resident ones too.

Measured: [merged_class](../../experiments/language/merged_class/README.md), the gate
class (`BLOCK = 2`) and Linear merged, on one card: every handler exact or within its
bound as alone; the gate 80.84 us against 80.54 alone, Linear's split 93.86 against
93.70, its gate 7.74 against 7.30. The kernels 98,656 B (Linear's alone 92,304, the gate's
38,064), so a kernel buffer of 97 KiB; at 91 KiB the load's pick moved `split`, then
`linear_gate`, then `gate_gate`, and the overlay builder refused each part (a stream's
handler keeps its own copy of `km::move_filled` and `km::move_drained`), so all four
stay resident and the load is refused with the bytes.

Not merged, each refused with its reason: a dynamic handler (its message loop is its
program's own kernels); a made program with `km.ANY` among its sets (its sets are
compiled on first use against its own program's load: list them to merge them);
`handlers=` on a merged class (`class Layer(A, B)`: its handlers take their defaults;
merge made programs to give sets); two classes whose handlers of one name both have
constant parameters (a set is picked by the handler's name); classes whose compute options or start-ups
differ; a define or the rectangle given two values; a merged program as a source; a
class in the kernel form with a hand program, and `form=` or `beside=` at the make of a
class merging classes (it is built in the kernel form). Classes made with
`form="runtime"` merge with each other and with hand programs, in the runtime's form
([Programs merged into one](host.md#programs-merged-into-one)).

A handler with constant parameters merges as its program built it: `km.merge` of a
program made with `handlers={"gate": [{"size": 4}, {"size": 8}]}` gives the merged
program `gate_size4` and `gate_size8`, at the numbers the class's own build gives them
(the first class keeps its numbers, the next ones follow), or `gate` at its defaults
when no set is listed. A call names the class's handler with the constants
(`Gate.gate`, `size=4` at its place) and the merged program's handler of that set
takes it; a set not listed is refused as in one class. The gate class merged with
GateBlocks made with blocks 8 and 4 listed ran on the card, 120 cores, 16 tiles a core:
`Gate.gate` 43.74 us, block 4 43.32 us, block 8 42.0 us, each 0 wrong.

A chain runs over a merged class's handlers as over one class's
([Chains](#chains)). `class Layer(Scale, Rest)` gets the chain's walk from a `km.chain`
method of its own or of a class it merges, or from `chain=True` at its make;
`km.merge(a, b)` of made programs gets it when a source was made with it. A chain's call
names a handler through the merged class (`Layer.scale`, `program.steps(...)`) or
through the class that declares it (`Scale.scale`), which the merged program's own copy
then stands for. Each handler's chain waits are read from its own class's file. Example
29's steps split over two classes and merged both ways ran on the card with the
unmerged class's words (list and method, factors 5 and 7, 0 wrong) and its time a chain
(21.48 / 20.94 us merged as a class, 21.18 / 21.02 by km.merge, 21.44 / 20.86 unmerged).
A merged program in the runtime's form has no walk (a class's walk is in its own
kernels).

## Dynamic handlers and messages

A handler with `for message in self.arrive():` is [dynamic](dynamic.md): its lines are
compute's alone and the runtime's movers carry its messages (`messages/reader.cpp`,
`messages/writer.cpp`); the program is made dynamic as any program is, with
`.dynamic(name, **options)`. A move in a dynamic handler is refused. A message is a class deriving `km.Message` whose int annotations are its
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
| `if self.seeding():` | `km_messages.seeding()`, its lines, then `km_messages.seeded()` |
| `seed = Visit(id=..., ...)` | 16 words, the named ones set, the others 0 |
| `self.send(rank, seed)` | `km_messages.send_or_give_up(rank, seed)`: with no quota left the run gives up (`quota`, the destination: the timeout's text says what to write), never a message dropped unseen |
| `went = self.send(rank, message)` | `const uint32_t went = km_messages.send(rank, message) ? 1 : 0`: 0 when this core has no quota left and nothing was sent ([quota](dynamic.md#how-a-message-travels)) |
| `for message in self.arrive():` | each packet of `km_messages.next()`, each message of it copied, then `km_messages.done(km_packet)` |
| `message.state`, `message.sum += x` | the message's word |
| `self.forward(rank, message)` | `km_messages.forward(rank, message)` |
| `went = self.direct(rank, message)` | `const uint32_t went = km_messages.direct(rank, message) ? 1 : 0`: 1 when it went into its slot at rank, a [declared pair](dynamic.md#a-direct-message); 0 when the slot is busy and nothing was sent (the call is refused anywhere but as a name's value) |
| `self.consume()` | `km_messages.consume()` |

A `km.cxx` body in a dynamic handler can call the handler's `km::Messages` by its name,
`km_messages`. For example, `km_messages.quota()` gives the sends this core has left,
as 05 forward's body reads it.

A class may have several dynamic handlers. One, the RUN command's (the program's
`.dynamic(name, ...)` names it first), has lines besides its loop: the seeding, a
word field's zeroing; each other is its loop alone, no words. Their loops are one
loop: each message is taken by the loop of the handler it names, and
`self.forward(rank, message, to=Class.handler)` (or `self.send(..., to=)`) names the
handler at the rank it goes to; without `to=` a message keeps its handler (a seed:
the RUN command's). From
[14 message handlers](../../examples/14_message_handlers/as_class/handlers_class.py):

```python
@km.handler
def value(self):
    for message in self.arrive():
        self.totals[VALUES] += 1
        message.sum += message.value
        message.steps -= 1
        if message.steps != 0:
            key = message.value ^ 0x85EBCA6B  # the next key
            key ^= key >> 16
            key *= 0x7FEB352D
            key ^= key >> 15
            key *= 0x846CA68B
            key ^= key >> 16
            message.key = key
            owner = key % self.size
            self.totals[OWN] += owner == self.rank
            self.forward(owner, message, to=MessageHandlers.lookup)
        else:
            self.totals[OWN] += message.origin == self.rank
            self.forward(message.origin, message, to=MessageHandlers.report)
```

| Line | Compute |
|---|---|
| the loops of `lookup`, `value`, `report` | one loop; a message's case by `messages.handler()`: `case kHandlerValue:` and the loop's lines; a message of no handler of the class consumed |
| `self.forward(rank, message, to=MessageHandlers.lookup)` | `messages.forward(rank, message, kHandlerLookup)` |

In a dynamic handler `self.rank` and `self.size` are the message loop's
(`messages.rank()`, `messages.count()`): the group without its counter tiles. A core's
totals go where a `km.cxx` body puts them (05 writes them to an L1 area whose address
is a command word).

## X280 threads

A thread stays C, built for the X280 and calling `kmx_*`. Each build writes
`<name>_thread.h`, a C header for a thread's commands, from the same build as the
kernels, so the C and the class cannot drift apart. It holds:

- each built handler's number (a constant set's under its set's name, a merged
  program's at the merged number), the place of each of its words (a0 is 0), their
  count, and whether its command asks for the leader's result, as before;
- the program field (`kProgram<Name>` in a run of several programs, else 0) and the
  class's constants at this make (`WORKCLUSTER_MAX_ROUNDS`);
- the load's words: each tensor of the make, its address, as a word the load gives the
  thread after the thread's own (`<NAME>_WORD_STATE`), its pages, and
  `<name>_state_dram(first)`, the `kmx_dram` that `kmx_fetch` and `kmx_store` take;
- the thread's regions: each one's id for `kmx_region` and its bytes, and a region's
  bank address and bank, a region parameter's two words;
- a function for each built handler that fills its whole command (`kmx_command`, the
  flags, the group, each word) from typed arguments: the make's tensors and the
  thread's regions come from the load, so they are not arguments;
- for a handler with tensor parameters, `<function>_with(...)`: the same arguments,
  then each tensor parameter's load-word place (`<NAME>_WORD_<TENSOR>`, of the make's
  tensors or of `thread_tensors=`, which join the load's words), and its pages words
  from that tensor. It returns `KMX_BAD` when the named tensor is of another kind than
  the make's (page bytes, format, layout or place: the kernels were built for the
  make's), else `KMX_OK`.

The class states its thread in its body: its regions with `km.region(bytes)` (an int
or of the class's constants; ids in body order), and its C with
`km.x280_thread(*sources, cluster=1, deadline=, includes=, defines=, prefetch=,
host_fallback=)`, sources and includes relative to the class's file.
`thread_words=[...]` at the make gives the thread's own words. From
[07 work cluster](../../examples/07_work_cluster/as_class/README.md):

```python
class WorkCluster(km.Program):
    MAX_ROUNDS = km.constant(1000)
    groups = km.region(4 * MAX_ROUNDS)
    thread = km.x280_thread(
        "thread/driver.c", "../decide.c", cluster=1, includes=[".."], deadline=10.0
    )

    @km.handler
    def step(self, state: km.Tensor, round: int, first: int): ...


program = WorkCluster(state=state, MAX_ROUNDS=rounds, thread_words=words)
```

```c
/* STEP, line 35: state the make's tensor, round, first. */
static inline void workcluster_step(
    km_command_t* km_command,
    uint32_t km_flags,
    uint32_t km_first,
    uint32_t km_count,
    uint32_t round,
    uint32_t first
) {
    kmx_command(km_command, WORKCLUSTER_PROGRAM, WORKCLUSTER_HANDLER_STEP, km_first, km_count);
    km_command->hdr |= km_flags << 8;
    km_command->a0 = workcluster_word(WORKCLUSTER_WORD_STATE);
    km_command->a1 = round;
    km_command->a2 = first;
}
```

The thread is made and its C built at the make, before any card job: the header the
thread was built with is in a folder of its own (`build/<name>-<hash>/thread-<hash>/`),
put first among its includes. The make refuses a region parameter that names no
region of the class. It also refuses C that names a handler, a tensor or a region
the program lacks. The message gives the class's line, the C's errors with their
lines, and what the program has:

```
Missing, line 97: the thread's C does not build with missing_thread.h:
  .../missing.c:10:5: error: implicit declaration of function 'missing_nothing'; ...
  missing_nothing: the program has no handler nothing (['total'])
```

`program.thread(...)` after the make, as a hand program's (`Program.thread`), keeps
working: the thread gets the same header, its words come first and the load's follow,
and it is built at that call. A class declares one thread, and `program.thread` is
refused on a class that declares one. On a made program, `program.thread` (the
declaration's name) is the `komorebi.program.thread.Thread`, and `program.threads` holds it as
a hand program's. `thread` names the module `komorebi.program.thread`, so the class's spelling
is `km.x280_thread`.

What stays out: nothing makes the thread anything but C, and no Python runs on the
X280. A set added on its first use (`km.ANY`) has no number at the build: the host
submits it, never the thread, and the header says so. In a load of several class
programs (`session.load({adder: (0, 4), doubler: (4, 4)})`, each made with
`form="runtime"`), the thread's header gets each other class program's parts at that
load: its constants, its command functions (whose commands carry its program field),
and its tensors as thread words after the thread's own. Every class program's header
has `<name>_load(first, count, deadline, &event)`, which is `kmx_load` of its program
field (`KMX_BAD` in a load of one program). C that names only another program's
functions is built at that load, not at the make; C that names what its own program
lacks is still refused at the make
([test_class_thread_programs.py](../../tests/core/dsl/thread/test_class_thread_programs.py),
example 21's [thread_load_class.py](../../examples/21_thread_load/as_class/thread_load_class.py)).
In a merge of classes (`class Layer(A, B)`), the thread one class
declares is made after the merge, with the merged numbers. Its regions are sized by
that class's constants, `thread_words=` and `thread_tensors=` go to it, and its C
includes `<layer>_thread.h`, where each class's functions keep their class's name.
Two classes that each declare a thread are refused, and so is `km.merge` of a program
made with its thread. A merge without a declared thread takes one after the merge
(`program.thread`). A region's size
is not checked against the pages a handler reads of it. In a run of several programs
the load builds the thread again with the programs' defines (`kProgram<Name>`); the
image is built at that load.

### A thread's region as a parameter

A handler's parameter `counts: km.Region` is an X280 thread's region: two command
words, its bank address and its bank (the bank's word after the others). In the
handler it is a tensor of the region's pages of 4,096 B in that one bank
(`km::RegionPages`, km_region.hpp): `.pages(...)`, `.items(...)`, streams, fills,
puts and bodies on a mover take it as they take a tensor parameter. `Run.submit`
takes it by the region's name, the region of the run's thread that has it (refused
when none or more than one has), or as (bank, bank address), what
`run.threads[c].regions.address(name)` gives:

```python
run.submit(RegionPasses.sum_tiles, (0, cores), args=(counters, prefixes, "sums", cores))
```

A region is in its island's DRAM: a program made with island tensors, whose other
islands' cores would read it, is refused at the parameter, naming the region's island
(`km.Region(bank, address, island=1)` as its example).

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

## Chains

A [chain](chain.md) can be a list of a class's handler calls: the class states the calls,
and each call's description (its groups, the block it writes, what it reads and from
which call) comes from its handler's lines, with no `kc.Call` written by hand. There are
two forms, and they give the same records
([29_chain as a class](../../examples/29_chain/as_class/steps_class.py)):

```python
class Steps(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, factor: float, width: int): ...

    @km.chain
    def steps(self, x, y, z, s, t, factor: float, width: int, tiles: int):
        self.scale(x, y, factor, width)
        self.scale(y, z, 3.0, width)
        self.add(z, km.past(y), s, width)  # y: written two calls back
        self.sum(s, t, tiles)


chain = run.chain(program.steps(x, y, z, s, t, 5.0, 2, 16), cores=(0, 8))
listed = run.chain([Steps.scale(x, y, 5.0, 2), ...], cores=(0, 8))
run.wait(run.submit(chain))
```

`Steps.scale(x, y, ...)` outside a command is a call, not a run: its values in the
handler's order (a constant parameter at its place, which picks the handler built for
it), and `cores=(first, count)` for its group when the handler has no parameter of that
name (else `run.chain`'s `cores`). A `kc.Call` written by hand is refused in the same
list: its tensors are names the class's calls do not share, so no input or prefetch
between them could be checked.

**A word given at each submit.** `km.at_submit("position")` in place of an int word of
a call makes that word the chain's per-submit word: the record holds 0 at its place
and names it, and `run.submit(chain, position=n)` gives it, so one chain built once runs
a decode layer at each position (or another layer's row, a step count) with no record
written again. One word a chain, at most one a call (the same word may go to several
calls, e.g. through the chain method's parameter). The calls' descriptions must not
depend on it: a read or a write whose pages change with the word is refused ("what it
reads or writes changes with width, given at each submit"), and so is the word at a
tensor, a pattern, a float or a constant parameter. A word bodies take (`takes=[...]`)
or a range of a body's own reads is free to change.

```python
chain = run.chain(program.layer(*tensors, km.at_submit("position")), cores=(0, 120))
for position in range(3):
    run.wait(run.submit(chain, position=position))
```

The tests are [test_class_chain_at_submit.py](../../tests/core/dsl/chain/inputs/test_class_chain_at_submit.py).

**A chain the thread runs.** `km.x280_thread("t.c", chains={"steps": {...}})` puts the
class's `km.chain` method `steps` in the thread's header. The dict gives its arguments:
a parameter named as a tensor of the make is taken by that name, `"x": "a"` names one,
a value is a value, `km.at_submit("n")` is the chain's per-submit word, and
`cores=(first, count)`, `run.chain`'s, is required. A tensor the make lacks is given with `thread_tensors={"z": z}` at
the make and joins the load's words. The chain is described and proven at the make as
`run.chain` does it, and its refusals name the class's line. The header then has
`<class>_chain_steps_write(region)`, which writes the records into the thread's region
(the last first, each naming the next; each 256th with next 0), and
`<class>_chain_steps_command(&command, region, word, piece)`, which fills piece `piece`
of the chain command for `kmx_submit` (its cores, the program field, the exclusive
flag; one piece a 256 records, `<CLASS>_CHAIN_STEPS_PIECES`, each submitted once the one
before is done); `<CLASS>_CHAIN_STEPS_BYTES` is the region's least size. A `km.direct` input needs nothing more from the thread: the
record's flag is the set-up, the landing area is the program's, and the host refuses
to read that tensor's DRAM (it has no copy) until a write. Every record matches
`run.chain`'s word for word, except `next`, which is the region's address
([test_class_thread_chain.py](../../tests/core/dsl/thread/test_class_thread_chain.py)).
Example 29's [thread_steps.py](../../examples/29_chain/as_class/thread_steps.py) ran it
on a card with 0 wrong words.

**The walk.** A class with a `km.chain` method, or made with `chain=True`, has the
chain's walk in its kernels: a handler of its own after the class's (`km_chain`), the
chain's words in buffer 32 (2,048 B), each kernel's case for a chain command handing
each record to the handler it names (`km_chain.hpp`'s reader, writer and compute walk),
each read of the record's input waiting for its marks, each case ending in its marks, a
tree's send waiting for its credit. A class without either has none of it. In example 29
the walk adds 15,920 B of kernel text (74,160 against 58,240 B: BRISC +4,332, NCRISC
+7,880, each TRISC about +1,220), past the default buffer of 70,656 B: such a class's
session gives more with `kernel_bytes`
([Host API, `Session`](host.md#session)). A plain command of a class with the walk took
0.16 us more on the card (21.25 against 20.62 us for four).

**Overlay handlers in the walk.** A handler marked `@km.handler(overlay=True)` keeps its
body in its overlay's parts in a class with the walk too. Its case in a chain's record
loads the overlay (the reader's load and the other cores' meet, as for a plain
command, [overlays](overlays.md)), then waits for the record's input whole when the
handler reads a tensor a call of the chain writes, then calls the part, then ends in
the record's marks. The part cannot reach the walk's state, so its reads cannot wait
line by line, as a resident handler's do: they all start after the input is marked,
and a `prefetch` of an outside tensor within the record is lost. A tree's send in a
part waits for its credit as in a resident case (`km::chain_credit` reads only the
chain's words). The slots are the overlays' (a slot a handler by default, up to four;
`program.slots(count=...)`): when they hold every handler's parts, a record loads
nothing (72 cycles: the check), and when they hold fewer, a record's load waits until
the core's other CPUs have ended the record before
([overlays, the meet](overlays.md#costs)) and then reads the parts. Example 31's chain
([swiglu](../../examples/31_swiglu/README.md#up-and-down-as-overlays)) took 105.67 us
resident, 111.92 us with up and down as overlays at the default slots (the whole
input waited for) and 115.70 us at 1 slot (two loads a chain), its words the same in
all three. Loading the next record's parts while a core waits for its input is not
built ([chain](chain.md#a-handler-as-an-overlay)).

**The lines a call is described from.** A handler's streams of page ranges
(`t.pages(first, count)`, a partition's share `s[part]`, a rectangle `t.tiles[r0:r1,
c0:c1]`), their zip, `.write` of a page range or of a rectangle (`t.tiles[r0:r1,
c0:c1].write(...)`), `.drain` and `.fill`, the assignments
before them, one `km.reduce(..., form="tree")` and the leader's `if` around a write (after the reduce, or `if self.rank % G
== 0:` with G the reduce's group: each group's root; before the reduce, rank 0 alone,
refused in a call of several groups). A
first page is a number of the call's words, the class's constants, `self.rank`, `size`
and `leader` (`+ - * // %`, `min`, `max`). From these: the output (the first tensor
written; each group's root writes block g, `width` tiles from g x width), the group size
(the reduce's group, else 1), and each read as a share of K tiles (each group's ranks
reading the same K tiles as `km::partition` shares them, whether by `self.partition` in
one group or by the rank's share written out) or as ranges of `base + step x g` (a
root's alone under its `if`). Groups whose reads are not of that form (a start
clamped at the end, `min(g, last)`) wait for the narrowest `base + step x g` range
that holds each group's evaluated range: a few more marks, never fewer; a direct input
read so is refused, and so is a read no range word holds. The decode layer's five calls (112, 120, 96 and 120 cores in groups of 7,
15, 2 and 15) are described so, both ways with equal records; where its chain stands
on the card is in [decode_class](../../experiments/language/decode_class/README.md). The record's input is the call before's output; another read is an
outside tensor (no call of the chain writes it) or, marked `km.past(t)`, an earlier
input that `run.chain` proves (as a written chain's,
[the earlier input](chain.md#the-earlier-input-and-the-direct-input)); unmarked, it is
refused ("call k (name): reads t, written by call j (name) further back; mark it
km.past(t) at the call, so the build proves it complete or refuses"). A rectangle of
several rows that are not whole rows (a weight's columns, `w.tiles[0:K, c:c + n]`) is
no range of tiles: it is read as an outside tensor, and refused when a call of the
chain writes it. A call's `cores=(first, count)` puts it on its own group (the README's
SwiGLU: `up` on 12 cores, `down` on 4).

**A share of K by island.** On islands a call's groups may read a share of K whose base
steps by island (`self.rank // 118 * K` on two cards of 118 ranks: island i's groups
from base + i x step, the same K, no group across an edge): its record carries the step,
each core waits for its own island's tiles and each root marks the cores, on any island,
whose share meets its block. The walk follows such a share in every island build with
the chain's walk (a `km.chain` method or `chain=True`: `KM_CHAIN_K_ISLANDS`, 316 B more
of the writer's text); a class chained without the walk is refused naming it, and a
direct input or a thread's chain of such a share is refused. On two cards (2026-11-17) a
chain of FILL over 236 ranks and TAKE in groups of 2, each card's groups summing their
own 118 pages, was exact as commands and as a chain (104.1 us).

**A covering range on islands.** A read that is not `base + step x g` waits for its
covering range, and on islands every mark of it from a rank of the other island crosses
the link as a put line and its confirmation, about 1.2 us each, one after another. On
two cards (2026-11-18, `experiments/forms/nn/halo_exchange.py`) the conv reads `zl` and
`zr` at tile row `(rank // CT + dy) x CT`, plus PAD on island 1: two ranks share a row,
and the rows jump at the edge. Its range is `Range(0, 0, 80)`, every block of the shift,
so each of its 32 ranks waits for all 40 of the shift's:

| As | µs | puts across |
|---|---|---|
| each alone: halo, shift (commands), conv (a chain of one) | 11.7, 10.9, 44.8 | 0 (the halo asks 16 pages) |
| chain [halo, shift] | 74.1 | 80 |
| chain [shift, conv] | 785.8 | 1,280 |
| chain [halo, shift, conv] | 837.0 | 1,360 |

A call with such a read whose group spans the edge runs as a command; or its reads are
written as `base + step x g` on each island (a call per island).

**A write by island.** On islands a call whose groups on each island write that island's
copy from block 0 (`first = self.rank % per * TILES`, `per` an island's groups: an add
of two cards' partials into each card's own copy) is a write by island: group g of an
island writes block g there. It needs every island to hold the same groups, none across
an edge, and its one output a copy (`copy=True`); else it is refused naming them, and so
is a gather or a direct input of its output. Its record carries a flag (the info word's
bit 31: a class's handlers number 1 to 127 in a chain); its roots mark their own
island's cores only, and a core after it waits for the blocks of its own island's
groups. The walk follows it in every island build with the chain's walk
(`KM_CHAIN_BY_ISLAND`: 308 B more of the writer's text, 4 B of each compute kernel's).
On two cards (2026-11-17) a 27B-class decode layer cut in turns (qkv and gate and up by
columns, o and down by K, the partials gathered) as one chain of 8 records (qkv, o, h's
gather, h's add by island, gate and up, down, y's gather, y's add by island) gave each
copy's words as the commands do (the adds 1.7e-3 against float64) in 670.7 us bfp8_b and
468.6 us bfp4_b, one command a layer, where the same layer as four commands (a chain,
h's add, a chain, y's add) took 658.5 and 465.7 us of the cards' time and three more
host round trips.

**Several rows.** A call whose roots each write rows r0 to r1 - 1 of their block (a
rectangle, `t.tiles[r0:r1, c:c + w]`, or a row a turn of `for r in range(n):`, whose
rectangles taken together make one) is described in tile columns: its width is w
columns, block g is columns g x w on over all those rows, and a mark tells of the
block's every row (the root marks after its writes are flushed). A reader of such a
tensor reads it as a rectangle; its ranges or K share are in columns and `run.chain`
proves its rows are inside r0 to r1 - 1, as it does for a `km.past` read. Refused, each
at its line: roots writing different rows, a row outside them, the tensor's pages read
as pages, and `km.direct` of it (not built). Records of calls of one row are unchanged.

**At a pitch.** A call whose roots write block g at tiles base + g x pitch (a pitch
of at least the block's width, every block inside the tensor: the interior of a
zero-padded tensor, `y.tiles[k:k + 1, 1:J + 1]` with `k = self.rank + 1`) is described
in block coordinates: its record counts blocks of its width as any call's, and every
read of the tensor later in the chain is mapped to them, a padding tile needing no
mark. A read whose blocks go up by a step is one range; one that reaches the block
before its own (a halo's row r - 1) cannot start below block 0, so it takes the
covering range and waits for every block. Refused: `km.direct` of it (not built), and
a reader in groups of more than one. An interior writer of 8 ranks (late, by a spin
before its writes) then a 3-tap halo over rows ran exact on card 0 and on cards 0 and
1 with the halo over the edge; with no spin, 12.86 us as a chain against 11.64 us as
two commands on one card, 171.38 against 144.43 us on two.

**Several outputs.** A handler that writes more than one tensor, each as block g of its
group (the decode layer's x and its rows' squares, a tile a group), is a call with
several outputs: the first written in the handler's lines is the record's output, and
each other is an output of its own width (`kc.Call(..., outputs=(("sq", 1),))`). The
root writes every output's block before its mark, so a mark tells of block g of each.
The next call reads any of them: its input's marks prove what they cover, and when its
reads of another output need blocks they do not cover (every core reads all the
squares, its input only its own block), the record waits for every block of the call
before (one range of the whole output). A call further on reads another output marked
`km.past`, proven by that output's width. Two outputs given one tensor, and a direct
input from a call of several outputs, are refused
([test_class_chain_outputs.py](../../tests/core/dsl/chain/test_class_chain_outputs.py)).

**prefetch.** A read of a tensor that no call of the chain writes goes before the
record's input is marked: the kernels' wait before each read line takes the record's
mask of chain-written tensors (a word after the call's own). `run.chain(calls,
prefetch=False)` makes every read wait for the input's marks, and so does
`km.Stream(..., prefetch=False)` for that read. `prefetch=True` on a read states it: a
call that reads a chain-written tensor with it is refused. In a loop over streams in
step (`for x, w in zip(km.Stream(...), km.Stream(...))`), the wait is a stream's, not
the line's: each stream of a chain-written tensor is held at the move
(`km::Gated`, `km::chain_gate`) until the input is marked, while the other streams
fill their halves (a product's W is read while its X is awaited). A `prefetch=False`
stream in such a loop makes the whole line wait
([test_class_chain_gated.py](../../tests/core/dsl/chain/test_class_chain_gated.py)).

**km.direct.** `km.direct(t)` at a call marks an input sent by the call before's roots
into L1. The handler declares which of its parameters may arrive so,
`@km.handler(direct=("b",))`. The mark is checked against that declaration and against
the call before's output, which the call reads by a share of K. A chain build of a
class with such a declaration gets one landing area a core: buffer `kDirectArea`, the
size of the largest declared tensor given at the make. The area's bytes come out of the
L1 room before the class's buffers are sized, and a merged class (`km.merge`,
`class Layer(A, B)`) gets the area too. A class without a declaration generates the same
text as before.

In a record marked direct, the call before's roots do not write their block to DRAM.
Each root sends each write of its block, as it is written, to every reading core whose
share meets it, and marks that core after its own writes are acknowledged. A root may
write its block in pieces (`out.pages(first, 4)` after one product, `out.pages(first +
4, 4)` after the next): its first pieces are sent while it computes the last. The
reading core's stream of that parameter reads its area (`km::direct_or`, a local read
into its stream buffer) rather than DRAM. In any other record, the same kernels read
and write DRAM. One direct input is allowed a chain ("calls i and j (...) both take a
direct input: a core has one landing area, which a later call's roots could write while
an earlier reader still reads it (send one through DRAM)"). The block must be written by
`.write(...)` of its pages or tiles: a write stream (`km.Stream(..., write=True)`)
writes its blocks to DRAM inside the movers, so `run.chain` refuses it before a direct
input, naming the stream's line ("call k (name): its direct input t is written by call
j (name) through a write stream (line n), whose blocks go to DRAM, not to the next
cores; ...").

`@km.handler(direct=("b",), in_place=True)` lands the input in the halves of the
stream that reads it, with no landing area and no copy: the stream gets its own pair
(`kDirectIn`), each half the share's blocks it holds, so the rings are back at their
start at each record's end. The roots write each block where its half takes it, and in
a direct record the stream's movers push the blocks without reading. It is refused at
the make, by name, unless the parameter is read by one stream of its pages in order,
with tiles and repetitions (1) that are constants of the class, in whole blocks, and
it is the class's only direct input; `run.chain` refuses an earlier call of the same
handler in the chain.

Example 31's SwiGLU sends the hidden from up's 12 roots to down's 4 cores in this way
(`examples/31_swiglu/swiglu_direct.py`, README "## the hidden sent direct").

**The chain's stamps.** `chain_stamps=True` at the make (off by default, in the build
key; refused without the walk) builds the class's kernels with `KM_CHAIN_STAMPS`. In
each record, the movers and compute add the chain's own stamps (the record taken, its
input marked, the writes acknowledged, the marks issued: km_chain.h) and the class's
two: the overlay loaded (word 2) and the record's body done (word 3). Each record of a
chain takes its index as its stamp row (rows 0 to 7; later records stamp nothing).
`komorebi.dsl.chain.chain_stamps` reads and zeroes them:

```python
from komorebi.dsl import chain_stamps

tensor = chain_stamps.tensor_for(session, 16)  # a page a flat rank: first + count
rows = chain_stamps.read(session, run, program, tensor, (0, 16))
for record, terms in enumerate(chain_stamps.terms(rows, mhz, 2)):
    print(record, terms["input wait"], terms["body"], terms["hand-over"])
```

Each core writes the page of its flat rank, so the tensor holds the group's first +
count pages: over islands, island 1's group of two cards (118, 118) needs 236
(`tensor_for(session.island(1), 236)`, read with `run.island(1)`). A smaller tensor, or
an a0 that is no tensor of the session, is refused at submit by name.

Each term is the (median, largest) over the cores that ran the record, in us: the input
wait, the overlay load, the body, the hand-over (the writes acknowledged, or a direct
input's sends) and the marks.

**A `km.chain` method** holds calls of the class's handlers in order (`self.scale(...)`),
assignments, and `for _ in range(N)` of a constant N, unrolled. It is read when the class
is made, and called on a program (`program.steps(...)`) it gives the list of calls, which
`run.chain` takes. Its own parameters are the values its calls name; `km.past` and
`km.direct` mark them at a call as in the list.

On a session of several islands, `run.chain` takes a class's calls too, described with
island 0's program and `cores` in flat ranks.

**A gather.** `km.all_gather(copy, first, pages)` in a `km.chain` method, between two
calls, gathers the copy's pages over the islands as a record of its own: each island's
leader waits for its own share, sends and takes the shares, and marks its own island's
cores of the next call ([a gather in a chain](chain.md#a-gather-in-a-chain)). On one
island it is a plain record with the marks of any record. Only a class that states it
gets its case and `KM_CHAIN_GATHER` in its kernels.

| Refused | When | Text names |
|---|---|---|
| a handler line a chain cannot describe: a branch on data, a loop's index in a page, a write other than block g, a reduce other than a tree, a page another rank of the group reads, an inline body that takes a tensor and states neither `reads=` nor `writes=` | `run.chain` | the handler, the line and why; it still runs as a command |
| a body's `reads=` of a tensor a call of the chain other than the one before writes; a read of a tensor a body writes anywhere (`writes=[t]`) in a call other than the one before; a body that writes on mover A; a write anywhere in groups of more than one | `run.chain` | the call, the line, the tensor, the other call |
| a call of a program made without the walk | `run.chain` | the handler; `chain=True` or a `km.chain` method |
| a rectangle of rows not whole read from the call before's output or from a tensor a call of the chain writes | `run.chain` | the call, the line, the tensor |
| a `kc.Call` among a class's calls | `run.chain` | the call |
| a read of a tensor an earlier call wrote, not marked `km.past` | `run.chain` | the call, the tensor, the call that wrote it |
| `km.past` of the call before's output, or of a tensor the handler does not read | `run.chain` | the call, the tensor |
| `km.direct` of a parameter the handler does not declare, of a tensor other than the call before's output, of one not read by a share of K | `run.chain` | the call, the parameter, the declaration |
| two direct inputs in one chain; a direct input from an overlay handler's call, or written by a write stream; a later call of the chain reading a tensor sent direct (`km.past` of it too); a direct input whose tiles' bytes are not the landing area's | `run.chain` | the calls, the tensor |
| a direct input declared on an overlay handler; declared inputs of different tile bytes; a landing area larger than the L1 left for buffers | the make | the handler, the parameter, the bytes |
| `in_place=True` on a parameter not read by one stream of its pages in order, with tiles that are not a constant, repetitions over 1, a ragged last block, or beside another direct input | the make | the handler, the parameter, the stream |
| a call of an in-place handler earlier in the chain than its direct call | `run.chain` | both calls |
| a host read (`session.read`, `tensor.read`) of a tensor a submitted chain sent direct, before a write or a chain writes it | the read | the tensor and the call that sent it |
| `prefetch=True` on a read of a tensor a call of the chain writes | `run.chain` | the call, the line, the tensor |
| `prefetch=` other than `True` or `False`, or on a write stream | the make | the line |
| in a `km.chain` method: `if`, `while`, a call of something other than a handler of the class, a loop of a count the build cannot know | the make | the method, the line |
| a class with the walk that also has a dynamic handler | the make | the class |
| `km.all_gather`: a root writing another island's share, a later read outside the gathered pages, a second gather of the copy, a tensor not a copy, over 2,047 pages from page 4,095 at most over islands (on one island, more ranks of 255 pages than the chain's) or 127 ranks, a direct input after it, or a listed gather in a class whose methods state none | `run.chain` | the call, the tensor, the pages ([chain](chain.md#a-gather-in-a-chain)) |

## The runtime's form

`Class(..., form="runtime")` builds the class's handlers, their lines unchanged, as a
program in the runtime's form ([host](host.md#several-programs-in-one-run)): the
runtime owns the kernels and the command loop, and the class's code is the program's
parts, which the loop calls for a command of that program. Such a program loads
alone, in a list or in a dict beside other programs of that form (written by hand or as
classes), and a group switches to it (`run.load`, or a thread's `kmx_load`). From the
dense experiment's Linear and example 22's Gate:

```python
linear = Linear(form="runtime", plan={"OH": 1}, session=session, **tensors)
gate = Gate(form="runtime", BLOCK=1, beside=linear, a=a, b=b, out=out, log=print)
run = session.load({linear: (0, 112), gate: (112, 8)})
run.wait(run.load(gate, group=(0, 112)))  # cores 0 to 111 now run the gate
```

How the class runs in this form:

- **The movement lines.** `movement.cpp` holds the handler switch as one function over
  the mover's runtime, used as the kernel form's `km::Movement` (as an overlay handler's
  part uses it). The reader's `km_program` (mover A) and the writer's (mover B) call it,
  each built from that file. A line lowers to the same code in both forms.
- **Compute.** `compute.cpp`'s switch is compute's `km_program`. The runtime runs the
  hardware start-up on the program's first buffer below 16 and its first from 16 when a
  group loads it. Each command first sets those two buffers' formats again
  (`reconfig_data_format`, `pack_reconfig_data_format`): in a list, the core's last
  command may have been another program's
  ([hazards](hazards.md#formats-another-program-left-on-the-same-cores)).
- **Buffers.** The class's buffers are the program's (the same indices, formats and
  pages). A dict's load sets them up for the group; a list holds every program's in one
  layout.
- **State.** A part keeps nothing from one call to the next (its `.bss` is zeroed at
  every call), and a class keeps nothing there: a field is a buffer, whose pages stay in
  L1 between commands; a stream's place is its buffer's pointers, which the runtime's
  kernels keep; everything else is a handler's local.
- **Arguments.** The movers' compile arguments (each tensor's accessor) are the reader's
  and the writer's parts'. A rectangle's per-core arguments are given at the program's
  place among the run's ([host](host.md#several-programs-in-one-run)).
- **Overlay handlers** are the program's own overlays, numbered after those of the
  programs before it ([overlays](overlays.md#several-programs-with-overlays)) and built
  with the program's compile arguments.

`beside=` is for a list: its programs' buffers are in one layout, with indices disjoint,
and a class picks its indices from 0, 8 and 16 as if alone. Made beside programs, it
leaves their indices and takes their `unpack_to_dest_mode` at those indices, so the run
has one set of compute options. The fidelity is the programs' to agree on: a class
whose tile functions have no product gives `compute={"math_fidelity": ...}`.

**A start-up of the class's own** (a body at a kernel's start, `km.cxx(...,
at="start")`, and `startup=`) runs at the load of the program: every part of a program
is called with the load command (`KM_HANDLER_LOAD`) when a group takes it, and the
class's parts run there what the kernel form runs before the commands, compute's
start-up (the class's `km.Startup`, else the one on the program's first buffers) and
then the bodies (`km_program_start` in the generated sources). In a dict's run it runs
when the group takes the program, again at each `run.load` of it. A class alone (or
alone in a list) is run as a dict over every core, so its run loads it before the first
command (buffer indices below 32, as any dict's). With `startup=`, each command's two
format calls are left out: the class's start-up set the formats, on cores that hold
the program alone. The names a start body declares are not seen by the handlers here
(the body is the load's function): a value a handler needs goes in an own field. On
the card (card 0, 960 tiles, x + 0.5 from a 0.5 tile built into an own field at
compute's start): the kernel form 18.12 us, the class with `startup=` alone in the
runtime's form 18.94 us, every word equal to the kernel form's; a dict of the class
(cores 0-59) and the one with `startup=` (60-119) 12.2 and 11.88 us, 0 wrong.

Refused, `ProgramError`:

- **A start-up of the class's own in a list of several programs**, at the load: every
  core holds every program and none is loaded, and one program's start-up would set
  the hardware up under the others' commands; a dict gives each program its cores.
  Sending a load of its own to every core at the run's start is not built for it.
- **km.merge of such a class**: a merged program has one load, the sources' start-ups
  are not merged.
- **A body at a kernel's end** (`at="end"`), when the class is made in this form,
  naming its line: no part is called when the run stops. It would need the runtime's
  three kernels to load and call the held program's part once after the stop, with a
  give-up path of its own; not built.
- **A dynamic handler**, naming its line: the runtime's form has no message loop of a
  program's own; it would need the runtime's message kernels to serve a program field,
  and two programs' dynamic handlers in one run are not built either
  ([overlays](overlays.md#not-in-this-version)).

An overlay handler is not refused: it is one of the program's own overlays (above).

A class made in the kernel form and given to a list or a dict is refused at the load
(`ProgramError`) with `form="runtime"` named, and with what that form would refuse in
the class when it uses one of the above, so it is not rebuilt only to be refused
again.

On the card (card 0, AICLK 1,350 MHz), each figure a command's time on the events,
the median of 5:

| Class | Kernel form | The runtime's form |
|---|---|---|
| [22 gate](../../examples/22_gate/as_class/README.md), 12 cores, 64 tiles a core (AAAB) | 81.14 us | 81.78 us; every word of the output equal |
| [linear layer](../../experiments/language/dense_class/README.md), the K split, 32 x 4096 x 4096 | 94.20 us (a kernel buffer of 94,208 B) | 95.58 us; Y's words equal (exact on integers; 4.38e-3 relative RMS on normal values, both); the first command 114.6 us (it loads the five parts, 71 KB) |
| kernel text | gate 38,080 B, Linear 92,304 B (mock) | the runtime's kernels 30,416 B (gate) and 30,592 B (Linear); parts, reader / writer / unpack / math / pack: gate 4,928 / 4,992 / 4,032 / 2,176 / 2,048 B, Linear 18,240 / 20,096 / 13,824 / 8,704 / 10,176 B |

The runtime's form costs 0.64 us a command of the gate and 1.38 us of Linear. A held
program's `load()` is part of it (113 cycles a command in
[example 16](../../examples/16_two_programs/README.md), 0.08 us); the rest is the parts'
code, called through a part's entry instead of inlined into the kernel's loop, and
not split further by measurement. These two rows were measured before each command set
its start-up's formats again; with those two calls a command, a later run gave 81.64 us
and 95.16 us: no cost above the noise, so every command makes them.

Two classes in one run, both in this form (the Linear above with one row tile a block,
the Gate at Linear's fidelity with blocks of one tile):

- **A dict**, Linear on cores 0 to 111 and the Gate on 112 to 119: the split exact
  (96.2 us), the gate 0 wrong (23.12 us); `run.load(gate, group=(0, 112))` 10.64 us
  from the host's submit to its event, then the gate on those 112 cores 0 wrong
  (47.9 us); `run.load(linear, ...)` back 4.68 us, and the split exact again (95.5 us).
  Example 16 measures a load on the cores: 3,529 to 3,791 cycles (2.6 to 2.8 us).
- **A list**, every core both: the split exact (113.4 us, its first command), then the
  gate on 120 cores 0 wrong (52.1 us). Before each command set its formats again, the
  gate's output after the split was wrong
  ([hazards](hazards.md#formats-another-program-left-on-the-same-cores)).

[12 overlays](../../examples/12_overlays/as_class/README.md)' class in this form (its two
handlers are overlays): 40 commands, every word as the example checks it; mix on mover
B, held, 116 / 982 cycles (load, call), scale on mover B 132 / 2,502.

## Refusals

A line the compiler does not take raises `Refused` (a `ProgramError`) when the class is
made (or compiled on the host alone, without a session:
`komorebi.dsl.compiler.compile_class(cls, plan, tensors, l1=None, constants=None,
handlers=None)`, which returns the compiled class: its handlers and buffers), naming the
handler, the line and what to write instead (the classes of
[test_classes.py](../../tests/core/dsl/test_classes.py)):

```text
Sliced.copy, line 34, `km.Stream(source[1:4]).drain()`: a tensor is in DRAM: a slice is not a value; take a part: source[self.partition(...)], or use a stream or read()
Unled.merge, line 44, `output.pages(0, 1).write(whole)`: whole is the reduce's sum, which only the leader holds; write it under `if self.rank == self.leader:`
Unsolved.count, line 49, `count: km.L1[km.fp32, TILES, 32, 32]`: TILES is solved against the L1 the buffers leave; make the program with session=session (or l1=session.l1()[1])
Product.multiply, line 93, `results.push(product(x, y))`: product: a tile function takes tiles of float32 or the B family (float32, bfloat16, bfp8_b, bfp4_b, bfp2_b), not float16, float16: its code runs on an fp32 Dst, which the card fills wrong from the A family; give it tiles of those
OtherFunction.apply, line 121, `acc += scaled(block, self.weight)`: acc += f(...) adds a product, and the tile function scaled (line 103 of test_classes.py) is not `return a @ b`; write scaled as `return a @ b`, a the stream's block, b a field
Unfilled.apply, line 131, `acc += project(block, self.weight)  # unfilled`: self.weight is not filled above in this handler, and no other handler of the class fills it; write self.weight.fill(km.Stream(...)) before, or in a handler of its own
Narrow.apply, line 142, `acc += project(block, self.weight)  # narrow`: self.weight is K rows of 2 tiles and acc holds 1; declare self.weight: km.L1[km.bf16, K, 1, 32, 32] or make acc km.tile.zeros(km.fp32, 2)
Mixed.apply, line 154, `acc += project(block, self.weight)  # mixed`: this loop has other tile lines, and a product holds the matrix unit's setup for the loop; keep the loop to the product
Stepped.copy, line 213, `km.Stream(x.tiles[0:4, 0:4:2]).drain()  # stepped`: a rectangle of tiles is two slices, rows then columns, each with its start and end, a step on the rows alone; write x.tiles[r0:r1, c0:c1] or [r0:r1:s, c0:c1]
Ragged.copy, line 210, `km.Stream(x.tiles[0:2, 0:n]).drain()  # ragged`: a stream's blocks are whole rows of its rectangle, and this rectangle's columns are a word; stream x.tiles[r0:r1, c:c + n], n a constant, or one row of it
Uneven.copy, line 216, `km.Stream(x.tiles[0:4, 0:3], block=4).drain()  # uneven`: a block of a rectangle of 3 columns is whole rows of it; give block= a multiple of 3
Owned.apply, line 228, `whole = km.reduce(acc, form="owner")  # owned`: a reduce of 2 tiles by owner holds a slot of them for each other core of the group; reduce them by tree: form="tree"
Tall.apply, line 242, `y.tiles[0:2, 0:1].write(acc)  # tall`: a page write of a rectangle is one row of it; write y.tiles[r:r + 1, c0:c1].write(...)
Vector.apply, line 254, `results.push(self.gated(x, y))  # vector`: fidelity= is a product's (the matrix unit's), and gated is the vector unit's; drop fidelity= from gated
Fidelities.apply, line 267, `acc += coarse(block, self.weight)  # fidelities`: coarse's fidelity is HiFi2 and another product's HiFi4: the fidelity is the compute kernel's; give the class's products one fidelity
Unpacked.copy, line 459, `km.Stream(items.items(0, 4, size=1000)).drain()`: items of 1000 B in pages of 4096 B: a page holds a whole number of items; give a size that divides 4096
Untaken.use, line 467, `km.cxx("use(table);", takes=[self.table], on="pack")`: self.table is not filled above in this handler (or a body took it), and no other handler keeps it filled; write self.table.fill(km.Stream(...)) before this body, or in a handler of its own
Outside.copy, line 473, `first, count, more = items.pages(0, 1).read()[0:2]`: words 0 to 1 of a page of 1024 words for 3 names; give a slice of the page with a name each word
Unknown, line 483: compute option 'dst_full_sync': the host API has no such option; give math_fidelity or one of fp32_dest_acc_en, dst_full_sync_en, bfp8_pack_precise, math_approx_mode, enable_trisc2_rvv
Unproduct.apply, line 583, `for h, g in zip(km.Stream(a.tiles[0:1, 0:2], block=1), km.Stream(w.tiles[0:2, 0:2], block=2)):`: streams in step with blocks of different sizes are a product's operands: a tile of one is a row of the other; keep the loop to `acc += self.f(a, b)`, f `return a @ b`
Unblocked.apply, line 596, `acc += coarse(xb, wb)  # unblocked`: acc is 2 tiles wide: each block of the second stream is the first's 1 K tiles by 2 columns (2 tiles), and it is 3; stream w.tiles[k0:k1, c:c + 2] with block=2 beside x's block=1
Swapped.apply, line 607, `acc += coarse(xb, wb)  # swapped`: the loop counts the first stream's tiles, and the product's first operand is not the first stream's block; zip x's stream first, then w's
Unsummed.apply, line 616, `whole = km.reduce(self.sums, form="tree", group=2)  # unsummed`: self.sums holds no sums above in this handler; add products into it first: self.sums += self.f(a, b)
Owners.apply, line 630, `whole = km.reduce(self.sums, form="owner")  # owners`: a reduce of self.sums by owner holds a slot of its 2 tiles for each other core of the group, and the group's size is the command's; give group=S, or reduce by tree: form="tree"
Unbounded.apply, line 640, `arrived_blocks = down.cast(x.tiles[0:2, 0:n].read())  # unbounded`: a cast's slot holds the rectangle's most rows and columns, and the build knows no most for a side here; write each side r:r + m or r:min(r + m, e), m a constant
Overwide.apply, line 656, `y.tiles[0:1, 0 : min(3, self.rank)].write(self.part)  # overwide`: self.part is 1 x 2 tiles; write a rectangle of them (a side a word: the build must know it is at most the field's); write y.tiles[r:r + 1, c:c + 2].write(self.part), or c:min(c + 2, e)
Unfillable.load, line 664, `self.weight.fill(km.Stream(w.tiles[0:2, 0:n]))  # unfillable`: the rectangle's columns are a word and self.weight's rows are 2 tiles: the build must know they are at most 2; write w.tiles[r0:r1, c:min(c + 2, e)]
Lone.copy, line 670, `km.Stream(x.pages(0, min(2))).drain()  # lone`: min(...) of level 1 takes two ints or more; write min(a, b)
```

A once-only object is used in the loops it was made in. A tile of DST (`km.tile.zeros`),
a reduce's sum, a block a broadcast cast, a field a product frees and a drained stream
are each made once by a line of the handler and freed once by another; a loop around
one of the two lines and not the other runs it a turn more than its partner, so the
class is refused at the consuming line, naming the line that made the object:

| Made | Used | Refused at | Right forms |
|---|---|---|---|
| `total = km.tile.zeros(...)` outside a range loop | `self.f.write(total)` in it | the write: "total was made at line N, outside the loop this write stands in" | make the tile in the loop; or add into it in the loop and write after it |
| a running total (below) | its write or reduce under an `if` or in an inner loop of the range loop | the line: "total is a running total (made at line N, outside this loop): it is written or reduced once a turn, as a line of the range loop's own body" | move the line into the range loop's body |
| a reduce's sum, outside a loop | written in a loop off the leader's `if`, or a field's sum written in a loop | the write or the body: "whole was made at line N, outside the loop this write stands in" | write it under the leader's `if` (a sum held across turns, below); or reduce and write both outside |
| `self.w.fill(...)` in a range loop | a product (`acc += f(block, self.w)`) after that loop | the product: "self.w is filled in a loop (line N) and a product frees it once" | fill before the loops, once; or read it in the loop that fills it (a field filled each turn, below) |
| a drained stream (`km.Stream(self.out, mover=...)`, its `with`) | its `end()` or its `for page in out:` at another depth | the end or the drain: "out was made at line N in 1 loop and its drain stands in no loop" | make, end and drain it in the same loop, or all three outside; for every turn's pages drained once, make it before the loop, its body lines in the loop, its end and drain after |

**A running total.** A tile made by `km.tile.zeros` before a range loop and written
(`pages.write(total)`) or reduced (`km.reduce(total)`) as a line of the loop's body is
the sum so far, a page a turn: turn i writes the sum of turns 0 to i.

```python
total = km.tile.zeros(km.fp32)
for i in range(turns):
    for tile in km.Stream(x.pages((self.rank * turns + i) * 2, 2)):
        total += tile
    out.pages(self.rank * turns + i, 1).write(total)  # the prefix total
```

At the write compute packs the tile's pages into the write's buffer and into a carry
buffer of the tile's width in fp32 (`kCarryTotal`, listed with the class's buffers:
4,096 B a tile), and as the turn's last line reads the carry back into the same DST
tiles, unchanged. A turn so costs one pack and one copy of the tile through the carry;
the movers' lines are those of any page write. A reduce of a running total carries the
tile before the reduce and reads the own total back at the turn's end, so the reduce's
sums never enter it. The total is in DST after the loop, written or reduced there as
any tile, or released at the handler's end. The tests are
[test_class_depth_mismatch.py](../../tests/core/dsl/compile/handler/test_class_depth_mismatch.py); on the
card ([loop forms](../../experiments/language/loop_forms/README.md)).

**A scan's state.** A tile made before a stream loop (`h = km.tile.zeros(km.fp32)`, or
`h = self.s`, s a float32 field of one tile filled above: its tile copied into DST and
its page freed) and stepped in the loop by a tile function that takes it, `h = f(x_t,
h)`, is the loop's state, a tile a token:

```python
h = km.tile.zeros(km.fp32)
results = km.Stream(y.pages(0, tokens), write=True)
for a, u in zip(km.Stream(xa.pages(0, tokens)), km.Stream(xu.pages(0, tokens))):
    h = gated(a, h, u)  # gated(a, h, u): return a * h + u
    results.push(h)  # each token's state
hs.pages(0, 1).write(h)  # the last, for the next command
```

Before the loop's tiles compute packs h into an fp32 carry buffer of one tile
(`kCarryH`, 4,096 B); each tile's call reads it back from the carry into the DST tile f
takes it in, beside the token's tiles, and packs f's result into the write stream (at
`results.push(h)`) and into the carry again, so a step costs one pack and one copy of
the state through the carry, exact in fp32, and the calls are one a tile (not batched:
each reads the last one's result). After the loop the carry is read back into DST 0,
the tile h names there, written as any tile (`hs.pages(0, 1).write(h)`,
`self.s.write(h)`) or released at the handler's end. A state started from a field
continues a scan across commands: the second of two commands of 32 tokens, its state
filled from the page the first wrote, gives the states of one command of 64 lane for
lane on the card ([tile scan](../../experiments/language/tile_scan/README.md): a tile a
token, `a * h + u` over 64 tokens within 6.6e-8 of float64 on float32 tiles and 8.8e-8
on bf16 ones, against the largest state). Refused at
the line: h read before it is set ("h is read before it is set"), a state of more than
one tile or passed twice, a field among f's operands, f with a matrix product or several
results, and a field of another format than float32 or of more than one tile as the
start ("a scan's state is carried in float32"). Instead of a bf16 state field: keep the
field float32 (4,096 B a tile) and push the states into a bfloat16 tensor where later
lines want them in bf16 (the push packs each state from fp32; built, its values not
measured on the card). A step's time through the carry is not measured.

**A block held across turns.** A block a broadcast casts once and a range loop the cast
does not stand in takes (a weight cast once and read each turn) is held: compute waits
for it once, at the cast; every take in the loop (`for tile in ws:`, a body that takes
it, a product `acc += f(xb, ws)`, two or more of them a turn) reads it in place; and it
is freed at the end of the cast's lines, after the loop. The movers cast one block, as
for any cast outside a loop: the cast's mover waits at its end for compute to free the
block, and when a stream or a page write on that mover follows the cast, that wait
moves to the handler's end, after the mover's last line. A second cast on that mover
is refused, naming the block whose finish moved.

```python
column = km.Broadcast(self, along="column", window=1, mover="A")
if column.sender():
    ws = column.cast(w.tiles[0:8, 0:4].read())
else:
    ws = column.arrive()
for i in range(turns):
    r = (self.y * 2 + self.x) * turns + i  # the core's place in a rectangle 2 wide
    acc = km.tile.zeros(km.fp32, 4)
    for xb in km.Stream(x.tiles[r : r + 1, 0:8], block=2, pattern="B"):
        acc += project(xb, ws)
    y.pages(r * 4, 4).write(acc)
```

The tests are [test_class_held_block.py](../../tests/core/dsl/compile/casts/test_class_held_block.py); on
the card, every word equal to the same rows with the block cast each turn
([loop forms](../../experiments/language/loop_forms/README.md)).

**A sum held across turns.** A reduce's sum made before a range loop and written in it
under the leader's `if` (`out.pages(i, 1).write(whole)`, the same sum on page i each
turn), or taken there by a body (`km.cxx(..., takes=[whole])`), is held. Each turn
waits for it without freeing it, and the leader frees it once, after the loop. A turn's
write copies the sum's tiles (fp32, exact) into DST and packs them into the write's own
pages, which mover B writes and frees as any page write. One write line or one take
line of the sum a loop. A field's sum (`km.reduce(self.sums, ...)`) written in a loop
is refused as before.

```python
whole = km.reduce(total, form="tree")
for i in range(2):
    if self.rank == self.leader:
        out.pages(i, 1).write(whole)  # the sum beside each row block
```

**A field filled each turn.** A field filled by a line of a range loop's body
(`self.w.fill(km.Stream(...))`) and read by a product in that loop (turn i's product
of weight i) is freed at the end of the turn, after the turn's last product. Compute
waits for each turn's fill, and mover A's next fill reserves the room the turn freed.
A product of the field after the loop is still refused: it would read one fill of
several. A field another handler reads stays in L1 between commands, so its fill in a
loop is refused as before.

```python
for i in range(turns):
    self.weight.fill(km.Stream(w.pages(i * 4, 4)))  # weight i
    acc = km.tile.zeros(km.fp32, 2)
    for block in km.Stream(x.pages(i * 2, 2), block=2):
        acc += project(block, self.weight)
    y.pages(i * 2, 2).write(acc)
```

A tile made inside a loop and written after it is refused the same way ("inside a loop
that has ended"). Accumulating into a tile made before a range loop and writing it
after the loop, a tile made in the loop that writes it, a cast in the loop that takes
its block, and a field filled once before the loops are the right forms and build as
before.

A chain stated with a class's handler calls refuses a broadcast's handler: its cast reads
a tensor under `if b.sender():`, and `Run.chain` raises `ChainRefused` for a read under an
if on the command's words ("a read or write under an if on the command's words", from
`komorebi.dsl.chain.chain_calls`), so no such chain holds a core the broadcast's rectangle
leaves out. The tests are
[test_class_depth_mismatch.py](../../tests/core/dsl/compile/handler/test_class_depth_mismatch.py).

A range of items where a line moves whole pages (a fill, a page write, a word read, a
partition, a cast) is refused at its line, and so is `compute={"unpack_to_dest_mode":
...}` (the class sets it from its float32 buffers).

`Run.submit` refuses a command of a reduce in groups of S whose cores are not a
multiple of S, naming the reduce's line.

On islands a group of `km.reduce` or `km.all_reduce` may span them: each island's part
is folded as its own group, then each part's first rank sends once to the group's rank
0 ([islands](islands.md#classes-on-islands)). Its parts are added in another order than
one island's tree, so an integer or exact combine gives the same words and a float sum
may differ by rounding (the decode layer on two cards with groups of 15 across the
edge: 996 of 65,536 output words, the largest 0.0065 relative; an example, not a
bound).

Streaming tiles of the two exponent families into a product is refused the same way:
"a product of float16 and bfloat16: its operands are of one exponent family (bfloat16
and bfp*_b, or float16 and bfp*)"; a float32 operand outside two streams in step, "a
product's left operand is float32: ...".

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
| [03 broadcast](../../examples/03_broadcast/as_class/README.md) | the same: 960 ranks, all exact (level 1a) | 48,720 / 53,408 B (mock) | 109.0 / 99.8 us a job, exclusive (level 1a) |
| [05 forward](../../examples/05_forward/as_class/README.md) | the same: exact in 3 of 3 runs, 10^6 forwards | 41,600 / 43,040 B | 12,162 / 12,297 us a run (median of 3) |
| [07 work cluster](../../examples/07_work_cluster/as_class/README.md) | the same: 204 rounds, sum 1,004,917, every state word | 21,200 / 21,424 B (mock) | the thread's loop 10.55 / 10.75 ms |
| [11 movement](../../examples/11_movement/as_class/README.md) | the same: every copy exact (level 1a) | 38,288 / 36,720 B | READ AAAB 308.74 / 310.44 GB/s (level 1a) |
| [12 overlays](../../examples/12_overlays/as_class/README.md) | the same: every mix word and scaled value | 33,296 / 28,016 B (mock), and the parts above | a turn from the host, one command each: scale 1,353 / 1,731 us, mix 1,940 / 2,095 us |
| [22 gate](../../examples/22_gate/as_class/README.md) | the hand program's bound: 0 wrong of 786,432; float32 against float64: 2.46e-7 (tolerance 1e-6) | 37,632 / 38,640 B (mock) | 83.70 / 76.24 us a command (bfloat16, 12 cores, 64 tiles a core, 2 tiles an acquire; one tile an acquire and the first body 86.68 / 76.06); the emitted body 1,597 cycles a tile on the math CPU, 1,597.6 printed (the first body 1,656, 1,657.6 printed; the hand body 1,901.5 in the one-core harness) |
| [14 message handlers](../../examples/14_message_handlers/as_class/README.md) | the same: exact in 3 of 3 runs, 1,013,360 messages of three handlers | 42,640 / 43,568 B (mock) | 10,987 / 11,149 us a run (median of 3) |
| [linear layer](../../experiments/language/dense_class/README.md), 32 x 4096 x 4096, the K split | Y equal to the host's product (integers); 4.38e-3 relative RMS (normal bfloat16, HiFi2) | 90,192 / 61,584 B (mock) | 94.2 / 92.4 us a command (112 cores, the hand's groups of 14 and blocks of 16) |
| [linear layer](../../experiments/language/dense_class/README.md), 1024 x 4096 x 4096, the grid form | the same | 90,192 / 69,296 B (mock) | 358.6 / 382.4 us (12 x 8 cores, as the hand's) |
| [tile rules](../../experiments/language/tile_rules/as_class/README.md): add and scale, load macro | float32 against float64: 5.96e-8 and 5.95e-8 (tolerance 2^-23) | 47,520 B (two stamped handlers) | the emitted bodies 91.0 and 59.0 cycles a tile on the math CPU, 91.6 and 59.5 printed; 93 and 62 in the one-core harness |
| [ray tracer's trace pass](../../experiments/language/raytrace_class/README.md) (TRACE, tiny scene, the tree in L1) | the hand program's hit pages word for word: 0 of 18,874,368 words differ (the cycles vector apart), the walk's counts equal on every core; against numpy's trace 658,018 hits, none differs | 48,240 / 85,808 B (mock) | 3,294.9 / 3,259.9 us a command (120 cores, 32,768 packets; the tree loaded once, 23.7 us) |
| [rasterizer's fill](../../experiments/language/raster_class/README.md) (FILL, tiny scene, the empty tile's planes in L1) | the hand program's frame word for word: 0 of 5,242,880 words differ; the counts equal on every core; the image exact | 44,000 / 32,896 B | 88.34 / 118.90 us a command (120 cores; the planes loaded once, 13.58 us) |
| [rasterizer's passes](../../experiments/language/raster_class/README.md) (COUNT, HIST, SETUP, tiny scene) | the hand program's counters, histograms and lists word for word (0 of 245,760, 491,520 and 182,272 words differ) | 60,608 / 55,424 B | 85.34 / 87.16, 23.96 / 24.18, 112.92 / 124.34 us, one command each |
| [tile rules](../../experiments/language/tile_rules/as_class/README.md): the row norm, row form | float32 against float64: 2.33e-7 (tolerance 1e-5) | 41,248 B (stamped) | the emitted body 687.0 cycles a row of 2 tiles on the math CPU, 677.0 printed, 689.5 in the one-core harness (the first body 771.0, 753.0 printed); the hand norm 3,201.5 |
| [tile rules](../../experiments/language/tile_rules/as_class/README.md): silu(a @ w1) (a @ w3), two products in the body | float32 against float64: 5.64e-4 (tolerance 2^-7) | 40,800 B (stamped) | the emitted body 1,706 cycles a tile on the math CPU, 1,678.6 printed, BODY 29's 1,707 in the one-core harness; 92.96 us a command (12 cores, 64 tiles a core) |

## Not in this version

Each of these is a line a user would write that this version refuses, with what it
would take:

- A rectangle per command (`run.submit(h, rect=...)`): the rectangle's words in the
  command, the index arithmetic at run time, and every table now sized by the
  rectangle at the build sized at run time (more than 200 lines).
- Slices and `.columns` of a field (`y.tiles[...].write(self.sums[:, 0:2])`): an
  offset into the field at each place that uses one (its pack, its write, its unpack),
  about 100 lines.
- A rectangle of tiles with a step on its columns (`x.tiles[0:2, 0:8:2]`): a column
  stride in the movement's rectangle walk (`km::Walk`) and in the compiler, about 80
  lines.
- A transposed rectangle outside a cast (`km.Stream(w.tiles[...].T)`): the cast's
  transposed walk moved into the stream's movement, about 80 lines.
- A stream of a rectangle of more than one row whose columns are words: the
  rectangle's row walk with run-time columns, about 80 lines.
- A field filled from a broadcast (`self.w.fill(down.arrive())`): the arrived block
  copied into the field's buffer, in step with the broadcast's window, about 50
  lines.
- A product's operands other than a stream's tile and a field, two streams' blocks,
  or two broadcasts' blocks into a float32 field (a stream's tile with a broadcast's
  block): a fifth product form, about 100 lines.
- A field of sums of more than one row from two streams: the field product's
  subblock over rows in the streams' block layout, about 80 lines.
- The fidelity derived from a tile function's tolerance: a model of the error by the
  data, which the build does not have; it stays `fidelity=`.
- A word field with a rectangle (`self.x`, a broadcast): the rectangle's words in the
  word table, about 60 lines.
- Two reduces of one handler of another form or group size (`km.reduce(a, group=2)`
  then `km.reduce(b, form="owner", group=2)`): refused at the second's line. A
  sender waits for its receiver's credit from a group's second turn, and the turn
  restarts when the form or the group changes (`reduce_turn_next`,
  `device/movement/km_port.hpp`), so the second reduce's senders would not wait: a
  turn kept by the handler's reduce count on both sides of each port, a device
  change, not built. Reduces of one form and group (or two handlers in a chain) are
  the form to write.
- An integer tile type in a tile function (`a & b`, `a + b` of uint32 tiles): a new
  type through the unpack, `Dst` and the pack, several days. Today: the handler's own
  int words (`& | ^ << >> % min max`), or a hand body (`km.cxx(..., on="pack")` with
  sfpi integers, or a mover's body over a page of words).
- A tile function of more than 4 operand tiles (a 6-input bilinear filter): more
  `Dst` tiles a call, device-side planning, several days. Today: split it over scratch
  tensors (two calls, one more DRAM round trip a part), pass the weights as float
  words, or a hand body `km.cxx(..., gives=km.Tile)`.
- A field's rows under `km.UPTO` (a held texture or weight whose rows change by
  command): a run-time L1 layout. Today: one handler set a size (`handlers={"load":
  [{"ROWS": 4}, {"ROWS": 8}]}`), or the field sized to the most.
- A DRAM read inside a dynamic handler's loop ([dynamic](dynamic.md)): a reader that
  waits for a message's page, several days with device work. Today: keep what the
  loop reads in L1, or bin the messages in the dynamic run and read the pages in a
  plain handler between runs.

A product and a tile function meet in one handler as a product's sums and the tile
function written of them (`y.pages(first, n).write(act(acc))` of DST sums,
`y.tiles[...].write(act(self.part))` of a field of sums: [Products](#products)); a
traced tile function's `act(a @ w)` multiplies one tile a side. Of a
class's X280 thread: a thread in a merged class's body, shared regions, a region's
size checked against the pages a handler reads of it, chain records in its header
(see [X280 threads](#x280-threads)).

A page assembled
from words of three CPUs (the ray tracer's profile page: the walk's counters from pack,
the reader's from mover A, the writer's from mover B, written by mover B): it would
need a mover-to-mover hand-over and a body on mover B over a write stream's page before
the movers write it, more than 40 lines of the compiler; the class's profile page holds
the pack CPU's words. Items of a tensor on two islands (refused when the kernel is
built: no call moves part of another island's page). A range loop whose count is one
mover's word (PLACE's headers, `item_count` of a mover B body): its lines are a body.

A class in the runtime's form with a dynamic handler or a body at a kernel's end:
refused ([The runtime's form](#the-runtimes-form)); the runtime's form has a message
loop of no program, and no program's part is called when the run stops. A body at a
kernel's start or `startup=` in a list of several programs: refused (no load there).
