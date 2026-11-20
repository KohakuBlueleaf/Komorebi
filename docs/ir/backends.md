# Backends

A backend turns a checked module into something a machine runs. Each backend says which parts of
a [byte-code file](bytecode.md) its output depends on: the program and its hints (S and H), or
also the positions and names (P), as a backend must whose text carries a script's names and lines.
Its cache key covers exactly those parts: the build hash, or the full hash.

## What a backend states

On a target (an architecture and its features) a backend states, before it sees a program:

- the abilities it provides beyond the base: `group = grid`, `dynamic`, islands, local state;
- the external implementations it has, by name;
- its limits on a program's text: handlers in a program, parameters or argument words of a
  handler, run or chain, fields of a message;
- the formats and the tile shapes it computes, and this page's section on how;
- the hints it reads, and its guarantees of progress.

On a machine (workers, worker memory, islands, and the load's named options) it states what
depends on that machine and on what was built: how many workers and islands, how many bytes of
state a worker holds, how many invocations may be in flight. A machine's islands are a graph
(`types::IslandGraph`, the reference interpreter's too): the workers each island holds, in order,
and the connections between islands, each both ways. No connections given means every pair is
connected; an empty list of several islands is not connected. Nothing names a shape or a count.
A graph that is not one of the machine's workers is refused at load with `K966`, with the same
text from a backend's load and the interpreter's. The order is: compile, build, this statement,
then the check at load.

A program that asks for more is refused by name at the construct that asks, before anything is
emitted (`K960` to `K966`, [the checks](checks.md)). A backend never substitutes other behaviour.

Its output is files, the entries an invocation names, and its build, as data: steps to run in the
output's directory, or kernels the machine's runtime compiles at load (a kernel's source, the
processor it runs on, defines, compile arguments, a tensor's layout among them filled at load,
and options, each a bool or a text) with the buffers in each core's memory they pass pages
through (an index, a format, the bytes of a page, the pages, and the other half's index when the
buffer is one half of a pair, one half for each mover). Each entry carries the invocation's ABI, the same on
every backend: its number, its argument words in order (a tensor's address, a tensor's tile
count, a parameter's value), and the bounds a host checks before sending it: tensor tiles whose
first and count are of the words, numbers, the worker's rank and the group's size, by `+ - *`,
checked for each rank of the group when they read the rank. Compiling runs nothing and writes nothing. The
output's files are kept under a key over the hash of the parts the backend reads, the backend's
name and version, and the target; what a build makes of them is the build's own cache.

## The CPU backend

Portable C++17 on a small framework the crate carries, one thread for each worker. Its output
depends on the program and its hints only; its key covers the build hash. Its build is steps; an
entry's number is its index, a tensor is one argument word, and it has no bounds: the framework
checks a view itself. It provides no
ability beyond the base, has no external implementation, and has no limit on a program's text.
One invocation is in flight at a time.

At this stage it emits what the first lesson uses: a handler on a group over tensors and scalars,
views of tiles, a partition by rank, streams and sinks of blocks of one tile, and functions on
tiles (a function on blocks is refused). Everything else is refused with `K964`, by name.

The output builds into `module.so`, the module behind a C interface (`komo_cpu.h`), and `driver`,
which runs a [run script](run-scripts.md) on the module and prints the same report as
`komo run`. The build runs `$CXX` (else `c++`) and keeps its results in a cache of two levels:

- the sources, under the output's key;
- what the compiler made, under that key, the compiler's `--version` line and its flags.

A different compiler therefore rebuilds only the second level. The cache is `KOMO_IR_CACHE`,
else `$XDG_CACHE_HOME/komorebi-ir`, else `~/.cache/komorebi-ir`.

The tests that build an output fail without a compiler; `KOMO_IR_NO_CXX=1` skips them. The parity
test runs every script in `tests/ir/parity/` on the interpreter and on the driver, and compares the
two reports by the rules of [run scripts](run-scripts.md#between-backends).

### How it computes

A format is a notation; this is how the CPU backend holds and computes each one it states.

| Format | Held as | Computed in |
|---|---|---|
| `u32` | `uint32_t` | `float`, converted back by truncation |
| `f32` | `float` | `float` |
| `bf16` | the high 16 bits of a `float` | `float`, stored rounded to nearest, ties to even; a NaN stays a NaN |

An element past a tensor's edge reads as zero and is never written.

## The Tensix backend

The C++ that today's class DSL writes for Tenstorrent's chips, compiled by the chip's runtime at
load. Its text names the script, each statement's line and the values' names, so its output
depends on the positions and names too; its key covers the full hash. Its build is kernels: the
movement source on the reader and on the writer, each tensor's layout as compile arguments, and
the compute source, with each stream's pair of buffers by today's layout rules. An entry's number counts from 1, a tensor is one argument word (its address),
and its bounds are each stream's tiles, which a host checks before sending.

A compile's target is `tensix` with no features: the text is one for both chips and the key holds
no chip. A chip's name there is refused; the chip, `blackhole` or `wormhole_b0`, is checked at
load. At load the backend also writes the thread's header from the tensors given (their pages and
kinds), which no compile knows.

At this stage it emits the first lesson and the corpus's program 1 (`Scalars.line`): one program
of one handler on a group, a loop over a read stream that pushes a tile function's result into a
sink of the same count, in one block. Each stream is of a partition of a tensor's tiles or of the
tiles `t.tiles(first, n)` themselves, its ends `u32` words, numbers, the rank, `+ - *` and named
values of these (`let first = worker.rank * 64`, a line of the movers). A kernel's case reads
only the rank, size and words its lines use. Partitions of the same values are printed once.
A tile function may be on blocks, the function applied to each valid tile of the block (today's
compute loop).

It also emits the corpus's program 2 (`StepHandlers.sum`). Its loop adds each `f32` tile of a
stream, one tile a block, to a `var` of `zeros`. After the loop, `reduce(total, with = sum)` sums
the group's totals, and the leader stores the result in one tile. Which worker is the leader is
the backend's statement, not the language's: on Tensix it is rank 0 of the group
(`km_rank == 0`). A `reduce` whose combine is commutative leaves the grouping and order to the
backend. On Tensix that is a tree over the NoC, as today's `form="tree"`. Its arrival buffer holds
slots of one tile, sized at load from the machine's islands as today's `tree_slots` sizes them
from a session's graph: 7 on one island; over several, the largest island's rounds and a slot
each other island, whichever is more (8 on two islands of 120, 10 on four). The count is a
constant of the header `<program>.h`, as in today's make, so the load writes that header again
for its machine, in place of the compile's, and the folder is the load's. Its out buffer holds
two tiles. Compute unpacks
the read pair and the arrivals to Dst as float32, unrounded (a buffer's `unpack`, today's
`unpack_to_dest_mode`). A reduce with another combine is refused. Everything else is refused with
`K964`, by name.

It also emits the corpus's program 3 (`StepHandlers.add`). Its loop zips two read streams, one
tile a block each, and pushes a tile function of both (`plus(x, y)`) into a sink. Both read pairs
and the write pair move in one call (`movement.move(km_read0, km_read1, km_write0)`). Compute
waits for each read half, copies each tile to its own Dst tile and frees each half. A call takes
the Dst tiles its unit states (today's larger of its operands and results), so a batch is the Dst
half's tiles over that: two calls of `plus` an acquire. The call's tiles must be the loop's, in
order, followed by `f32` words; any other call, a third read stream, or a loop other than one
over each read stream once is refused by name.

It also emits the corpus's program 4 (`Gate.gate`): program 3's form in blocks of a module
parameter (`param BLOCK: u32 = 8 @bind(build)`, `block = BLOCK`), its function a template of the
port (`silu(a) * b`). The Tensix backend reads one hint, `@tensix.pattern(p)`, which names a
`u32` parameter of the handler: on a loop, it applies to each stream the loop walks; on a
`let` of a sink, it applies to that sink. Each stream then deals its blocks to the movers
by the command's word, `km::Pattern::of`, instead of its role's pattern (`AAAB` for a read,
`B` for a write). That word only chooses the mover of each block. The result is the same for
every valid pattern, so it is a hint: the program does not read the parameter, and a use of it
as a number is refused. Compute ends the case on a word that is not a pattern, as today's does, so the entry point
states the parameter as a pattern (`patterns`, beside the bounds), and a host sends only a valid
pattern's word. A `@tensix.pattern` naming anything else, on another statement, or on a reduce's
stream is refused by name (`K964`). The thread header prints each module parameter fixed at
build, of an int, as a constant of the class (`#define GATE_BLOCK 8u`), as today's does.

An entry point states every range of its tensors as a bound a host computes from the words and
the group (words, numbers, the rank and size, `+ - *`): program 1's `a.tiles(worker.rank * 64, 64)`
is checked for each rank. A range that cannot be stated is refused by name (`K964`): no range of
a tensor goes unchecked.

The tile function's two files, `tile_<f>.hpp` and its listing, are made by the port of today's
tile compiler (`backend/tensix/tile/`): the function traced into today's tile IR, its vector-unit
function, its unit program with its rules and hazard checks, Wormhole's forms of its words, and
the header with the emit's checks. `tests/ir/tensix_tile.rs` compares each stage with today's on a
corpus of tile functions. A tile function the port does not take yet is refused with `K964`, by
name; one today's compiler refuses too, with `K962` and today's text.
A tile function's products are at the fidelity its hint names, `func f(...) -> T
@tensix.fidelity(HiFi2) { ... }` (`LoFi`, `HiFi2`, `HiFi3` or `HiFi4`; another name is refused
with `K962` and today's text), and with no hint at today's default, `HiFi4`. A `func` takes
hints after its head, as a `combine` and an `extern func` do, and only hints: an undotted
attribute there is `K006`.

The tests `tests/ir/tensix.rs` and `tests/ir/frontend.rs` run today's emitter on the scripts (`python3`, else
`KOMO_IR_PYTHON`), with no card and no compiler, and compares every file byte for byte; it fails
without Python unless `KOMO_IR_NO_PYTHON=1` skips it.

## From Python

`komorebi_ir.emit(program, backend="tensix", target=None)` gives a backend's output for a
module's text or byte-code file, as plain data. Nothing is written and nothing is run. The backend
is `"tensix"` or `"cpu"`. With no `target`, the Tensix backend compiles for its target with no
chip, `tensix`, and the CPU backend for the host's architecture.

The result, `Emitted`, holds:
- the cache key, in hex;
- the files, each its name and bytes;
- the entry points, each with its number, argument words, the ranges a host checks and the
  parameters whose word is a pattern;
- how the output is built:
  - for the Tensix backend, the kernels (source, processor, defines, compile-time arguments,
    options, each a bool or a text) and the buffers (index, format, page bytes, pages, the
    other half of its pair, how compute unpacks it to Dst when not the default: `unpack`,
    tt-metal's name, `UnpackToDestFp32`) the machine's runtime compiles and makes at load;
  - for the CPU backend, the steps run in the folder of the files.

A backend's refusal raises `IrError` with each refusal's code, and so does a backend name that
is not one of these two.

`komorebi_ir.load(program, tensors, chip=None, machine=None)` gives what a load on Tensix makes.
Its `tensors` are `LoadTensor`s by parameter name, each with its pages, its layout words and its
page in DRAM. Its `machine` is a `Machine`: its workers, a worker's memory, its islands as a graph
(`Islands`: the workers each holds, in order, and the connections between them, None for every
pair) and the load's options. With none, the machine is one island holding every worker. The result holds:
- the program's header `<program>.h` as the machine sizes it, in place of the compile's;
- the thread header, its name and text;
- the kernels, each tensor's layout among the compile arguments;
- the buffers, each format by the name today's loader gives it (`bfloat16`; the block-float
  formats are the `_b` family, `bfp8` as `bfp8_b`).

A named chip must be `blackhole` or `wormhole_b0`. A missing tensor, a tensor named twice and a
name the program does not take are each refused by name, and so is a machine whose islands are
not a graph of its workers (`K966`).
