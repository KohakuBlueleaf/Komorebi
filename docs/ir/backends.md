# Backends

A backend turns a checked module into something a machine runs. It reads the program and its
hints only (the byte code's parts S and H, see [the byte-code file](bytecode.md)); the notes never
change what it emits, so one build hash names one output on every backend.

## What a backend states

On a target (an architecture and its features) a backend states, before it sees a program:

- the abilities it provides beyond the base: `group = grid`, `dynamic`, islands, local state;
- the external implementations it has, by name;
- its limits on a program's text: handlers in a program, parameters of a handler, run or chain,
  fields of a message;
- the formats it computes, and this page's section on how;
- the hints it reads, and its guarantees of progress.

On a machine (workers, worker memory, islands) it states, when a program is loaded, what depends
on that machine: how many workers and islands, how many bytes of state a worker holds, how many
invocations may be in flight. Islands are a count; nothing names their shape.

A program that asks for more is refused by name at the construct that asks, before anything is
emitted (`K960` to `K964`, [the checks](checks.md)). A backend never substitutes other behaviour.

Its output is files, the entries an invocation names, and the steps that build them, as data.
Compiling runs nothing and writes nothing. The output is kept under a key over the build hash,
the backend's name and version, and the target.

## The CPU backend

Portable C++17 on a small framework the crate carries, one thread for each worker. It provides no
ability beyond the base, has no external implementation, and has no limit on a program's text.
One invocation is in flight at a time.

At this stage it emits what the first lesson uses: a handler on a group over tensors and scalars,
views of tiles, a partition by rank, streams and sinks of blocks of one tile, and functions on
tiles. Everything else is refused with `K964`, by name.

The build runs `$CXX` (else `c++`). The tests that build an output fail without a compiler;
`KOMO_IR_NO_CXX=1` skips them.

### How it computes

A format is a notation; this is how the CPU backend holds and computes each one it states.

| Format | Held as | Computed in |
|---|---|---|
| `u32` | `uint32_t` | `float`, converted back by truncation |
| `f32` | `float` | `float` |
| `bf16` | the high 16 bits of a `float` | `float`, stored rounded to nearest, ties to even; a NaN stays a NaN |

An element past a tensor's edge reads as zero and is never written.
