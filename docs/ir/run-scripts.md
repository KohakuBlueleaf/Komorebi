# Run scripts

!!! note "Being built"
    The Komorebi IR and its command `komo` are work in progress: the language still changes
    and nothing here is a released interface.

A run script is what a host does with a loaded module, as plain text: the tensors it makes, the
invocations and control programs it runs, and the tensors it reads. The same script is the input of
`komo run module script`, which runs it on the interpreter, and of a backend's driver. Both print
the same report, so the parity tests compare the two reports.

```
# T1 on a group of 3: five tiles of x, each element of tile i is i + 1, scaled by 2.
options workers 8
tensor x bf16 160 32 fill tile+1
tensor y bf16 160 32
invoke Scale.run x y 0:u32 5:u32 2.0:f32 on 0 3
read y
```

## Lines

One statement a line; `#` starts a comment that runs to the line's end; an empty line is nothing.
A line the reader does not know, or a malformed one, is an error that names the line's number and
the line.

| Line | Does |
|---|---|
| `options workers N islands A,B,... links A-B,C-D memory BYTES live N steps N` | Loads the module on `N` workers (1 if no line says), in islands of `A`, `B`, ... workers (one island if none). `links`: the pairs of islands joined (every pair if none; it needs `islands`). `memory`: the bytes of a worker's memory, which a `fit` parameter fills. `live`: the bound on a dynamic run's live messages (none if not given). `steps`: how many instructions an invocation may run on each worker before it gives up. Each word pair is optional, in any order; the line, when there is one, comes first. |
| `load NAME VALUE:TYPE` | Gives the parameter `NAME`, bound at load, its value. Only before the first `tensor`, `invoke`, `control`, `read` or `state` line. |
| `tensor NAME FORMAT ROWS COLS` | Makes a tensor of `ROWS` by `COLS` elements of `FORMAT`, every element 0. Its rows are padded up to a multiple of the row count of every tile shape of the module (their least common multiple), its columns likewise, so that it is whole tiles of each shape; it is stored in tiles of the module's default shape (its first `tile` line). With one shape, that shape's tiles. |
| `... replicated` | One copy of the tensor for each island, each holding the fill at first. Written before the fill. |
| `... fill VALUE` | Every element `VALUE`. |
| `... fill tile+K` | Every element of tile `i` of the default shape (row of tiles after row of tiles) `i + K`; `fill tile` is `fill tile+0`. |
| `... fill element+K` | Element `i`, counted row by row over the tensor's `ROWS` by `COLS`, `i + K`; an element past the tensor's edge 0. |
| `invoke Program.name ARG... on FIRST COUNT` | Invokes the handler, chain or dynamic run on the workers `FIRST` to `FIRST + COUNT - 1` and waits for its end. |
| `... with NAME=VALUE:TYPE ...` | Gives parameters bound at the invocation their values, by name; a parameter not given takes its declared value. |
| `control NAME ARG...` | Runs the control program `NAME`. |
| `read NAME` | Shows the tensor as it is now in the report, in tiles of the module's default shape; a replicated tensor copy by copy. |
| `state Program.member` | Shows the exported state member as it is now in the report, worker by worker. |

An argument is a tensor's name, a scalar `VALUE:TYPE` (`5:u32`, `2.0:f32`, `-1:i32`), `true` or
`false`.

### Examples of each line

Three islands in a line, of two workers each, and each worker's memory 64 KiB:

```
options workers 6 islands 2,2,2 links 0-1,1-2 memory 65536
```

A dynamic run with room for seven live messages, and a loop that gives up after 1000 steps:

```
options workers 4 live 7 steps 1000
```

A parameter bound at load (T3's `ROOM`), given before the tensors:

```
options workers 4
load ROOM 2:u32
tensor g bf16 64 32 fill tile+4
```

A replicated tensor, and a tensor whose elements count up row by row from 0:

```
options workers 4 islands 2,2
tensor w f32 128 32 replicated
tensor table u32 32 32 fill element+0
```

An invocation that gives the parameter `BLOCK`, bound at the invocation, the value 2:

```
invoke Count.blocks x 0:u32 20:u32 on 0 4 with BLOCK=2:u32
```

An exported state member, worker by worker:

```
invoke Spread.locate on 0 4
state Spread.seen
```

prints

```
#0 Spread.locate group(0, 4): completed
state Spread.seen
  worker 0: 2:u32, 0:u32*3
  worker 1: 2:u32, 0:u32, 1:u32*2
  ...
```

Each lesson has its scripts, with the report each gives, in `examples/ir/lessons/`.

## The report

One line for each invocation as it ends, and each tensor read where it was read:

```
#0 Scale.run group(0, 3): completed
tensor y bf16 [160, 32]
  tile 0: 2.0*1024
  tile 1: 4.0*1024
  ...
```

A run the interpreter refuses before anything runs, such as a load refused by name, has as its
report the one line `refused: ` followed by the refusal (`komo run` writes the refusal as an
error).

An end is `completed` (with the leader's result, if the handler has one), `failed REASON`, or
`invalid: WHAT`. The interpreter also writes where a failure happened (`at rank 0, p0.h0 i5`).
An invocation that does not start because one of its `after` did not complete ends
`failed dependency (#N Program.handler)`, `#N` and the name those of the line of the one that did
not complete.
A tile's elements are written in order, with a run of one value written once with its count. A
float is the shortest text that reads back to its value.

## Between backends

Two reports of one script are the same when:

- Each invocation completed, failed or was invalid on both.
- A failure's reason is the same too, unless the script has the line `# parity: any failure`.
  When several workers fail for different reasons, which reason is reported is the backend's.
- Every tensor read is equal, element by element, inside the tensor. Elements past its edge in
  the last row or column of tiles are not compared.
- A tensor that a failed or invalid invocation was given is not compared, since what that
  invocation wrote is not defined.

Floats are compared exactly. A script chooses values that every backend computes exactly, such as
small integers and powers of two; nothing here states a precision.

The CPU backend's driver does not yet run control programs, more than one island, or tensors and
scalars other than `u32`, `f32` and `bf16`. It refuses each by name.
