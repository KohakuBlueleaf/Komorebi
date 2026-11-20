# Run scripts

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

One statement a line; `#` starts a comment that runs to the line's end.

| Line | Does |
|---|---|
| `options workers N islands A,B,...` | Loads the module on `N` workers (1 if no line says), in islands of `A`, `B`, ... workers (one island if none). Each option is optional; the line, when there is one, comes first. |
| `tensor NAME FORMAT ROWS COLS` | Makes a tensor of `ROWS` by `COLS` elements of `FORMAT`, in tiles of the module's tile, every element 0. |
| `... fill VALUE` | Every element `VALUE`. |
| `... fill tile+K` | Every element of tile `i` (row of tiles after row of tiles) `i + K`; `fill tile` is `fill tile+0`. |
| `invoke Program.handler ARG... on FIRST COUNT` | Invokes the handler on the workers `FIRST` to `FIRST + COUNT - 1` and waits for its end. |
| `control NAME ARG...` | Runs the control program `NAME`. |
| `read NAME` | Shows the tensor as it is now in the report. |

An argument is a tensor's name, a scalar `VALUE:TYPE` (`5:u32`, `2.0:f32`, `-1:i32`), `true` or
`false`.

## The report

One line for each invocation as it ends, and each tensor read where it was read:

```
#0 Scale.run group(0, 3): completed
tensor y bf16 [160, 32]
  tile 0: 2.0*1024
  tile 1: 4.0*1024
  ...
```

An end is `completed` (with the leader's result, if the handler has one), `failed REASON`, or
`invalid: WHAT`. The interpreter also writes where a failure happened (`at rank 0, p0.h0 i5`).
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
