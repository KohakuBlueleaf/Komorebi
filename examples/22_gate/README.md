# 22 gate

The gate of a SwiGLU layer, `out = silu(a) * b`, over two tensors in DRAM, as a program of
two kernels: the one data-movement source built for both movers and a compute kernel.
This is the kernel pair the [README](../../README.md#a-kernel) shows, built and run on a
group of cores and checked against numpy.

A command goes to a core group; core k of the group takes the pages from k x `tiles` for
`tiles` pages. The movers bring that share of `a` and `b` from DRAM into L1 through two buffer
pairs and take `out` back through a third ([Movement](../../docs/reference/movement.md)),
the blocks of every stream dealt to the two movers by the command's pattern. Compute takes
the blocks in order, each from the half its mover filled, and for each block of
`kBlock` = 8 tiles (what the vector unit's registers hold) copies `a` into the registers,
applies `silu`, multiplies by the block of `b` in place and packs the result to the
output pair.

| Handler | Arguments | What a core does |
|---|---|---|
| `GATE` | a0 `a_address`, a1 `b_address`, a2 `out_address`, a3 `tiles` (a multiple of 8), a4 pattern word | out = silu(a) * b over the core's `tiles` pages |

The host fills `a` and `b` with random bfloat16 normal numbers, runs the command, reads
`out` back by chunk and compares every value with `silu(a) * b` in float64 from the same
bfloat16 inputs. A value is wrong when it is farther than 2^-6 of its size plus 2^-10
from that. The script prints the largest error and the largest relative error (over
values of 2^-4 or more), and exits non-zero if any value is wrong.

## Files

| File | What it is |
|---|---|
| `kernels/movement.cpp` | the data-movement kernel of both movers: reads of `a` and `b` and the write of `out`, one `km::Movement` |
| `kernels/compute.cpp` | the compute kernel: copy, `silu`, multiply, pack, a block at a time |
| `kernels/gate.h` | the handler, `kBlock` and the indices of the three buffer pairs |
| `gate.py` | declares the program, runs the command, checks every value, prints one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python gate.py --cores 12
python gate.py --cores 1 --tiles 64 --pattern AB
```

`--cores 0` (the default) is every core of the grid.

## Numbers

A Blackhole p150a, `python gate.py --cores N --tiles 64` (pattern `AAAB`, seed 1, host
load 5.1 to 5.3), each run once:

| Cores | Largest error | Largest relative error | Values wrong | Card time |
|---|---|---|---|---|
| 1 | 4.4e-2 of values up to 8.5 | not printed by that version of the script | 0 of 65,536 | 75.0 µs |
| 12 | 5.5e-2 of values up to 12.5 | 1.45e-2 | 0 of 786,432 | 76.0 µs |

The card time is the command's, from its issue to its event. The error is more than the
output's rounding to bfloat16 (at most 2^-9 of a value): the rest comes from `silu` and
the multiply in the vector unit at the program's default fidelity; the example did not
measure the two apart.
