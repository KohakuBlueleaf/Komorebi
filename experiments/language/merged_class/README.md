# Two classes merged into one program

Question: can two programs written as classes, the gate of
[22 gate](../../../examples/22_gate/as_class/README.md) and the linear layer of
[dense_class](../dense_class/README.md), be ONE program (`km.merge`), each handler
keeping its own class's constants and tensors, and what does a command of each cost
beside its own program?

`merged_class.py` makes `Gate` (bfloat16, 12 cores of 64 tiles, example 22's command)
and `Linear` (the K split for one row tile, 112 cores, and its SwiGLU gate), then
`km.merge(gate, linear)`
([Classes merged into one](../../../docs/reference/classes.md#classes-merged-into-one)).
With `--beside` each program is also loaded alone, Linear first and the gate last, and
the same commands run on it.

```bash
python merged_class.py --beside
```

The checks are the two scripts' own: the gate's output against float64 (values wrong
past the hand program's bound), Linear's `exact` (small integers, Y must equal the
host's product) and `swiglu` (relative RMS against float64), read after the run stops.

## What the build makes

Four handlers, numbered over the two classes: `gate.gate` keeps 1 (as `gate_gate`, a
name both classes have), Linear's `split`, `grid` and `gate` take 2, 3 and 4 (the last
as `linear_gate`). Both classes name tensors `a`, `b` and `out` with different
examples (768 pages; 1,024 x 4,096 tiles): Linear's are the kernel's `linear_a`,
`linear_b`, `linear_out`, which its handlers' lines name in a block of their own.

The gate's `BLOCK` is 2 here, not its default 8. Card 1, 2026-11-25, `--beside
--gate-block 8`: the build makes the merged program only at shared indices of other
pages (1,363,968 B of the 1,372,048 B left; 1,642,496 B as one set: the gate's three
pairs of 8-tile blocks are not Linear's), and the load refuses it: its kernels take
99,936 B of the 97 KiB (99,328 B) kernel buffer. The load places `linear_gate`, then
`gate_gate`, then `split` as an overlay; a handler's overlay parts do not set its pages
at its entry, so each of those builds needs 1,642,496 B, as one set or by shape
(1,372,032 B left), and the handler stays resident, and the load ends with `KernelSizeError`
(`Session(kernel_bytes=100992)` would hold the kernels). The same command a second
time ends the same: the script gives the session its kernel buffer, so no size is
remembered. At 9 the build itself refuses it (1,683,456 B as one set, 1,384,448 B at
shared indices, of 1,372,096 B, a run without `--beside`). At 2 the gate's streams
share Linear's pairs (the same role, format, page and block) and add no buffer.

On a mock device (the build of each load, the JIT cache the card would use):

| Program | Kernel text (BRISC, NCRISC, TRISC0 to 2) | Total |
|---|---|---:|
| Gate, `BLOCK = 2` | 14,532, 10,424, 6,372, 2,724, 3,284 | 38,064 B |
| Linear | 28,288, 25,640, 16,636, 9,232, 11,644 | 92,304 B |
| merged | 29,592, 27,144, 16,440, 11,164, 13,436 | 98,656 B |

The merged kernels are 6,352 B over Linear's: the gate's case and its own copy of the
tile function `silu_gate` (the two classes' functions are two functions). Linear alone
is over its own script's kernel buffer of 89 KiB in this tree (92,304 B; its README
measured 90,192), so this script gives the session 97 KiB, and the buffers still fit
(1,373,904 B of 1,432,448 B in the load's table).

At 91 KiB the load does not fit and picks: `split` (7,497 B of generated C++, the
largest of the handlers that may move; `grid` holds broadcasts and stays), then
`linear_gate`, then `gate_gate`, each placed as an overlay and built again. The overlay
builder refuses each part (a stream's handler keeps its own copy of the runtime's
`km::move_filled` and `km::move_drained`), so each stays resident, and with none left
the load is refused with the bytes.

## On the card

Card 1, a Blackhole p150a, 2026-11-16, seed 1, `--beside`: Linear alone, then merged,
then the gate alone, in one session; three timed commands a handler, the median.

| Handler | Check | Alone, us | Merged, us |
|---|---|---:|---:|
| gate (12 cores, 64 tiles, `BLOCK = 2`) | 0 of 786,432 wrong, 3.89e-3 largest relative | 80.54 | 80.84 |
| Linear's split (112 cores), exact | 0 wrong | 94.68 | 94.72 |
| Linear's split, random values | RMS 4.38e-3 | 93.70 | 93.86 |
| Linear's gate (64 cores) | RMS 1.64e-3, SwiGLU 8.03e-3 | 7.30 | 7.74 |

Every handler is exact (or within its bound) merged as alone, with the same errors to
the last digit. Merged, a command costs 0.04 to 0.4 us more, the gate's 0.4 %. The load
printed the four handlers resident, in the kernels. Beside each program's own page:
the gate class at `BLOCK = 8`, 83.70 us; Linear's K split, 94.2 us.

A first run loaded the gate alone, then Linear alone, whose launch tt-metal refused:
"Statically allocated circular buffers ... clash with L1 buffers ... L1 buffer
allocated at 1423872 and static circular buffer region ends at 1464448", though
Linear's L1 table fits; after the gate's load, whose host transfer buffer is 82,000 B
against Linear's 32,848, an L1 buffer stays where Linear's buffers reach. So the
script loads Linear first.
