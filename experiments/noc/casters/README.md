# Several casters into one rectangle

A counted VALID lets every core of a broadcast multicast its own part of each block
into the same rectangle at once, where the runtime's broadcast has one sender that every
other core hands its rows to first. The chip's NoC documentation warns that a broadcast
with path reservation (`NOC_CMD_PATH_RESERVE`, which tt-metal sets on every multicast)
"can lose a flit and hang the NoC" when its reservation is rejected two or more hops on
because another broadcast holds the virtual channel there
(`BlackholeA0/NoC/MemoryMap.md`). Several unlinked casters into one rectangle are that
arrangement. This probe runs it and checks every word. The protocol lives here, in
`kernels/counted.hpp`; the runtime's broadcast keeps one caster a rectangle
([Broadcast](../../../docs/reference/broadcast.md)).

## The counted VALID

`km::counted::Broadcast` is the runtime's `km::Broadcast` with one more argument word,
`casters`: ranks 0 to `casters - 1` cast parts. Rank 0 keeps the slot's freedom: once
every rank has popped the block a window back, it writes the mark (the sequence number
and `block + 1`) into every other core's VALID slot, which then means "the slot is free".
Each caster writes its part straight into every core's slot, waits for those writes to
be acknowledged, and adds one to a count word on every core with the NoC's atomic
increment, its own included. A block is there when a core's count reaches
`casters x (block + 1)`. The count is word 3 of the broadcast area's STATS slot, which
the runtime does not use; no core stores to it but the clear in `finish`, which comes
when the count is `casters x blocks` and before the core's last DONE-BY, which rank 0
waits for before the next command's first mark.

| Call | A caster | A core that does not cast |
|---|---|---|
| `reserve(block)` | its own slot free, then the slot free everywhere; the slot's address | gives up |
| `part(block, source, offset, bytes)` | `bytes` from L1 `source` into every core's slot at `offset`, its own too | gives up |
| `cast(block)` | its writes acknowledged, one to every core's count, then a timed wait for the block's count | gives up |
| `arrive(block)` | gives up | a timed wait for the block's count |

Rank 0 sits outside the receivers' rectangle and multicasts into it; a caster inside it
sends a unicast into rank 0's slot and at most two multicasts, the ranks before it and
the ranks after it, so no rectangle holds its own source. The driver checks that the
receivers' NoC addresses step by one a rank, as a caster finds the others.

## What runs

| Part | File |
|---|---|
| The counted VALID's broadcast (`km::counted`) | `kernels/counted.hpp` |
| Handler, buffers, the word expected at each place of each block | `kernels/casters.h` |
| Reader (NoC0): each caster writes its part of a block and casts it; every core checks every word of every block and the count beside it, then writes its result row | `kernels/reader.cpp` |
| Compute: takes and gives back each block | `kernels/compute.cpp` |
| Writer: no part | `kernels/writer.cpp` |
| Driver: builds the program, runs jobs, stops, reads and checks every row | `casters.py` |

A job casts `--blocks` blocks of `--pages` 2 KiB pages with a window of `--window`. Every
core checks that each word of each block is the one its place and block give, that its
count after block `b` lies between `casters x (b + 1)` and `casters x (b + window)`, that
it is `casters x blocks` at the end, and that `finish` cleared it. The rows are read once
the program has stopped. `--casters 0` runs the same blocks with the sender alone.

## Result

One p150a, a column of 10 cores (`--rect 0,0,1,10`), all 10 casting, 64 KiB blocks, a
window of 2, 2,000 blocks a job, 40 jobs in 20 s: 80,000 blocks, 800,000 casts. A block
makes 18 data multicasts into the column (rank 0 one into ranks 1 to 9; ranks 1 to 8 two
each, the ranks before and after them; rank 9 one) and 9 unicasts into rank 0's slot,
then as many multicast and unicast atomic increments: 1.44 million data multicasts and
1.44 million multicast increments in all. Every word (1.3 × 10^10 checks) and every count
was right, and nothing hung. 255.7 us a block, most of it the word check on each core.
That is a count beside the documentation's warning, not a guarantee from it.

```
python experiments/noc/casters/casters.py --rect 0,0,1,10 --seconds 20 --jobs 100
```

## Tried in the linear prefill

The dense linear layer's prefill was run with every rank of a spread row or column
broadcast casting its own share by this protocol, in place of handing it to the sender:
two cards, the five shapes whose term left was the broadcast's step. Every product was
exact on both cards, and every shape was slower (us a call):

| M x K x N | Steps | Hand-over | Counted | Counted / hand-over |
|---|---|---|---|---|
| 1024 x 4096 x 4096 | 8 | 126.3 | 260.9 | 2.07 |
| 4096 x 4096 x 4096 | 16 | 430.8 | 565.0 | 1.31 |
| 4096 x 17408 x 5120 | 68 | 1,572.6 | 2,480.8 | 1.58 |
| 4096 x 15360 x 3840 | 48 | 1,159.3 | 2,108.7 | 1.82 |
| 4096 x 9216 x 2560 | 24 | 681.5 | 1,126.7 | 1.65 |

The account of the slowest core says where. At 1024 x 4096 x 4096 (237 us) X waits 89.7
us for the slot to be free, reads 10.8 and casts 114.3; W waits 86.2, reads 12.6 and
casts 126.3, where the hand-over's row step was 55 us of hand-over and 37 of cast. At
4096 x 15360 x 3840 X's free wait is 1,685.7 of 2,073.9 us; at 4096 x 9216 x 2560 W's
cast is 807.7 of 1,062.8. The free mark waits for every core of the rectangle, so each
step waits for the slowest core of the row and, through compute, of its columns; the
cast waits for every caster's acknowledged parts and then its increments. That is two
round trips a step that every caster finishes in lockstep, where the hand-over is one,
pipelined. So the runtime keeps one caster a rectangle.

The first two-card run gave up on the second card: a caster's write to its own slot and
its own increment took the core's flat rank as a core-table index, which is the same
number only on the first card. That is fixed in `kernels/counted.hpp`.

Measured on 2026-11-15. JSON: the dense project's `linear.py --counted --json`
(the slowest and the busiest core's account). The run's `--counted` flag of the dense
project is not kept.
