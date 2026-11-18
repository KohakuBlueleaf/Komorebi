# 24 thread rank

X280 cores as ranks of a dynamic run
([X280 ranks](../../docs/reference/dynamic.md#x280-ranks)). The run's ranks are 12
cores and X280 cores 0 to 3 of work cluster 1's thread: ranks 0 to 11 are the cores,
12 to 15 the X280 cores. Messages go both ways, and the run ends by itself when every
line is quiet, as a dynamic run of cores alone does.

Each core seeds 64 messages, each to an X280 rank. A message carries an id, its hops
and a value; every visit steps the value (a xorshift on a core, a multiply-add on an
X280 rank that adds its rank). While hops are left a core sends it on to the X280 rank
`count + value % 4` and an X280 rank to the core `value % 12`; the visit that finds
no hop left consumes it. So a message of 100 hops crosses between the grid and the
X280 cores 101 times, and where it ends depends on every value before.

The thread is one call ([X280 cores as ranks](../../docs/reference/threads.md#x280-cores-as-ranks)):

```c
int kmx_main(int core) {
    /* ... totals: region 0 */
    return kmx_messages_run(rank, totals, deadline);
}
```

`rank` takes its messages with `kmx_messages_next` until it returns 0, steps each, and
consumes it or forwards it with `kmx_messages_forward(messages, value % count, message,
kHandlerGrid)`.

| Handler | Where | Arguments |
|---|---|---|
| `GRID` | the cores' compute (the `RUN` names it) | a0 the messages each core seeds, a1 the hops of an even id (odd: one more), a2 flags: `kSeedToCores` (1) each seed to the core `value % count` first, `kSeedEven` (2) odd ids no hop more |
| `X280` | the X280 ranks | - |

The host replays every message in numpy and checks each consumer exactly (the count,
the sum and an xor of id and value at each core and each X280 rank), that the messages
the cores sent to the X280 ranks are the ones the X280 ranks took, that the X280 ranks'
forwards are the cores' visits, that no X280 rank gave up, that each X280 rank started
with `max_live // the run's ranks` sends (its quota, `x280_quota`: 8,192 // 16 = 512 by
default, though the load holds the whole grid), and that the `RUN` and the thread both
ended done.

## Files

| File | What it is |
|---|---|
| `kernels/rank.h` | what the cores and the thread share (plain C): handlers, message words, result words, the two steps |
| `kernels/compute.cpp` | the cores' `GRID` body under the runtime's message reader and writer |
| `thread/rank.c` | the thread: `kmx_messages_run` and the X280 ranks' function |
| `thread_rank.py` | the program, the run, the replay and every check; one JSON line |
| `islands.py` | the same program on two cards: island 0 holds the X280 ranks, the cores of both islands send to them; every consumer and each core's visits against the replay |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python thread_rank.py --x280 1 --seeds 4 --hops 2   # small
python thread_rank.py                               # 12 cores, 4 X280 ranks
```

`--cores` (12), `--x280` (1 to 4, 4), `--seeds` (64), `--hops` (100), `--live`
(`max_live` and `queue_messages`, 8,192), `--deadline` (the `RUN`'s, 10 s), `--wait`
(the wait limit, 1 s). The script exits non-zero if a check fails.

On two cards ([X280 ranks over islands](../../docs/reference/dynamic.md#x280-ranks)),
`islands.py` takes the group by `--first` and `--ranks` (flat, counter tiles
included; island 1 begins at island 0's count), `--to-cores` and `--even` (the
flags above). An X280 rank forwards to the core ranks of both islands: to island 1's
through a ring of its own on island 0's link core, the link service sending each line
on as a core's delivery.

```
python islands.py                                          # island 0's 12 cores
python islands.py --first 106 --ranks 26                   # both islands, both ways
```

## Numbers

A Blackhole p150a at AICLK 1,350 MHz, 2026-11-14, host load 4 to 5, one run each:

| | `--x280 1 --seeds 4 --hops 2` | default |
|---|---|---|
| messages to the X280 ranks | 96 | 39,168 |
| messages from the X280 ranks to the cores | 72 | 38,784 |
| consumed at the cores / at the X280 ranks | 24 / 24 | 384 / 384 |
| the `RUN`, host submit to done | 0.6 ms | 14.5 ms |

Every consumer exact in both, no X280 rank gave up, no send rejected. The default run
moves 77,952 messages across in 14.5 ms, about 0.19 µs a crossing over the four X280
ranks and 12 cores together, the host's time around the wait included.

On two cards, `islands.py --first 106 --ranks 26` (the group's 26 cores, 12 on island
0 and 14 on island 1, counter tiles included; four X280 ranks), 2026-11-18, one run: every consumer and every core's visits
exact, no X280 rank gave up; the X280 ranks sent 77,568 messages, island 1's cores took
45,398 visits, and the `RUN` took 0.21 s from the host's submit to done.

## What this example does not do

- X280 ranks of more than one cluster, or an X280 rank that sends to another X280
  rank: not possible in this version.
- A grid of more than 124 cores with X280 ranks: the X280 ranks use core indices 124
  to 127.
