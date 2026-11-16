# direct_ring

How fast does a message go from one core's L1 into another's, against the queues in
DRAM? One token a rank hops from rank to rank over pairs declared with
`program.dynamic(..., direct=...)`; at each hop the core tries `messages.direct()` and
forwards through the queue when the slot is busy (mode direct), or always forwards
(mode queue). The path of every token depends only on its own state, so the host
replays each one and checks every core's visits and each token's last rank, state and
sum of the ranks it passed.

```
python experiments/programs/direct_ring/ring.py --json ring.json
python experiments/programs/direct_ring/ring.py --as-class --cases small ring
python experiments/programs/direct_ring/ring.py --stamps --modes kModeQueue --cases ring pairs
```

`--as-class` runs `ring_class.py`, the ring pattern written as a class
(`went = self.direct(to, message)`, a forward when it is 0), with the same results
words and checks. `--stamps` builds the runtime's stamps and logs their sums
([What a queued hop costs](../../../docs/reference/dynamic.md#what-a-queued-hop-costs)).
`--modes` and `--cases` choose the runs.

One session, a load a run, eight runs (direct, then queue, for each):

| Case | Pairs declared | Group |
|---|---|---|
| small | 4 ranks as pairs, r to r + 1 mod 4; 1 token a rank, 10 hops | cores 0 to 5 (4 ranks, 2 counter tiles) |
| ring | the offset 1: r to r + 1 mod 118 | the grid, 118 ranks |
| pairs | r to 5r + 1 and to r² + 3 (mod 118), the token's state picks one; a rank has 1 to several senders | the grid |
| body | the offset 1, each token with a 2 KiB body (`--body`) that the core where it ends checks | the grid |

`--tokens` and `--hops` (1 and 8475: 1,000,050 hops) set the grid cases. The
[dynamic handlers](../../../docs/reference/dynamic.md#a-direct-message) page has the
calls.

## Measured

Card 1, 2026-11-16, AICLK 1,350 MHz, host load 3.9 to 6.2. A hop is the `RUN`
command's time over the hops of one token; all eight runs exact.

| Case | direct, forward when busy | queue only |
|---|---|---|
| small, 40 hops | 41.8 µs; 34 direct, 6 busy | 42.5 µs |
| ring, 1,000,050 hops | 16.44 ms, 1.94 µs a hop; 752,032 direct, 248,018 busy | 97.93 ms, 11.55 µs a hop |
| pairs, 1,000,050 hops | 26.20 ms, 3.09 µs a hop; 668,353 direct, 331,697 busy | 26.93 ms, 3.18 µs a hop |
| body, 1,000,050 hops | 22.32 ms, 2.63 µs a hop; 719,119 direct, 280,931 busy | 43.99 ms, 5.19 µs a hop |

An earlier job on the runtime before the writer's two-stage flush: ring 1.85 / 10.66 µs
a hop, pairs 3.04 / 3.19.

A busy slot is a receiver that has not yet taken the pair's last token: in the ring
every token reaches its next rank about when that rank's previous visitor does. In the
pairs case a rank with several senders gets several tokens at once, a third of the hops
find their slot busy, and those go through the queues.

The ring through the queues alone takes 11.6 µs a hop because its tokens move as a
convoy: the stamps show 22.6 tokens a flush, each waiting in its bin while its core
handles the whole packet. The class's ring (`--as-class`), card 1, all exact: direct
1.97 µs a hop (769,013 direct, 231,037 busy), queue 14.41.
