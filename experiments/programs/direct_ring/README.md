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
```

One session, a load a run, six runs (direct, then queue, for each):

| Case | Pairs declared | Group |
|---|---|---|
| small | 4 ranks as pairs, r to r + 1 mod 4; 1 token a rank, 10 hops | cores 0 to 5 (4 ranks, 2 counter tiles) |
| ring | the offset 1: r to r + 1 mod 118 | the grid, 118 ranks |
| pairs | r to 5r + 1 and to r² + 3 (mod 118), the token's state picks one; a rank has 1 to several senders | the grid |

`--tokens` and `--hops` (1 and 8475: 1,000,050 hops) set the grid cases. The
[dynamic handlers](../../../docs/reference/dynamic.md#a-direct-message) page has the
calls.

## Measured

Card 1, 2026-11-16, AICLK 1,350 MHz, host load 5.6 to 5.8. A hop is the `RUN`
command's time over the hops of one token; all six runs exact.

| Case | direct, forward when busy | queue only |
|---|---|---|
| small, 40 hops | 40.5 µs; 37 direct, 3 busy | 42.5 µs |
| ring, 1,000,050 hops | 15.72 ms, 1.85 µs a hop; 768,387 direct, 231,663 busy | 90.34 ms, 10.66 µs a hop |
| pairs, 1,000,050 hops | 25.73 ms, 3.04 µs a hop; 674,350 direct, 325,700 busy | 27.05 ms, 3.19 µs a hop |

A busy slot is a receiver that has not yet taken the pair's last token: in the ring
every token reaches its next rank about when that rank's previous visitor does. In the
pairs case a rank with several senders gets several tokens at once, a third of the hops
find their slot busy, and those go through the queues. Why the ring through the queues
alone takes 10.7 µs a hop, five times a single thread's hop (2.1 µs in
[23_message_body](../../../examples/23_message_body/README.md)), these runs do not say.
