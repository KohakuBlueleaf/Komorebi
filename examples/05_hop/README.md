# 05 hop

A dynamic segment: items move between cores as their data decides, and the run ends by
itself. An item is `{id, hops left, state, sum}` (words 1 to 4 of 16; the
runtime owns word 0). Each visit advances a small generator in `state`
and adds it to `sum`; while hops are left the item hops to member `state mod count` of
the team, and when none are left it retires into the core's totals. The team's last two
cores are its counter tiles (`--counters`), which hold the queue tails and take no item. The host replays
every item in numpy and checks each core's count, total and visits.

The program is the compute kernel alone: the reader and writer are the runtime's
generic dynamic kernels. The [dynamic segments](../../docs/reference/dynamic.md) page
explains the calls, the quota and the quiet end.

| Record argument | Meaning |
|---|---|
| a0 | items, shared over the team's members, each seeded by its own core |
| a1 | hops per item |
| a2, a3 | mode: 0 random destinations, 1 every hop to member a3 |
| a4 | member + 1 that stalls in its body (0: none), a test of the timeout |
| a5 | member + 1 that spawns until its quota is gone (0: none), a test of the quota |

## Running it

Through a session: the dispatch kernel on X280 cluster 0 issues the RUN record, reads
the cores' status lines and ends the run at the quiet end:

```
cd examples/05_hop
python hop.py --count 3 --items 1 --hops 1
python hop.py --items 10000 --hops 100 --repeat 30
python hop.py --items 10000 --hops 10 --one-core 0
python hop.py --count 8 --items 16 --hops 4 --stall 2 --deadline 2 --again
```

`--count` 0 (the default) is every core. The same run under the test feeder, a kernel on
the last core that stands in for the dispatch kernel (no X280):

```
cd tests/hardware/feeder
python hop_fed.py --count 3 --items 1 --hops 1
python hop_fed.py --count 119 --items 10000 --hops 100
python hop_fed.py --count 119 --items 10000 --hops 10 --one-core 0
python hop_fed.py --count 8 --items 16 --hops 4 --probe 3
python hop_fed.py --count 8 --items 16 --hops 4 --stall 2 --deadline 2
python hop_fed.py --count 3 --max-live 1000000 --queue-items 65536
```

It prints one JSON line: `exact` (every core against the replay), retired and expected,
hops, the sums of the status lines, any give-up per core, and the times from the
issue and from the last retire to the quiet end.

## Measured

Through a session (`hop.py`), card 0, 2026-11-13, every core (120: 118 members and the
counter tiles 118 and 119) unless named, 64-byte items, queues of 16384 items,
`max_live` 16384 (quota 138 per member). Times are the `RUN` record's, from its issue by
the dispatch kernel to its completion, which follows the quiet end.

| Case | Result |
|---|---|
| 3 cores (2 members, 1 counter tile), 1 item, 1 hop | exact; 23 µs |
| 100 items, 10 hops each (10^3 hops) | exact; 158 µs; the kernel checked the lines 1109 times |
| 1000 items, 100 hops each (10^5 hops) | exact; 3.44 ms |
| 10000 items, 100 hops each (10^6 hops) | exact in 91 of 91 runs: 30 in a row in one session, 10 in each of 5 more sessions, 11 more; median 11.8 ms (11.6 to 12.0) |
| the same with one counter tile (`--counters 1`, 119 members) | exact in 10 of 10; median 11.6 ms |
| 10000 items, 10 hops each, every hop to member 0 | exact; 23.6 ms |
| member 3 spawns until its quota is gone (8 cores) | 3 seeded items and 135 spawns, then `spawn` returned false; exact |
| member 2 stalls in its body (8 cores) | the `RUN` record's timeout names its ticket; the kernel names core 2 (`KMD_H_DYN_GAVE`), whose reader gave up waiting for a free packet page (24); the next run in the same session exact |
| `max_live` 1000000 with queues of 65536 | refused before any device opened; `ok` false, exit code 1 |

Under the test feeder (`hop_fed.py`), card 0, AICLK 1350 MHz, team of cores 0 to 118
(117 members, counter tiles 117 and 118), 64-byte items, queues of 16384 items,
`max_live` 16384. The rows after the first were measured before the counter tiles,
with every core a member.

| Case | Result |
|---|---|
| 10000 items, 100 hops each (10^6 hops) | exact in 3 of 3 runs; issue to quiet end 11.4 ms |
| 10000 items, 10 hops each, every hop to member 0 | exact; issue to quiet end 22.9 ms |
| member 3 spawns until its quota is gone (8 cores) | 2 seeded items and 135 spawns, then `spawn` returned false; exact |
| member 2 stalls in its body (8 cores) | the run stopped; core 2 named, its reader gave up waiting for a free packet page; the next run on the card exact, no reset |
| `max_live` 1000000 with queues of 65536 | refused before any device opened |

Before the counter tiles, when every core kept its own tail, 4 of 22 runs of 10^6 hops
under the feeder ended with a reported reason: a reservation's answer was lost (32) or a
packet wait ran out (24, 28). See [atomic_answers](../../experiments/atomic_answers/).

A reserved position that is never written (a gap) cannot be made on the card; the CPU
harness (`experiments/runtime_harness`, case `dyn_gap`) shows it reported with the core
and the position.

## Files

| File | What it is |
|---|---|
| `kernels/compute.cpp` | the segment body: seeding, the visit, hop or retire, the results |
| `kernels/hop.h` | segment number, item words, modes, result words, the generator |
| `replay.py` | the numpy replay of every item |
| `tests/hardware/feeder/hop_fed.py` | the run under the test feeder and its checks |
