# 05 hop

A dynamic segment: items move between cores as their data decides, and the run ends by
itself. An item is `{id, hops left, state, sum}` (words 1 to 4 of 16; the
runtime owns word 0). Each visit advances a small generator in `state`
and adds it to `sum`; while hops are left the item hops to member `state mod count` of
the team, and when none are left it retires into the core's totals. The host replays
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

The dispatch kernel does not end dynamic runs yet, so the example runs under the test
feeder, a kernel on the last core that issues records and watches the status lines:

```
cd tests/hardware/feeder
python hop_fed.py --count 2 --items 1 --hops 1
python hop_fed.py --count 119 --items 10000 --hops 100
python hop_fed.py --count 119 --items 10000 --hops 10 --one-core 0
python hop_fed.py --count 8 --items 16 --hops 4 --probe 3
python hop_fed.py --count 8 --items 16 --hops 4 --stall 2 --deadline 2
python hop_fed.py --count 2 --max-live 1000000 --queue-items 65536
```

It prints one JSON line: `exact` (every core against the replay), retired and expected,
hops, the sums of the status lines, any give-up per core, and the times from the
issue and from the last retire to the quiet end.

## Measured

One Blackhole p150a, slow dispatch, AICLK 1350 MHz, team of cores 0 to 118, 64-byte
items, queues of 16384 items, `max_live` 16384 (quota 137 per core).

| Case | Result |
|---|---|
| 10000 items, 100 hops each (10^6 hops) | exact in 17 of 18 runs; 1010000 items sent and consumed; issue to quiet end 11.4 ms; last retire to quiet end 28 to 33 µs; quiet end to the end record's completion 5.7 µs |
| 10000 items, 10 hops each, every hop to member 0 | exact; issue to quiet end 22.9 ms |
| member 3 spawns until its quota is gone (8 cores) | 2 seeded items and 135 spawns, then `spawn` returned false; exact |
| member 2 stalls in its body (8 cores) | the run stopped; core 2 named, its reader gave up waiting for a free packet page; the next run on the card exact, no reset |
| `max_live` 1000000 with queues of 65536 | refused before any device opened |

Not every run passes yet: one run of 10^6 hops in 18 stopped with reason 32 (`lost`). A writer's atomic add is counted as answered and the destination's tail has
moved, but the returned position never lands in the writer's L1, so the reserved
positions are never written and the destination reports the gap. The cause is open.

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
