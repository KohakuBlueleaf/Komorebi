# 05 forward

A dynamic handler: messages move between cores as their data decides, and the run ends by
itself. A message is `{id, forwards left, state, sum}` (words 1 to 4 of 16; the
runtime owns word 0). Each visit advances a small generator in `state`
and adds it to `sum`; while forwards are left the message forwards to rank `state mod count` of
the group, and when none are left it consumes into the core's totals. The group's last two
cores are its counter tiles (`--counters`), which hold the queue tails and take no message. The host replays
every message in numpy and checks each core's count, total and visits.

The program is the compute kernel alone: the reader and writer are the runtime's
generic dynamic kernels. The [dynamic handlers](../../docs/reference/dynamic.md) page
explains the calls, the quota and the quiet end.

| Command argument | Meaning |
|---|---|
| a0 | `message_count`, shared over the cores of the group, each seeded by its own core |
| a1 | `forwards` per message |
| a2, a3 | `mode`: 0 random destinations, 1 every forward to rank `target_rank` (a3) |
| a4 | `stall_rank_plus_one`: rank + 1 that stalls in its body (0: none), a test of the timeout |
| a5 | `probe_rank_plus_one`: rank + 1 that sends until its quota is gone (0: none), a test of the quota |

## Running it

Through a session: the dispatcher on X280 cluster 0 submits the RUN command, reads
the cores' status lines and ends the run at the quiet end:

```
cd examples/05_forward
python forward.py --count 3 --messages 1 --forwards 1
python forward.py --messages 10000 --forwards 100 --repeat 30
python forward.py --messages 10000 --forwards 10 --one-core 0
python forward.py --messages 118 --forwards 72000 --one-core 0 --max-live 128 --queue-messages 128
python forward.py --count 8 --messages 16 --forwards 4 --stall 2 --deadline 2 --again
```

`--count` 0 (the default) is every core. The same run under the test feeder, a kernel on
the last core that stands in for the dispatcher (no X280):

```
cd tests/hardware/feeder
python forward_fed.py --count 3 --messages 1 --forwards 1
python forward_fed.py --count 119 --messages 10000 --forwards 100
python forward_fed.py --count 119 --messages 10000 --forwards 10 --one-core 0
python forward_fed.py --count 8 --messages 16 --forwards 4 --probe 3
python forward_fed.py --count 8 --messages 16 --forwards 4 --stall 2 --deadline 2
python forward_fed.py --count 3 --max-live 1000000 --queue-messages 65536
```

It prints one JSON line: `exact` (every core against the replay), consumed and expected,
forwards, the sums of the status lines, any give-up per core, and the times from the
submit and from the last consume to the quiet end.

## Measured

Through a session (`forward.py`), card 0, 2026-11-13, every core (120: 118 cores and the
counter tiles 118 and 119) unless named, 64-byte messages, queues of 16384 messages,
`max_live` 16384 (quota 138 per core). Times are the `RUN` command's, from its submit by
the dispatcher to its completion, which follows the quiet end. X280 at the
session's 1000 MHz; the 1750 MHz opt-in's times, from the runs before the default
changed, in brackets (the stall, quota and rejection rows were not re-run).

| Case | Result |
|---|---|
| 3 cores (2 ranks, 1 counter tile), 1 message, 1 forward | exact; 24 µs [23] |
| 100 messages, 10 forwards each (10^3 forwards) | exact; 172 µs; the kernel checked the lines 487 times [158 µs, 1109 checks] |
| 1000 messages, 100 forwards each (10^5 forwards) | exact; 3.43 ms [3.44] |
| 10000 messages, 100 forwards each (10^6 forwards) | exact in 30 of 30 runs in one session (`--repeat 30`), median 11.8 ms (11.6 to 12.0); [exact in 91 of 91 runs: 30 in a row in one session, 10 in each of 5 more sessions, 11 more; median 11.8 ms (11.6 to 12.0)] |
| the same with one counter tile (`--counters 1 --repeat 10`, 119 cores) | exact in 10 of 10; median 11.6 ms [11.6] |
| 10000 messages, 10 forwards each, every forward to rank 0 | exact; 23.6 ms [23.6] |
| 118 messages, 72000 forwards each, every forward to rank 0, queues of 128 and `max_live` 128 (2026-11-14): 8,496,001 positions of rank 0's queue, 66,375 laps, so its lap mark wrapped once | exact; 2.17 s (card 1) |
| rank 3 sends until its quota is gone (8 cores) | 3 seeded messages and 135 sends, then `send` returned false; exact |
| rank 2 stalls in its body (8 cores) | the `RUN` command's timeout names its handle; the kernel names core 2 (`KMD_H_MESSAGES_GAVE`), whose reader gave up waiting for a free packet page (24); the next run in the same session exact |
| `max_live` 1000000 with queues of 65536 | rejected before any device opened; `ok` false, exit code 1 |

Under the test feeder (`tests/hardware/feeder/forward_fed.py`), card 0, AICLK 1350 MHz, core group 0 to 118
(117 cores, counter tiles 117 and 118), 64-byte messages, queues of 16384 messages,
`max_live` 16384. The rows after the first were measured before the counter tiles,
with every core in the group.

| Case | Result |
|---|---|
| 10000 messages, 100 forwards each (10^6 forwards) | exact in 3 of 3 runs; submit to quiet end 11.4 ms |
| 10000 messages, 10 forwards each, every forward to rank 0 | exact; submit to quiet end 22.9 ms |
| rank 3 sends until its quota is gone (8 cores) | 2 seeded messages and 135 sends, then `send` returned false; exact |
| rank 2 stalls in its body (8 cores) | the run stopped; core 2 named, its reader gave up waiting for a free packet page; the next run on the card exact, no reset |
| `max_live` 1000000 with queues of 65536 | rejected before any device opened |

Before the counter tiles, when every core kept its own tail, 4 of 22 runs of 10^6 forwards
under the feeder ended with a reported reason: a reservation's answer was lost (32) or a
packet wait ran out (24, 28). See [atomic_answers](../../experiments/noc/atomic_answers/).

A reserved position that is never written (a gap) cannot be made on the card; the CPU
harness (`experiments/checks/runtime_harness`, case `messages_gap`) shows it reported with the core
and the position.

## Over two cards

`islands.py` runs the same program and kernels on two cards (`Session(cards=[0, 1])`,
nothing else) or a split card (`--cards 1 --islands 2`, `--line`): one RUN over every
rank by default, or over `--group first,count` in flat ranks, so a forward to a rank of
another island crosses. The script opens one link (`--links`; one link core an island,
238 ranks, 234 message ranks); with `--links 4` each island keeps its last two cores as
link cores (no rank), so the run has 236 ranks; the last two ranks of each island's part
of the group are its counter tiles. A forward to another island is a delivery by the
link service (its island's link core puts it in the rank's queue). `--pages n` writes n
pages of island 0 into a tensor of island 1 while the run goes on and checks them. The
script reads `run.crossings()` before and after the RUN and checks that the deliveries
counted are the replay's forwards to another island (one each, however many links it
crosses).

```
python islands.py --messages 4000 --forwards 250  # 10^6, two cards
python islands.py --cards 1 --islands 2 --group 4,100 --messages 4096 --forwards 64 --pages 64
python islands.py --cards 1 --islands 3 --line --group 10,100 --messages 4096 --forwards 64 --pages 64
```

Cards 0 and 1 (Blackhole p150a), 2026-11-16: 10^6 forwards (4000 messages of 250) exact,
1,522.9 ms on one link, 499,980 deliveries; 1,469.1 ms with `--links 4` (an island's
deliveries go through one lead CPU, so link 1 carries them all); one card 11.77 ms. The
message service this replaced took 302.0 ms on one link. Group 30,150 with 64 pages
across: exact, 369.1 ms, every page right.

Card 1 split into two islands, group 4,100 (53 and 43 message ranks), 4096 messages of
64 forwards with 64 pages across: exact, 301.9 ms, 129,636 deliveries, every page right.
As three islands in a line, group 10,100: exact in one run (433.7 ms, 173,458
deliveries, a forward two islands away forwarded through the island between) and NOT
exact in another (one delivery more than the replay; cause not found).

## Files

| File | What it is |
|---|---|
| `kernels/compute.cpp` | the handler body: seeding, the visit, forward or consume, the results |
| `kernels/forward.h` | handler number, message words, modes, result words, the generator |
| `replay.py` | the numpy replay of every message |
| `islands.py` | the run over two cards or a split card, checked against the replay |
| `tests/hardware/feeder/forward_fed.py` | the run under the test feeder and its checks |
