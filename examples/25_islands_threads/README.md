# 25 islands threads

Commands that start on one island and run partly on the other
([islands](../../docs/reference/runtime.md#islands),
[commands across islands](../../docs/reference/threads.md#commands-across-islands)).
Two cards are one run of 238 ranks: each island's last core is its link core 0, not a
rank, so island 0 holds ranks 0 to 118 and island 1 ranks 119 to 237 (`--links 4`: two
link cores an island, 236 ranks). Each island runs example 01's kernels, unchanged, and
a thread on work cluster 1 (`thread/across.c`).

The steps (`--steps`, all five by default):

| Step | What it does | What it checks |
|---|---|---|
| `nop` | NOPs to island 0's ranks and to island 1's, all submitted to island 0's ring only | every event done; the median µs a NOP of each |
| `map` | island 0's thread sends example 01's MAP to island 0's ranks, then to all its ranks | every tile of the last MAP exact against example 01's sums; the µs a MAP of each |
| `ask` | island 0's thread sends FOLD to island 1's thread (`KMX_THREAD(1, 1)`, number 4) with its result | `kmx_island_of(4) == 1`; every answer is island 1's (`x * 0x9E3779B1 + 1`); the µs a FOLD |
| `hang` | island 0's thread sends example 01's HANG to ranks 100 to 139; rank 130 (island 1) waits forever | the thread gets `KM_EVENT_TIMEOUT` whose auxiliary names island 1 (`KM_EVENT_ISLAND + 1`), then a MAP on all ranks is exact |
| `count` | example 01's COUNT (each core adds one to its count) on flat ranks 60 to 179: `--counts` times from the host (its split, each line marked a part), one EXCLUSIVE from the host, then `--counts` times from island 0's thread (crossed) | every rank of the group counts each command once (`--counts` + 1, then 2 x `--counts` + 1), the others 0; the µs a command each way |
| `host-hang` (only when named) | the thread's HANG as in `hang`, then the same HANG from the host to island 0's run alone (no part mark: it crosses), then a NOP on each island's own ring, example 01's MAP from the host on island 0's ranks and on island 1's ranks past the HANG group (140 on), and a MAP from island 0's thread on those (it crosses) | the host's HANG is given up naming island 1 and the run goes on: island 0's count of the host's give-ups grows by one, island 1's not, and the thread's HANG counts on neither; every command on the other ranks is answered. The hung cores' kernels end at their wait's own limit; the give-up resets and loads them again from the run's reset lists, so the stop's EXIT is answered on both islands and the step exits 0. Not a default step: each HANG waits out its deadline |

A command whose group reaches past the dispatcher's island is split by the island table:
the dispatcher runs its own part and sends the other part to the other island's
dispatcher as a command packet over link 0, sent by the island's link core 0; that dispatcher runs it as its own
command and sends DONE back. The ticket completes once every part is back, or gives the
command up at its deadline naming the island whose part did not come back.

## Files

| File | What it is |
|---|---|
| `thread/across.c` | the thread: MAPS, ASK, FOLD and HANG (`kmx_program`) |
| `islands_threads.py` | the programs, the run, the steps and every check; one JSON line |

## What it needs

A Komorebi install and two cards joined by a link, prepared as in
[Running on a card](../../docs/install.md#running-on-a-card).

```
python islands_threads.py --steps nop        # step 1 alone
python islands_threads.py                     # all five
```

`--cards` (0,1), `--links` (0: every link found), `--nop-jobs` (2,000 a round), `--rounds` (3), `--maps` (500),
`--asks` (500). `--reset-link` (with `count`): after the thread's COUNTs, the host resets
island 0's link core 0, the lines link's core (its connection closed at the router, the
core loaded again, the connection opened again), and the thread's COUNTs run once more,
each rank's count exact (3 x `--counts` + 1); on two p150a cards the reset took 95.0 ms,
opening again 0.24 ms, and the crossed COUNTs 10.97 µs a command before and after. The
script exits non-zero if a check fails.

`--cards 0 --islands n --steps nop`: card 0 split into n islands, the NOPs to island 1's
ranks crossing the wire to its dispatcher: 2.966 µs to island 0's ranks and 3.891 µs to
island 1's with 2 islands (two cards: 2.98 / 3.94 on 2026-11-14; 3.00 / 4.43 to 5.15
over four runs on 2026-11-16, all five steps), 3.034 / 5.067 with 3, 3.042 / 6.278
with 4 (3 and 2 commands across in flight then, 8 now with any number of islands). The other steps need X280
threads, which a split card does not run.

On two cards the NOP across and the thread's steps are above their 2026-11-14 times
(NOP across 3.94 µs; the thread's MAP to island 0's ranks 6.05, now 6.65 µs; its ask
10.61, now 11.96 µs). Per-command checks in the dispatcher's take and the bookkeeping of
a lost crossing were added since; they are the candidates, not shown.
