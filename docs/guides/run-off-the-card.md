---
title: Run off the card
summary: What runs without a Tenstorrent card, the two fast checks, and how many of the documented commands build to their end on the mock card.
tags:
  - guide
  - checks
---

# Run off the card

Use this to check a change without a card, or to see which documented commands you can
run on a machine without one. Off the card a script runs under `kmr prebuild`, on the
mock card (tt-metal's Blackhole P150 cluster descriptor): its kernels are built and its
loads checked as the card would, and its reads give zeros.

```
kmr prebuild --cards 0 examples/06_graph/graph.py      # one card
kmr prebuild --cards 0,1 examples/26_islands_sum/sum.py  # two cards
```

## The two fast checks

```
python -m komorebi.tools.host_check                 # off the card, under a minute
python -m komorebi.tools.card_check --prebuild      # no card: builds what the next one loads
python -m komorebi.tools.card_check                 # card 0, then cards 0 and 1
```

`host_check` needs no card. It runs the checks under `tests/host_checks` (stand-ins,
refusals and builds on the mock) at once, each under its own limit, and prints a line a
check and the total: 45 checks in about 20 s on a 96-core machine, once built. A
failing check's line says why (its exit, or its limit) and names its log, which the run
keeps in `build/host_check/<time>/`; a check whose run compiled kernels runs again (at
most three times) and only a run that compiled nothing is judged, so a cold cache
passes in one call (45 in about 115 s). `card_check` runs the examples' exactness
checks on the cards: 94 checks (75 on card 0, 19 on cards 0 and 1); their seconds are
measured by the next sizing run. A failing check's line names the cause (its exit, a
key that does not hold, a kernel compiled, a second open) and its log in
`build/card_check/`.
[Development](../development.md#fast-checks) describes both.

## The documented commands on the mock

Every command in a code block of a README or a docs page, run once through
`kmr prebuild` on the mock at the cards it names (main at 69f3cf44, 2026-11-18):

- **builds to its end**: the script ran to its own end, every load built;
- **stops by name**: it stopped with a line that names why: an input made by an earlier
  command or fetched by hand, an optional module, a packaged runtime that a checkout
  does not hold, a refusal the page itself shows, a kernel buffer the command gives
  that its kernels no longer fit, or reads that were zeros;
- **card only**: it says it needs the card, or it drives the X280 runtime, a mesh or the
  dispatcher harness, which the mock does not have;
- **stops without a name**: it stopped on the mock's zero reads (an index, a division,
  an empty array), exited 1 with no line saying why, or ran past 15 minutes;
- **not run**: listed only (pytest, installs, the docs build, the native build), or one
  of the repository's own lint commands, which need a git checkout.

| folder | commands | builds to its end | stops by name | card only | stops without a name | not run |
|---|---:|---:|---:|---:|---:|---:|
| `README.md` | 9 | 2 | 0 | 0 | 0 | 7 |
| `docs` (but the tutorial) | 21 | 2 | 0 | 0 | 0 | 19 |
| `docs/tutorial` | 18 | 10 | 0 | 0 | 6 | 2 |
| `examples` | 154 | 151 | 1 | 0 | 0 | 2 |
| `experiments` | 305 | 238 | 22 | 7 | 27 | 11 |
| `packaging` | 4 | 0 | 4 | 0 | 0 | 0 |
| `projects` | 84 | 64 | 6 | 0 | 6 | 8 |
| `tests` | 7 | 0 | 0 | 6 | 0 | 1 |
| all | 602 | 467 | 33 | 13 | 39 | 50 |

The 39 that stopped without a name were fixed after this count: each now runs to its
end off the card or stops with a line that says why. A command that stops off the card may still run to
its end on the card; one that builds to its end off the card has had its loads checked,
not its numbers.
