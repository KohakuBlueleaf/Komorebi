# 10 X280 ports

X280 cores and Tensix cores talk to each other without the dispatcher in the
path ([X280 ports](../../docs/threads.md#x280-ports)). The thread submits one command per
round to the grid's Tensix cores and serves them on every X280 core of every cluster
until the command's event:

1. each X280 core tells each Tensix core it serves a salt for the round (a tell);
2. the Tensix core asks its X280 core for 100 values, one at a time, and hashes them
   (ask and answer: the X280 core computes each value from the ask and the core);
3. the Tensix core tells its X280 core the hash and the cycles its asks took (a tell);
   the X280 core keeps them in a line per Tensix core of region `out`.

The host checks every hash against numpy and turns the cycles into the round trip of
one ask.

    python x280_ports.py [--clusters 1 2 3] [--cores 0] [--rounds 10] [--asks 100]

`--cores 0` is every Tensix core of the grid; the ports deal them over the threads' X280
cores in grid order (`prog.ports()`): ten to an X280 core with three clusters, thirty
with one. Prints one JSON line: whether every hash is exact (`exact`), the mean and the
largest round trip of one ask (`ask_us_mean`, `ask_us_max`), a round's time on the
lead's timer (`round_us_mean`, `round_us_best`), and the asks answered a second
(`asks_per_s`). `ok` is false, and the script exits non-zero, unless every thread
returned code 0 without a library error, every round's command completed and every hash
is exact.

## Numbers

A Blackhole p150a, AICLK 1350 MHz, X280 at the session's 1000 MHz, 10 rounds of 100
asks per Tensix core, every hash exact:

| Tensix cores | X280 cores | Tensix cores each | One ask, mean µs | largest | A round, µs | Asks a second |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1.04 | 1.53 | 112 | 0.90 M |
| 120 | 4 (`--clusters 1`) | 30 | 26.6 | 27.4 | 2,720 | 4.4 M |
| 120 | 12 | 10 | 8.93 | 9.56 | 935 | 12.8 M |

An X280 core answers an ask in about 0.9 µs and serves its Tensix cores in turn, so the
round trip grows with how many it serves; twelve X280 cores give three times the rate
of four. A round includes the salts, the hashes told back and the end of the serve.

## The pull

    python x280_ports.py --pull [--x280-cores 12] [--units 31000] [--per 16]
        [--base 20250] [--extra 16200] [--centre 9300] [--half 6200]

A pass over `--units` units on every Tensix core of the grid, unit `u` busy for
`HX_COST` cycles: `--base`, plus up to `--extra` in a triangle of half-width `--half`
about unit `--centre` (by default 15 µs, up to 27 µs, a fifth of the units heavier),
the shape of a frame whose cost gathers in one part of the screen. The thread
(`thread/pull.c`) runs five passes, each a command, then two report commands in which
every Tensix core tells its numbers:

| Pass | How the units are dealt |
|---|---|
| `equal` | equal contiguous ranges, core `i` the `i`-th (`km::partition`) |
| `pull` | ranges of `--per` units pulled (`kmx_pull_run`, `KMX_PULL_COUNT`; `port.next`) |
| `log` | the same, and each core's ranges logged in region `log` (`KMX_PULL_LOG`) |
| `replay` | each core gets the ranges the log holds for it (`KMX_PULL_REPLAY`) |
| `pick` | pulled, but `pick` deals range `k` of core `c`: `c + k * cores` |

`--x280-cores` 1, 4 or 12 serve the Tensix cores (one X280 core, cluster 1, or all
three clusters). Prints one JSON line with the model's slowest core over mean for equal
ranges, and per pass: its time on the lead's timer (`pass_us`), the slowest core's
busy cycles over the mean (`slowest_over_mean`), the pulls, a core's wait in
`next()` a pull (`wait_us_per_pull`), and `exact`: every unit ran once (count and a
sum of marks per unit), each `equal` core ran its own range, the `replay` cores ran
what the `log` cores ran, core by core, and each `pick` core got its ranges.

### Numbers

The same card, defaults (31,000 units, 17.4 µs mean, 278 µs a range, 1,938 ranges):

| X280 cores | equal: pass, slowest/mean | pull | log | replay | pick | wait a pull |
|---|---|---|---|---|---|---|
| 12 | 6,960 µs, 1.544 | 4,728 µs, 1.045 | 4,739 µs | 4,732 µs | 4,736 µs | 0.68 µs |
| 4 | 6,959 µs, 1.544 | 4,755 µs, 1.046 | 4,769 µs | 4,748 µs | 4,743 µs | 0.96 µs |
| 1 | 6,959 µs, 1.544 | 4,774 µs, 1.046 | 4,832 µs | 4,856 µs | 4,771 µs | 2.37 µs |

Perfect balance would take 4,495 µs; the pull is 5 % above it, the end of the last
ranges. **When the pull loses**: units of 1 to 2 µs (`--base 1350 --extra 1080`), equal
ranges take 482 µs:

| `--per` (a range) | 12 X280 cores | 1 X280 core |
|---|---|---|
| 1 (1.2 µs) | 1,510 µs | 16,473 µs |
| 4 (4.6 µs) | 402 µs | 4,185 µs |
| 16 (18.6 µs) | 351 µs | 1,140 µs |

A range must take longer than a round trip (9 µs with ten Tensix cores on an X280 core,
60 µs with 120 on one), else the cores wait for their X280 core, which answers about one
ask a microsecond.

## How it ends

`kmx_port_run(&r, serve, round, &ev)` on X280 core 0 of the lead submits the command and
runs `serve` on every X280 core as `kmx_par_all` does. Each loops while
`kmx_port_open()` returns 1; X280 core 0 of the lead looks for the record's event in
its ring, and when it comes, every next look returns 0 and `serve` returns.
`kmx_port_run` returns the event once all have: nothing is stored into a Tensix core
after it, so the next round's tells cannot land in this one. A command that times out
ends the serve the same way (its event says `KM_EV_TIMEOUT`).

## The thread

| Words | `rounds, cores, asks` (`cores` 0: every Tensix core) |
|---|---|
| Region | `out`: a 64-byte line per Tensix core: the hash of the last round, rounds told, asks, their cycles (64 bits), the largest, `1 + (cluster << 2 \| cpu)` (`cpu` the X280 core), in the bank of the cluster that serves it |
| Codes | 0 done; 1 bad words or region; 2 a run was late or stopped |
| Result words | rounds done, rounds timed out, the rounds' ticks (64 bits), the best and the worst round in ticks (50 per µs) |

The `--pull` thread: words `units, per, base, extra, centre, half`; regions `out` (a
32-byte line per kind of pass and Tensix core: units, busy cycles, span, the sum of
marks, pulls, cycles waited, `1 + (cluster << 2 \| cpu)`) and `log`; codes as above
(1 also for a log that ran out of room); result words: each pass's ticks.

## Files

| File | What it is |
|---|---|
| `thread/ports.c` | the thread: the rounds, the serve on every X280 core, the lines |
| `thread/pull.c` | the `--pull` thread: the five passes, the reports, the lines |
| `kernels/reader.cpp` | the grid's reader: owns the Tensix core's port, waits for the salt, asks, tells the hash; runs a pass's units and reports |
| `kernels/writer.cpp`, `kernels/compute.cpp` | the grid's other kernels, which do nothing |
| `x280_ports.h` | what the reader and the thread share: the salt, the values, the hash, the line |
| `x280_ports.py` | the host script: the program, the ports, the numpy check |
