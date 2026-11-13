# 10 X280 ports

X280 cores and Tensix cores talk to each other without the dispatch kernel in the
path ([X280 ports](../../docs/threads.md#x280-ports)). The thread issues one record per
round to the grid's Tensix cores and serves them on every X280 core of every cluster
until the record's event:

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
returned code 0 without a library error, every round's record completed and every hash
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

## How it ends

`kmx_port_run(&r, serve, round, &ev)` on X280 core 0 of the lead issues the record and
runs `serve` on every X280 core as `kmx_par_all` does. Each loops while
`kmx_port_open()` returns 1; X280 core 0 of the lead looks for the record's event in
its ring, and when it comes, every next look returns 0 and `serve` returns.
`kmx_port_run` returns the event once all have: nothing is stored into a Tensix core
after it, so the next round's tells cannot land in this one. A record that times out
ends the serve the same way (its event says `KM_EV_TIMEOUT`).

## The thread

| Words | `rounds, cores, asks` (`cores` 0: every Tensix core) |
|---|---|
| Region | `out`: a 64-byte line per Tensix core: the hash of the last round, rounds told, asks, their cycles (64 bits), the largest, `1 + (cluster << 2 \| cpu)` (`cpu` the X280 core), in the bank of the cluster that serves it |
| Codes | 0 done; 1 bad words or region; 2 a run was late or stopped |
| Result words | rounds done, rounds timed out, the rounds' ticks (64 bits), the best and the worst round in ticks (50 per µs) |

## Files

| File | What it is |
|---|---|
| `thread/ports.c` | the thread: the rounds, the serve on every X280 core, the lines |
| `kernels/reader.cpp` | the grid's reader: owns the Tensix core's port, waits for the salt, asks, tells the hash |
| `kernels/writer.cpp`, `kernels/compute.cpp` | the grid's other kernels, which do nothing |
| `x280_ports.h` | what the reader and the thread share: the salt, the values, the hash, the line |
| `x280_ports.py` | the host script: the program, the ports, the numpy check |
