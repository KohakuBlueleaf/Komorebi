# 02 merge tree

A team of threads adds up T input tiles: each thread adds its own share, then the
team merges the parts into one tile, either at one owner or along a tree. Each part
crosses from one core to one other core: a value emitted to one receiver.

## The thread program

```
thread merge(j, n, T, form):                  # one per core of the team
    part = sum(x[tile] for tile in share(T, j, n))
    if form == owner:
        if j == 0:
            parts = wait(n - 1)                  # one slot per sender
            for k in 1 .. n - 1:
                part += parts[k]                 # in rank order, not arrival order
        else:
            emit part -> thread(0)
            return
    else:                                        # tree
        for r in 0 .. levels(n) - 1:
            if j % 2^(r+1) == 0:
                if j + 2^r < n:
                    part += wait(1)              # my partner of round r
            else:
                emit part -> thread(j - 2^r)
                return
    write out[job] = part                        # only thread 0 arrives here
```

```
 tree, 8 threads:
 round 0:   t1 -> t0     t3 -> t2     t5 -> t4     t7 -> t6
 round 1:   t2 -> t0                  t6 -> t4
 round 2:   t4 -> t0
```

Both forms add the parts in a fixed order given by the ranks, never by arrival
order, so a job gives the same bits every time. On exact-valued inputs both forms
give the same bits as each other.

## On the card

A job is one record, `MERGE`, sent to the team's cores; every core of the team runs
it in its three kernels.

| Kernel | What it does |
|---|---|
| `reader.cpp` | reads its share of the input tiles; then, for each hand-over this member receives, `km::arrive`: waits until the port has counted its arrivals and hands the whole arrival buffer to compute (`km::reduce_arrive` does the rounds) |
| `compute.cpp` | adds its tiles into P float32 tiles in DST; `km::reduce` takes each hand-over and calls the program's combine step for its slots (here: copy the page, add it); packs the result |
| `writer.cpp` | member 0 writes the result to `out[a3]` to `out[a3 + P - 1]`; every other member emits its part: put into the receiver's arrival slot, write flush, bump the receiver's port, atomic flush (`km::reduce_emit`) |

| Record word | Meaning |
|---|---|
| a0, a1, a6 | input tensor, tiles T, first input tile |
| a2, a3 | output tensor, output page |
| a4 | form: 0 by owner, 1 by tree |
| a5 | a test: this member stalls instead of sending (none: 0xFFFFFFFF) |
| a7 | tiles per part P (1 to 3): input tile t is added into tile t mod P of the part |

- **Ports and slots.** By owner, member k sends into slot k - 1 of member 0 and bumps
  its port 0, and member 0 waits for n - 1. By tree, the member of round r sends into
  slot r and bumps port r, and its receiver waits for 1. Who sends where is computed
  from the record and the rank: nothing is queued and nothing is addressed by content.
- **The arrival buffer** (`kCbArrive`, 118 float32 tiles, P to a slot: by owner, a slot
  for every other member of a team of 118 / P + 1) is handed over whole, so its slot addresses are the same on every
  core and never drift. Compute reads the pages it needs by index.
- **Records carry `KM_F_EXCL`**: two records whose teams overlap are never in flight
  together, so nothing of one record can land in the next one's slots or ports.
- **A member that does not send.** Every wait is bounded. The receiver's wait on the
  port gives up first and its state names the port; the port keeps the count it
  wanted, so the host reads the missing count. The feeder's deadline then stops the
  run, every kernel leaves, and the next launch runs without a reset.

## Running it

`merge.py` loads the program with `komorebi.Session` and issues one `MERGE` record per
job with `Run.issue(..., exclusive=True, result=True)`: the records reach the grid
through the dispatch kernel on X280 cluster 0. Member 0 returns the total's checksum
(its float32 words summed in 32 bits) as the record's result word, which arrives as
`Event.result`; it sums only when the record asks for a result (`KM_F_RESULT`), since
the pass over the total costs about 7.6 µs per output tile. The driver checks every total against numpy (bit-equal on integer
inputs, by relative RMS on uniform random ones) and every result word against the
checksum of numpy's total (integer inputs) or of the total read back (random ones).
One JSON line, with `ok`: every job and result word right, no core given up, no port
left waiting and every core ended (with `--stall-member`: the stalled load ends in the
deadline on job 1 and the load after it is clean); `merge.py` and `merge_fed.py` exit
non-zero without it. Setup is that of
[01_mapcollect](../01_mapcollect/README.md#what-it-needs).

```
python merge.py --jobs 20 --count 8 --form both
python merge.py --jobs 1000 --teams random --form random --out-pages 1024
python merge.py --team-list 0:8,0:119 --form both --stall-member 3
python merge.py --team-list 0:8,0:60,3:60,40:2,0:33 --form both --part 2
```

`--form both` runs every job twice, by owner and then by tree, on the same input, and
compares the two (`pairs_bit_equal`). `--stall-member M` adds two loads of the
program in the same session, of the first four jobs: in the first, member M of job 1
stalls instead of sending; the host raises `RecordTimeout` naming the ticket and the
members that did not report, and the driver reads the cores that gave up (a reader
waiting on a port names it) and the ports still waiting with their missing counts.
The second load runs the same jobs clean.

A team is at most 119 cores by owner (one arrival slot per other member); by tree it
may cover the whole grid. `merge_fed.py` takes the same options and runs the jobs
under the test feeder on core 119 instead (no X280; cores 0 to 118); its records ask
for no result word, so member 0 computes no checksum.

Measured on a Blackhole p150a (card 1, 2026-11-13) at 1350 MHz, T = 64, records issued
four at a time, `--form both`, 100 jobs each run by both forms:

```
python merge_fed.py --jobs 100 --count N        # the test feeder
python merge.py --jobs 100 --count N            # through a session
```

| Members | µs per job, test feeder | µs per job, session (host's clock, with the checksum) |
|---|---|---|
| 8 | 7.3 | 36.7 |
| 32 | 10.2 | |
| 60 | 14.5 | |
| 119 | 24.0 | 51.8 |

The session's times are the host's, from the first issue to the last event, through
the dispatch kernel (X280 at 1000 MHz), and every record asks for its result word: the checksum is about 7.6 µs of every
job (the test feeder's 8 members took 15.0 µs per job while its records summed too).

1,000 jobs on random teams with random forms: 1,000 checked, 0 wrong. 500 random
teams, each by both forms: 250 integer-input pairs, all bit-equal. Parts of 2 tiles
(`--part 2`, by owner up to 60 members): the five teams above by both forms, all exact
and bit-equal pairs, 16.2 µs per job.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels |
| `kernels/merge.h` | segment, buffers and record words shared by the kernels |
| `merge.py` | builds the records, runs them through a session, checks the outputs and result words |
| `merge_fed.py` | the same jobs under the test feeder |

The calls are in `komorebi/device/km_port.hpp` (emit, arrive, take, reduce) and
`komorebi/device/km_group.hpp` (groups of a team, the two schedules).
