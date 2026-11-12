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

`merge_fed.py` runs the kernels on cores 0 to 118 with the test feeder on core 119
(no X280) and checks every job against numpy: bit-equal on integer inputs, by
relative RMS on uniform random ones. One JSON line.

```
python merge_fed.py --jobs 20 --count 8 --form both
python merge_fed.py --jobs 1000 --teams random --form random --out-pages 1024
python merge_fed.py --team-list 0:8,0:119 --form both --stall-member 3
python merge_fed.py --team-list 0:8,0:60,3:60,40:2,0:33 --form both --part 2
```

`--form both` runs every job twice, by owner and then by tree, on the same input, and
compares the two (`pairs_bit_equal`). `--stall-member M` adds two launches: one in
which member M of job 1 stalls instead of sending (it reports the timeout, the cores
that gave up and the ports still waiting with their missing counts), and a clean one
after it.

Measured on a Blackhole p150a at 1350 MHz, T = 64, records issued four at a time:

| Members | µs per job |
|---|---|
| 8 | 7.3 |
| 32 | 10.2 |
| 60 | 14.5 |
| 119 | 24.0 |

1,000 jobs on random teams with random forms: 1,000 checked, 0 wrong. 500 random
teams, each by both forms: 250 integer-input pairs, all bit-equal. Parts of 2 tiles
(`--part 2`, by owner up to 60 members): the five teams above by both forms, all exact
and bit-equal pairs, 16.2 µs per job.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels |
| `kernels/merge.h` | segment, buffers and record words shared by the kernels |
| `merge_fed.py` | builds the records, runs them under the test feeder, checks the outputs |

The calls are in `komorebi/device/km_port.hpp` (emit, arrive, take, reduce) and
`komorebi/device/km_group.hpp` (groups of a team, the two schedules).
