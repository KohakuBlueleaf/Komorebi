# 02 merge tree

A group of threads adds up T input tiles: each thread adds its own partition, then the
group merges the parts into one tile, either at one owner or along a tree. Each part
goes from one core to one other core: a value sent to one receiver.

## What each thread does

Each of the n threads of the group has an index `j`, 0 to n - 1.

1. It adds up its own partition of the T tiles.
2. With `form == owner`, thread 0 waits for the n - 1 other parts, one slot for each
   sender, and adds them in rank order, not in arrival order. Every other thread sends
   its part to thread 0 and ends.
3. With the tree form, the threads merge in rounds. In round r, a thread whose index is a
   multiple of 2^(r+1) adds the part of its partner `j + 2^r`, if that partner exists; any
   other thread sends its part to `j - 2^r` and ends.
4. Thread 0 writes the sum of all the parts to the output.

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

A job is one command, `MERGE`, sent to the group's cores; every core of the group runs
it in its three kernels.

| Kernel | What it does |
|---|---|
| `reader.cpp` | reads its partition of the input tiles; then, for each arrival this core receives, `km::arrive`: waits until the port has counted its arrivals and hands the whole arrival buffer to compute (`km::reduce_arrive` does the rounds) |
| `compute.cpp` | adds its tiles into P float32 tiles in DST; `km::reduce` takes each arrival and calls the program's combine step for its slots (here: copy the page, add it); packs the result |
| `writer.cpp` | rank 0 writes the result to `output[first_page]` to `output[first_page + P - 1]`; every other core of the group emits its part: put into the receiver's arrival slot, write flush, bump the receiver's port, atomic flush (`km::reduce_emit`) |

| Command word | Meaning |
|---|---|
| a0, a1, a6 | `input_address`, `tiles` (T), `first_tile` |
| a2, a3 | `output_address`, `first_page` |
| a4 | `form`: 0 by owner, 1 by tree |
| a5 | `stalled_rank` (a test: this rank stalls instead of sending; none: 0xFFFFFFFF) |
| a7 | `part_tiles` P (1 to 3): input tile t is added into tile t mod P of the part |

- **Ports and slots.** By owner, rank k sends into slot k - 1 of rank 0 and bumps
  its port 0, and rank 0 waits for n - 1. By tree, the core of round r sends into
  slot r and bumps port r, and its receiver waits for 1. Who sends where is computed
  from the command and the rank: nothing is queued and nothing is addressed by content.
- **The arrival buffer** (`kCbArrive`, 118 float32 tiles, P to a slot: by owner, a slot
  for every other core of a group of 118 / P + 1) is passed to compute whole, so its slot addresses are the same on every
  core and never drift. Compute reads the pages it needs by index.
- **Commands carry `KM_FLAG_EXCLUSIVE`**: two commands whose groups overlap are never in flight
  together, so nothing of one command can land in the next one's slots or ports.
- **A core that does not send.** Every wait has a timeout. The receiver's wait on the
  port gives up first and its state names the port; the port keeps the count it
  wanted, so the host reads the missing count. The feeder's deadline then stops the
  run, every kernel leaves, and the next launch runs without a reset.

## Running it

`merge.py` loads the program with `komorebi.Session` and submits one `MERGE` command per
job with `Run.submit(..., exclusive=True, result=True)`: the commands reach the grid
through the dispatcher on X280 cluster 0. Rank 0 returns the total's checksum
(its float32 words summed in 32 bits) as the command's result word, which arrives as
`Event.result`; it sums only when the command asks for a result (`KM_FLAG_RESULT`), since
the pass over the total costs about 7.6 µs per output tile. The driver checks every total against numpy (bit-equal on integer
inputs, by relative RMS on uniform random ones) and every result word against the
checksum of numpy's total (integer inputs) or of the total read back (random ones).
One JSON line, with `ok`: every job and result word right, no core given up, no port
left waiting and every core ended (with `--stall-rank`: the stalled load ends in the
deadline and the host's timeout, with a core given up, and the load after it is
clean); `merge.py` and `merge_fed.py` exit
non-zero without it. Setup is that of
[01_mapcollect](../01_mapcollect/README.md#what-it-needs).

```
python merge.py --jobs 20 --count 8 --form both
python merge.py --jobs 1000 --groups random --form random --out-pages 1024
python merge.py --group-list 0:8,0:119 --form both --stall-rank 3
python merge.py --group-list 0:8,0:60,3:60,40:2,0:33 --form both --part 2
```

`--form both` runs every job twice, by owner and then by tree, on the same input, and
compares the two (`pairs_bit_equal`). `--stall-rank M` adds two loads of the
program in the same session, of the first four jobs: in the first, rank M of job 1
stalls instead of sending; the host raises `CommandTimeout` naming the handle (the last
command, once the stalled job has been given up) and the cores that did not report, and the driver reads the cores that gave up (a reader
waiting on a port names it) and the ports still waiting with their missing counts.
The second load runs the same jobs clean.

A group is at most 119 cores by owner (one arrival slot per other rank); by tree it
may cover the whole grid. `merge_fed.py` takes the same options and runs the jobs
under the test feeder on core 119 instead (no X280; cores 0 to 118); its commands ask
for no result word, so rank 0 computes no checksum.

Measured on a Blackhole p150a (card 1, 2026-11-13) at 1350 MHz, T = 64, commands submitted
four at a time, `--form both`, 100 jobs each run by both forms:

```
python merge_fed.py --jobs 100 --count N        # the test feeder
python merge.py --jobs 100 --count N            # through a session
```

| Cores | µs per job, test feeder | µs per job, session (host's clock, with the checksum) |
|---|---|---|
| 8 | 7.3 | 36.7 |
| 32 | 10.2 | |
| 60 | 14.5 | |
| 119 | 24.0 | 51.8 |

The session's times are the host's, from the first submit to the last event, through
the dispatcher (X280 at 1000 MHz), and every command asks for its result word: the checksum is about 7.6 µs of every
job (the test feeder's 8 cores took 15.0 µs per job while its commands summed too).

1,000 jobs on random groups with random forms: 1,000 checked, 0 wrong. 500 random
groups, each by both forms: 250 integer-input pairs, all bit-equal. Parts of 2 tiles
(`--part 2`, by owner up to 60 cores): the five groups above by both forms, all exact
and bit-equal pairs, 16.2 µs per job.

## Over two cards

`islands.py` runs the same program and kernels on two cards joined by a link
(`Session(cards=[0, 1], links=1)`, 238 ranks: each island's last core is its link core).
A MERGE whose group spans both islands folds inside each island first, as a group of its
own (by owner or by tree), and each island's first rank then sends its part once to the
group's rank 0, one more arrival at rank 0 (by owner: the slot after its island's; by
tree: the round after its island's). The part crosses through the relay: the sender asks
its island's link core 0, which sends the pages over link 0 to the far link core 0, which
writes them into rank 0's arrival slot and bumps its port. The kernels do not change.
The commands go to island 0's dispatcher, which sends island 1's part across. The link
cores' services make the kernels larger than the runtime's default buffer; a session over
two cards opens with one that holds them.

```
python islands.py --rounds 20
```

Cards 0 and 1 (Blackhole p150a, one link), T = 512 integer tiles, P = 1, 20 commands a
case, waited one at a time, each asking for rank 0's result word (one more pass over the
total there; the script now leaves it out); the median from submit to the event on the
host:

| Group | Form | µs a round | Total |
|---|---|---|---|
| all 238 ranks | tree | 172.6 | exact |
| 59 ranks of each island (60 to 177) | owner | 195.1 | exact |
| island 0's 119 ranks | tree | 162.8 | exact |
| island 0's 119 ranks | owner | 188.1 | exact |

A group over both islands by owner needs an arrival slot for every other rank of rank 0's
island and one for each other island: at most 118 slots (`kArrivePages`), so the whole of
both islands goes by tree.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels |
| `kernels/merge.h` | handler, buffers and command words shared by the kernels |
| `merge.py` | builds the commands, runs them through a session, checks the outputs and result words |
| `merge_fed.py` | the same jobs under the test feeder |
| `islands.py` | MERGE over two cards, the kernels unchanged, checked against numpy |

The calls are in `komorebi/device/km_port.hpp` (emit, arrive, take, reduce) and
`komorebi/device/km_group.hpp` (groups of a group, the two schedules).
