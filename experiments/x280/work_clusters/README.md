# X280 work clusters 1 to 3

Questions threads on all three work clusters depend on, asked on a card before the
host and the dispatcher served them: which DRAM bank each cluster reaches through
its cache, whether clusters on one bank share its rate, what moving data to another bank
costs each way, and how fast one cluster can wake another.

## Results

| Question | Answer |
|---|---|
| Which bank does each cluster reach? | Cluster 1 (NoC 8,9) bank 6; cluster 2 (8,5) and cluster 3 (8,7) bank 7. The start-up bank check passes on each, and an image of the work cluster example runs on 2 or 3 exactly as on 1 (204 rounds, the same sum, 10.71 ms against 10.8) |
| Do clusters on one bank share its rate? | No: the limit is per cluster. Each reads and writes its bank at about 1.2 GB/s with four cores, alone or with the others; clusters 2 and 3 together on bank 7 move 2.3 GB/s, all three 3.5 GB/s |
| What does crossing to another bank cost? | Pushing (stores through a 2 MiB NoC window, posted): 2.28 GB/s. Pulling (loads through the window): 0.024 GB/s, 95 times slower. Data crossing banks is pushed by its producer |
| How fast does a cluster wake another? | 0.66 µs round trip with a flag in the peer's LIM written through a window; 0.96 to 1.04 µs with a flag pushed into the peer's bank and read after a take; 1.18 to 1.22 µs between 2 and 3 through their shared bank with give and take. A `kmx_parallel` call on one cluster: 0.53 µs |
| Flush and invalidate | An invalidate sweep runs at 4.57 GB/s of region, a flush at 4.28 to 4.42, on every cluster |

Every case's data was checked word for word (sums, exchanges of 1 MiB both ways,
pushes): all exact. One Blackhole p150a, slow dispatch, the X280 PLL at the session's
1,000 MHz, the host's load average between 4.4 and 9.8.

## The probe

`x280/probe.c` is one kmx thread; word 0 picks its mode. `clusters.py` writes each
cluster's regions, loads a persistent grid and the dispatcher, starts the probe on
each cluster asked for through the runtime port, writes a go line once every cluster
has started (so clusters that run together start together) and prints one JSON line.

```sh
python clusters.py sum --clusters 1 2 3 --bytes 1048576
python clusters.py rates --clusters 2 3 --bytes 8388608 --reps 4
python clusters.py push --cluster 2 --bank 6 --bytes 4194304 --pull 262144
python clusters.py ping --pair 2 3 --kind shared --rounds 1000 --xbytes 1048576
python clusters.py parallel --cluster 1 --calls 10000
```

- `sum`: each cluster sums a region the host filled with its own random words and
  checks `kmx_cluster()`.
- `rates`: per repetition an invalidate, four cores reading their quarter of the region
  with vector loads, then writing it, then a flush; the first repetition is dropped.
- `push`: the cluster stores a region into the other bank through a window (four
  cores, 64-byte stores), then loads part of it back on core 0; the host checks both.
- `ping`: two clusters exchange 1 MiB, then pass a flag back and forth: `shared`
  (both on bank 7, a region shared with flush and invalidate), `dram` (a flag pushed into the
  peer's bank), `lim` (a flag in the peer's LIM).
- `parallel`: empty `kmx_parallel` calls.

## Rates

Four cores per cluster, 8 MiB per cluster, repetitions 2 to 4 of 4.

| Clusters at once | read GB/s per cluster | write GB/s per cluster | read GB/s in all |
|---|---|---|---|
| 1 alone (bank 6) | 1.15-1.20 | 1.22-1.23 | 1.2 |
| 2 alone (bank 7) | 1.14-1.22 | 1.18-1.22 | 1.2 |
| 3 alone (bank 7) | 1.16-1.26 | 1.20-1.21 | 1.2 |
| 2+3 (both bank 7) | 1.08-1.25 | 1.15-1.20 | 2.3 |
| 1+2 (banks 6, 7) | 1.13-1.28 | 1.21-1.22 | 2.4 |
| 1+2+3 | 1.12-1.24 | 1.14-1.23 | 3.5 |
| 1 alone, 16 MiB | 1.19-1.25 | 1.22-1.23 | 1.2 |

Clusters 2 and 3 together move 16 MiB of bank 7 in the time cluster 1 alone takes
for 8 MiB of bank 6: a second cluster on a bank adds its rate.

## Across banks and wakes

| Case | Result |
|---|---|
| push, 4 MiB, 2 -> bank 6, 3 -> bank 6, 1 -> bank 7 | 2.28 GB/s each, exact |
| pull, 256 KiB, core 0, 8-byte loads | 0.024 GB/s, exact |
| `kmx_parallel`, empty, cluster 1 or 2 | 0.533 µs per call |
| wake round trip, flag in LIM through a window (1-2, 2-3, 1-3) | 0.66-0.67 µs |
| wake round trip, flag pushed into DRAM, read after an invalidate (1-2, 1-3) | 0.96-1.04 µs |
| wake round trip, 2 and 3 through shared bank 7 | 1.18-1.22 µs |

A wake must be acknowledged before the next one is sent: a first version of `ping`
sent the start flag and round 1 back to back, the peer missed the flag, and both ended
at their deadline (with a timeout, the clusters clean).
