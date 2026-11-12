# 01 map/collect

Resident workers on every Tensix core but one, fed records by the test feeder on the
last core (no X280). A record names a segment and a team, cores `[first, first + count)`
of the core order; cores outside the team ignore it.

| Segment | Arguments | What a member does |
|---|---|---|
| `MAP` | a0 input, a1 tiles T, a2 output, a3 output page of member 0 | reads its share of the T input tiles, adds them into one tile (fp32 in DST), stores it at page a3 + member, reports |
| `STALL` | a0 member | member a0 waits for the stop word and does not report; the others report |
| `NOP` | none | reports |
| `EXIT` | none | the feeder sends it last to every core; it completes like any record, then every kernel leaves |

The host sums the members' tiles and compares them with numpy: every member's tile is
bit-exact on integer inputs (even jobs), and the total's relative RMS error is reported
on uniform random inputs (odd jobs).

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels: a switch on the segment inside the runtime's loops (`komorebi/device/km_worker.hpp`) |
| `kernels/mapcollect.h` | segments, buffers and record arguments shared by the kernels |
| `mapcollect.py` | builds the records, runs them through the feeder, checks the outputs, prints one JSON line |

```
python mapcollect.py --jobs 1000 --teams random --depth 4 --out-tensors 32
python mapcollect.py --jobs 200 --seg nop --count 60 --depth 1 --trace
python mapcollect.py --jobs 20 --teams random --stall 7 --deadline 0.02
```

Completion: every member increments a counter at the team's first member, which
pushes the sequence number to the feeder's acknowledgement word of the slot. With
`--rows` (compiled in as `KM_ROW_LEADERS`), members report to the first team member of
their grid row and row leaders to the first member; both levels use the same atomic
increment.

`--teams` is `fixed` (`--first`, `--count`), `random` (first and count uniform) or
`pair` (two disjoint halves, alternating). `--depth` is the number of ring slots the
feeder uses and the workers scan. Outputs go to a ring of `--out-tensors` tensors of
`--out-pages` tiles; only jobs whose pages no later job overwrote are checked, and
reading back runs at about 1 MB/s under slow dispatch. `--trace` adds the per-record
phases (microseconds after the feeder began issuing the record): first and last member
took it, last member's outputs written, leader pushed the completion, feeder saw it.
With `--stall`, the feeder's deadline ends the run: it writes the stop word, every
kernel returns, and the JSON names the timed-out slot and sequence, the team members
that did not report, and the cores that waited in the stall or in a completion.

Runs in slow dispatch on one Blackhole card (`TT_METAL_SLOW_DISPATCH_MODE=1`).
