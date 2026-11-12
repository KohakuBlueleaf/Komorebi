# X280 record multicast against lane broadcasts

The X280 sends every work record as a 64-byte multicast on NoC0 to all Tensix cores
(see [landing](../landing/README.md)). Programs also want to broadcast inside the grid:
one core of a lane (a run of cores in one grid column) multicasts a block down the
column, then a VALID word, and every receiver signals back. The Blackhole NoC
documentation warns that a broadcast with path reservation (`NOC_CMD_PATH_RESERVE`,
which tt-metal sets on every multicast) "can lose a flit and hang the NoC if its
reservation is refused at a router two or more hops past the initiating tile's router,
which happens when another broadcast ... holds the virtual channel that the reservation
needs there" (tt-isa-documentation, `BlackholeA0/NoC/MemoryMap.md`). The question: can
the two run at the same time on NoC0, and if not, which arrangement is clean?

## What runs

Every core runs two data-movement kernels, one per NoC (the two kernels of a core
cannot share a NoC in tt-metal's dedicated mode).

| part | file |
|---|---|
| X280 image (hart 0): the landing image, its two multicast windows moved to slots 16 and 17 | `x280/record_seq.c` |
| Record kernel: accepts every record (echo words, protocol A of the landing experiment), checks every payload word, completes it through the column leaders, then raises a records-done word | `kernels/records.cpp` |
| Lane kernel: stop-and-wait block broadcast down its lane, every word checked | `kernels/lane.cpp` |
| Lane kernel, stream mode: a fixed body multicast back to back with a window | `kernels/stream.cpp` |
| Lane buffer layout, block contents, counter indices | `kernels/lane.h` |
| Record layout and payload function (unchanged copy from the landing experiment) | `kernels/landing.h` |
| Device template of tt-x280-dispatch (unchanged copy, see the landing README) | `kernels/ttx_dispatch.hpp` |
| Host module: the landing module under another name, plus translated coordinates | `host/collision_x280.cpp` |
| Driver: builds the program, runs, closes the X280 runtime, reads both kernels' counters | `collision.py` |

**The lane.** The grid column (10 cores) is cut into lanes of L = 2, 5 or 10 cores.
For k = 1, 2, ... the sender waits until every receiver stamped k - 1, writes block k
(B bytes; word w is `base(k, tag) + w * c`, so every word depends on the block, its
index and the run's random tag), multicasts it to the receivers
(`noc_async_write_multicast`), barriers, writes VALID k into its own word and
multicasts it (`noc_semaphore_set_multicast`), barriers. Each receiver waits for
VALID k, checks every word of the block and writes its stamp k into its own word at
the sender (`noc_async_write` of 4 bytes), barrier. On NoC0 (which runs +y) the
sender is the lane's top core; on NoC1 (which runs -y) it is the bottom core and the
rectangle's corners are swapped.

**Stream mode.** The long multicast is the case the documented defect is about (it
holds its path longest, and a 64-byte record may need a channel it holds), but in the
mode above the sender spends most of each 2 KiB or 24 KiB block writing words, not
multicasting. In stream mode (`--stream`) the sender fills the bodies of W window slots
once; for block k it waits until every receiver stamped at least k - W, writes block
k's 64-byte header (its stamp and its number) into slot k mod W, multicasts the slot,
barriers (which also acknowledges the previous VALID) and multicasts VALID k without
waiting. A receiver waits for VALID >= k, checks the header of every block, and checks
the whole body of blocks 1..W and of every N-th block; the body of a slot is
overwritten with its words inverted (poisoned) before such a block lands in it, so
every word of a checked block must have arrived. The sender stamps the cycles inside
multicast plus barrier; their share of the sender's wall time is the duty below.
Stop-and-wait (W = 1) kept the sender inside multicast plus barrier only 32% of the time
at 2 KiB and 52% at 24 KiB (lanes alone, 20,000 blocks, full check every 1,024
blocks); W = 4 raised that to 54-58% and 69-71%, and W = 8 added nothing, so the runs
use W = 4. N is 512 at 2 KiB and 2,048 at 24 KiB, which keeps the receivers' full checks
and poisoning under 6% of their time (measured, below).

**Overlap.** In the runs with records the lanes start with the program and stop when
the record kernel of the sender's core has completed its last record (the sender then
multicasts a stop word in place of VALID). The X280 starts issuing only once every
core has checked in, so every lane broadcasts during the whole record stream.

**Bounded waits.** Every wait in both kernels gives up after one second and leaves a
where code; a lane kernel that gives up reads its peer's state words over the NoC
(also bounded) and reports them. tt-metal's write barrier is not bounded, so before
each barrier the kernel stores "about to barrier, block k" in its state words. The
driver sets `TT_METAL_OPERATION_TIMEOUT_SECONDS=20`, so a core that never finishes
makes the host's wait throw with the core named; the run then prints its JSON line
with the error and exits with code 7.

**Counts per run**, summed over the 120 cores: records torn (a payload word wrong in an
accepted record), records lost (a bounded wait gave up), blocks and words wrong at the
receivers, stamps missing at the senders, bounded waits that gave up (core, where
code, block, the peer's phase), receivers whose block count differs from their
sender's. Times: X280 `mtime` per record (as in the landing experiment) and sender
wall-clock cycles per block.

## Build and run

As the landing experiment (`build.sh` with `TT_METAL_TREE` and `TTX_DISPATCH`), then

```
python collision.py --records N [--lane L --bytes B] [--blocks K] [--until] \
    [--lane-noc 0|1] [--columns all|x,...] [--stream --window W --check-every N]
```

`--records 0` leaves the X280 idle after start-up (it waits for the check-in, then
ends); `--lane 0` runs no lanes; without `--until` the lanes run K blocks;
`--lane-noc 0` puts the lane kernel on RISCV_1 with NoC0 and the record kernel on
RISCV_0 with NoC1, `--lane-noc 1` the other way round. `run.sh <tag> <args>` wraps one
run in the host's hang watcher (`COLLISION_ENV`, `HANGWATCH`, `LOCK_DIR`, optional
`CARD`, `LIMIT`), like the landing experiment's.

## Results

p150a, 120 Tensix cores (12 x 10, two multicast panels), AICLK 1350 MHz, X280 at
1750 MHz, slow dispatch, posted multicast windows, record pacing depth 4, measured
with the code in this folder. 153 runs, each in its own process; every one exited
normally with every check passing: no torn or lost record, no wrong word, no missing
stamp, no bounded wait that gave up, no hang. Host load average (one minute) 3.2 to
14.5 on 96 CPUs.

### Every word of every block checked

| arrangement | runs with records | core-records checked | blocks broadcast | receiver block checks | wrong or missing |
|---|---|---|---|---|---|
| lanes on NoC0, every column | 21 | 924,840,000 | 127,155,084 | 330,213,845 | 0 |
| lanes on NoC1, every column | 21 | 924,840,000 | 125,373,358 | 325,998,186 | 0 |
| lanes on NoC0, column 0 only | 15 | 660,600,000 | 10,697,284 | 27,553,803 | 0 |

Each arrangement ran lanes of 5 and 10 with blocks of 64 B, 2 KiB and 24 KiB (NoC0 and
NoC1 also lanes of 2 with 64 B; column 0: lanes of 2, 5 and 10 with 64 B and lanes of
10 with 2 KiB and 24 KiB), each against 10^3, 10^5 and 10^6 records.

Microseconds per block (sender, mean over senders), lanes alone (1,000 blocks, X280
idle) and during 10^6 records:

| lanes | block | NoC0 alone | NoC0 with records | NoC1 alone | NoC1 with records |
|---|---|---|---|---|---|
| 2 | 64 B | 0.636 | 0.660 | 0.640 | 0.669 |
| 5 | 64 B | 0.748 | 0.759 | 0.749 | 0.761 |
| 10 | 64 B | 0.899 | 0.913 | 0.906 | 0.916 |
| 5 | 2 KiB | 6.693 | 6.710 | 6.688 | 6.702 |
| 10 | 2 KiB | 6.841 | 6.864 | 6.842 | 6.857 |
| 5 | 24 KiB | 74.23 | 74.34 | 74.19 | 74.20 |
| 10 | 24 KiB | 74.46 | 74.56 | 74.41 | 74.41 |

Microseconds per record (10^6 records): 0.7752 alone (10^5: 0.7734 with the record
kernel on NoC1, 0.7706 on NoC0); 0.7757 to 0.7780 with lanes on NoC0, 0.7719 to 0.7742
with lanes on NoC1, 0.7755 to 0.7777 with one column of lanes.

What was observed:

- No lost flit, wrong word or hang in any arrangement, including lanes of 5 and 10
  whose multicasts pass routers two or more hops from the sender while the X280's
  records cross the same column routers on NoC0.
- The cost of running together is small: blocks take 1 to 5% longer at 64 B and under
  0.3% at 2 KiB and 24 KiB; records take under 0.4% longer.
- Every column broadcasting at once and one column alone behaved the same (no error
  either way; the same per-block times within 0.5%).
- At 2 KiB and 24 KiB the block time is almost all the sender's fill and the
  receivers' check (about 16 cycles per word for the two together): a 24 KiB multicast
  at one link's rate takes about 0.3 us of the 74 us (an estimate), so these shapes
  kept the lane multicasts in flight for a small fraction of the run. The stream runs
  below cover that case.

### Stream mode: long multicasts back to back

Lanes of 2, 5 and 10 in every column, window 4, against 10^5 records once and 10^6
records three times per shape and NoC (plus one run of 20,000 blocks with the X280
idle):

| arrangement | runs with records | core-records checked | blocks broadcast | receiver header checks | full body checks | wrong or missing |
|---|---|---|---|---|---|---|
| lanes on NoC0 | 24 | 2,232,000,000 | 1,006,612,800 | 2,395,743,015 | 3,478,883 | 0 |
| lanes on NoC1 | 24 | 2,232,000,000 | 998,646,441 | 2,380,849,942 | 3,453,316 | 0 |

Per shape: microseconds per block and duty (share of the sender's time inside
multicast plus barrier, mean over senders), lanes alone and during 10^6 records (the
three runs agree to the digits shown), the largest share of a receiver's time spent in
full checks and poisoning, and microseconds per record:

| lanes | block | NoC0 alone | NoC0 with records | receiver checks | us/record | NoC1 alone | NoC1 with records | receiver checks | us/record |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2 KiB | 0.302 / 0.57 | 0.321 / 0.54 | 0.036 | 0.7807 | 0.305 / 0.58 | 0.323 / 0.55 | 0.036 | 0.7748 |
| 5 | 2 KiB | 0.378 / 0.55 | 0.397 / 0.52 | 0.029 | 0.7795 | 0.383 / 0.55 | 0.400 / 0.53 | 0.029 | 0.7753 |
| 10 | 2 KiB | 0.490 / 0.53 | 0.511 / 0.52 | 0.023 | 0.7786 | 0.495 / 0.54 | 0.513 / 0.52 | 0.023 | 0.7740 |
| 2 | 24 KiB | 0.630 / 0.74 | 0.640 / 0.74 | 0.056 | 0.7781 | 0.628 / 0.74 | 0.638 / 0.74 | 0.056 | 0.7733 |
| 5 | 24 KiB | 0.752 / 0.73 | 0.759 / 0.72 | 0.047 | 0.7779 | 0.748 / 0.73 | 0.758 / 0.72 | 0.047 | 0.7728 |
| 10 | 24 KiB | 0.935 / 0.72 | 0.939 / 0.71 | 0.038 | 0.7775 | 0.927 / 0.72 | 0.937 / 0.71 | 0.038 | 0.7727 |

Records alone take 0.7752 us (10^6, record kernel on NoC1, the NoC0-lane arrangement)
and 0.7706 us (10^5, record kernel on NoC0, the NoC1-lane arrangement).

What was observed:

- No lost flit, wrong word or hang in about 10^9 long multicasts per NoC, each lane
  sender inside multicast plus barrier 52 to 74% of the time, while the X280's records
  crossed the same column routers on NoC0. 60 lanes of 2 at 24 KiB moved about 2.3 TB/s
  of broadcast data (24 KiB per 0.64 us per lane).
- The records slowed by under 0.8% (NoC0 lanes) and the blocks by under 7% (2 KiB,
  lanes of 2) or 2% (24 KiB).
- A lost body flit can only show as a wrong word in a block whose body is checked in
  full (every 512th or 2,048th block); in the others it would show as a wrong header
  if it hit the header line, or as a barrier that never completes (a hang). None of
  these happened.

## Answer

- Lanes on NoC0 with the X280's records on NoC0: **clean**, both with every word
  checked (lanes of 2, 5 and 10; 64 B to 24 KiB) and with long multicasts back to back
  (2 KiB and 24 KiB, duty 52-74%).
- Lanes on NoC1: **clean** in the same runs.
- One column of lanes against every column: **clean**, no difference.

In all, about 7.0 x 10^9 checked core-records, 5.5 x 10^9 checked receiver blocks or
headers and no lost or wrong word, no hang. This is an observation of a hardware race
that did not occur, not a proof that it cannot: the documented defect needs a
reservation refused two or more hops on, and its rate under other traffic patterns is
unknown.

For a runtime that lets programs broadcast while the X280 dispatches, NoC1 is the
arrangement that does not depend on that observation: the lane broadcast then never
shares a router with the X280's NoC0 broadcasts (NoC0 and NoC1 are separate networks;
an inference from the documentation, not tested beyond the runs above). It costs
nothing measurable here (block and record times within 1.5% of NoC0's). On NoC1 the
lane sender is the lane's bottom core and the data-movement kernel on NoC1 issues it.

## Not tested

- A sender inside multicast plus barrier more than 74% of the time (several blocks in
  flight per barrier); record depth 1; the host-fed dispatch path; lanes of other
  lengths than 2, 5 and 10.
- Broadcasts that leave one column, or cross a panel boundary.
- Other cards, clocks, or harvesting patterns than this one p150a.
