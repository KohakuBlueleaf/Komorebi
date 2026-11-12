# X280 threads on a work cluster

Questions a user thread on X280 cluster 1 depends on, asked on a card before the
thread library was built on them: whether a second image can run beside the dispatch
kernel, how a Tensix core and the X280 see the same DRAM, how fast the X280 reads and
writes it, and what the host can do about a thread that does not return.

## Results

| Question | Answer |
|---|---|
| Can cluster 1 run its own image beside the dispatch kernel on cluster 0 while a resident grid runs? | Yes: 100 of 100 start, run and stop cycles clean; the kernel saw no error, every core's state words were intact |
| Does a Tensix core's NoC write into the cluster's DRAM bank reach the X280, and the reverse? | Only with a hand-over in each direction. A Flush64 sweep of the cluster's cache controller over the region's lines is enough both ways (30 ns per line) |
| What address does each side use? | The X280: `0x400030000000` + the bank address. A Tensix core: `get_noc_addr_from_bank_id<true>(6, bank address)`. Cluster 1 reaches bank 6 |
| How fast does the X280 read and write its bank? | About 1.3 GB/s read and 1.8 GB/s write with four cores, the same for 1 MB and 64 MB; Flush64 sweeps run at 5 to 8 GB/s of region |
| What does a decision over every core's data cost? | Prefix sums over 119 x 1,024 counters: 2.13 ms on four cores, take and give included, exact |
| A thread that ignores its deadline? | The host asks it to stop; if it polls nothing, the cluster stays stuck until a card reset (from the runtime's sources, not run) |

All card numbers: one Blackhole p150a, slow dispatch, the X280 at the runtime's
1,750 MHz PLL setting, Tensix clock 1,350 MHz throughout, the host's load average
between 2.7 and 9.2. The X280 timer counts 50 MHz; times are from it.

## Two images (`probe.py two`)

The dispatch kernel runs on cluster 0 under the resident grid of the map/collect
example. The `beat` image (`x280/beat.c`: every hart counts loop passes for a given
number of timer ticks, then stores its count) is started on cluster 1, eight NOP jobs
of random teams go through the kernel meanwhile, the image is waited for and stopped
through the runtime; that is one cycle.

    python probe.py two --cycles 1 --ticks 0 --jobs 1        # smallest
    python probe.py two --cycles 100 --ticks 500000 --jobs 8

| | |
|---|---|
| cycles clean | 100 of 100 |
| records and events of the kernel | 801 and 801, error none |
| cores with intact state words, cores that gave up a wait | 120, 0 |
| loop passes per hart in 10 ms | 970,323 to 970,326 |
| start, wait, stop (median, host time) | 0.32 ms, 17.1 ms, 0.18 ms |

## DRAM visibility (`probe.py dram`)

Two regions of 4 pages of 4 KB in bank 6, the bank cluster 1's memory port reaches.
The host fills region A; the X280 reads it; a Tensix core (`kernels/dram_rw.cpp`)
writes another pattern into it through the NoC; the X280 reads it without a hand-over
and after a Flush64 sweep. Then the X280 writes region B with plain stores, and a
Tensix core copies B into an output buffer three times: with no hand-over, after a
Flush64 sweep, and after a second pattern and a sweep. The host steps the two sides
through a go line in the cluster's local memory and waits for the Tensix kernel's
report line (written last) before each step.

    python probe.py dram --pages 4

| Direction | Hand-over | Words seen of 4,096 |
|---|---|---|
| Tensix writes, X280 reads | none | 0 new (all 4,096 still the host's) |
| Tensix writes, X280 reads | Flush64 over A | 4,096 new |
| X280 writes, Tensix reads | none | 0 new (all 4,096 still the host's) |
| X280 writes, Tensix reads | Flush64 over B | 4,096 |
| X280 writes again, Tensix reads | Flush64 over B | 4,096 of the second pattern |

The Flush64 sweep over 256 lines took 388 ticks (7.8 µs, 30 ns per line); the plain
stores of 16 KB took 2,376 ticks (47.5 µs). The Flush64 store did not fault (the image
ran it with the runtime's probe mode armed).

Addresses: region A at bank address `0x593e80`; the X280 read it at
`0x400030000000 + 0x593e80`; the Tensix core's `get_noc_addr_from_bank_id<true>(6,
0x593e80)` returned `0x512000593e80`, that is NoC (18, 20), local `0x593e80`, with a
bank offset of 0. The bank the X280 saw was checked by the pattern itself: page 0 of
A held the host pattern of bank 6.

So a region changes hands with one sweep: before the X280 reads what the grid wrote
(write back and drop the lines it holds) and after it wrote what the grid will read
(write them back). Flush64 does both, so one operation serves both directions.

## Rates (`probe.py rates`)

The `rates` image (`x280/rates.c`) runs on all four harts, each on its quarter of a
region of bank 6, with a barrier between phases; each phase's time is the slowest
hart's. Then the prefix-sum job: a cores x bins table of counters (random, 0 to 63),
`out[c][b]` = the counts of every core in bins before `b` plus bin `b` on the cores
before `c`; each hart owns a quarter of the bins, takes its part of the input with a
Flush64 sweep, and gives its part of the output with another. The host checks the
output against numpy.

    python probe.py rates --small 1048576 --large 0 --cores 8 --bins 64 --reps 1   # smallest
    python probe.py rates --small 1048576 --large 67108864 --cores 119 --bins 1024 --reps 3

| Phase (four harts) | 1 MB | | 64 MB | |
|---|---|---|---|---|
| Flush64 sweep, clean lines | 170 µs | 6.2 GB/s | 8.43 ms | 8.0 GB/s |
| vector read, after the sweep | 790 µs | 1.33 GB/s | 49.3 ms | 1.36 GB/s |
| the same read again | 810 µs | 1.29 GB/s | 49.8 ms | 1.35 GB/s |
| vector write | 612 µs | 1.71 GB/s | 37.7 ms | 1.78 GB/s |
| Flush64 sweep, dirty lines | 211 µs | 5.0 GB/s | 8.47 ms | 7.9 GB/s |
| 8-byte scalar read | 830 µs | 1.26 GB/s | 52.5 ms | 1.28 GB/s |

The second read of 1 MB is no faster than the first: the region does not stay in the
cache in a way that helps a streaming pass, so a thread's cost is its bytes over
about 1.3 GB/s.

| Prefix-sum job, 119 x 1,024 counters (476 KB in, 476 KB out) | |
|---|---|
| take, give and compute, three runs | 2,137.0 µs, 2,130.8 µs, 2,138.7 µs |
| of which the take sweep | 61 to 71 µs |
| output against numpy | exact, 0 words wrong |

A per-frame decision over every core's counts costs about 2 ms of the cluster: the
order of a frame, not of a segment.

## A thread that ignores its deadline (from the sources, not run)

Read in tt-metal's X280 runtime (`tt_metal/fabric/experimental/x280`):

- The runtime's stop is a word: `cleanup()` publishes a STOP command in each hart's
  control line and polls, up to its timeout, for every hart to return to the idle
  firmware. A hart sees it only if it polls `x280_hart_stop_requested`; there is no
  interrupt.
- A hart that never returns makes `cleanup()` end in `StillRunningActiveFw`, and the
  tile is latched unhealthy for the rest of the process: later starts on that cluster
  are refused. The other clusters are separate tiles with their own sessions and go on;
  the dispatch kernel on cluster 0 can still be stopped normally.
- A tile's reset can only be released, not asserted again: the next process finds the
  cluster out of reset without its idle firmware (`ResetRequiredNoResidentIdle`), and
  only a card reset brings it back.

So the rule for thread code: every loop polls the stop (the library's own waits do),
and a missed deadline ends the thread with a result instead of a wait. The host's stop
line and the runtime's STOP then end any thread that follows the rule.

## Files

| File | What |
|---|---|
| `probe.py` | the host side of the three card probes; one JSON line per run |
| `build.py` | builds the images into `build/` with the X280 toolchain of the tt-metal tree |
| `x280/probe.h` | the images' shared layout: argument block, progress stamps, go line, result words, bounded waits, the Flush64 sweep |
| `x280/beat.c`, `x280/dram.c`, `x280/rates.c` | the images |
| `kernels/dram_rw.cpp`, `kernels/dram_probe.h` | the Tensix side of the DRAM probe and the pattern both sides check |

Every wait in the images ends by a deadline or the runtime's stop and leaves a
progress stamp; every wait on the host is bounded. Run under a hang watcher, one card,
`python build.py` first.
