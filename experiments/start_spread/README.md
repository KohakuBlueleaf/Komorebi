# Why one launch in several starts its cores late

[host_push](../host_push/README.md) saw that with two or more cores one push in three
had a span 2 to 10 times longer than the others, every byte still exact. This
experiment asks what those pushes have in common. The span runs from the first core's
start to the last core's last write, so it can grow because a core writes slowly or
because a core starts late. It was the second.

## Result

Blackhole p150a, card 0, slow dispatch, AICLK 1350 MHz (sampled about every 0.4 ms in
one job: 25,290 samples, all 1350), host load 2.8 to 5.7. Cores 0 to n-1 of row 0 are
launched together by one `Launch.run()`; a core's start is the wall clock it reads in
its first instructions, as a cycle offset from the first core to start.

- **A core's own push time does not grow in the slow pushes.** In 12 pushes of 4 MiB on
  12 cores, every core's time from its start to its last write was 36,000 to 76,000
  cycles in the ten slow ones; in the two fast ones, where all cores overlap, up to
  0.24 M. The slow pushes are pushes whose cores started far apart: 3.5 to 4.9 M cycles
  from the first start to the last in nine of eleven pushes (the first of a process
  excluded), 0.24 M in the others.
- **A normal launch starts the cores 21 to 22 k cycles apart** (16 µs a core, 0.24 M for
  12), in order. A slow one starts the first cores like that, then waits between
  consecutive cores, up to 0.48 ms a gap, one to eight times a launch:
  `[22, 22, 596, 251, 484, 642, 221, 532, 643, 179, 572]` (thousand cycles between
  cores 0 to 11). With two cores the one gap is either 21 to 22 k cycles (217 of 300)
  or slow (83 of 300): 55 of them between 640 k and 655 k (0.474 to 0.485 ms), 27
  between 30 k and 640 k, one 669 k.
- **The late core is always the later one.** In the 300 two-core gaps the minimum was
  21.3 k and none was negative: the delay falls between the first core's start and the
  second's, never on the first.
- **It does not need a push.** A kernel that only stamps its start and returns shows it
  (launches with a spread over 1 M cycles, the first of a process excluded, 12 cores:
  1 of 11, 35 of 39, 7 of 39, 12 of 79, 10 of 79 in five jobs; two cores, a gap over
  100 k: 11 of 99, 37 of 99, 28 of 99; a spin of 50 k cycles instead of a write, 12
  cores: 17 of 39). The push kernel with 4 KiB a core or 350 KB a
  core shows it in 9 of 11 launches each. The share changes from job to job, not with
  the kernel.
- **What it is not:**
  - *The bytes pushed.* 4 KiB a core and 350 KB a core: the same 9 of 11. No write at
    all: 35 of 39 in one job.
  - *A clock change.* AICLK never left 1350 in the sampled job, in the window of any of
    its 37 slow launches either.
  - *Telemetry reads.* A child process reading the card's `tt_aiclk` file in a loop
    for blocks of 10 launches, alternating with blocks without it: 5 of 40 against 7 of
    39 slow.
  - *The other card.* A scan of `/proc` before every launch of an 80-launch job found no
    process holding the other card open, and 10 of 79 launches were slow.
  - *The host thread being switched out.* In 40 launches the thread's CPU time equalled
    its wall time (it is busy for the whole launch call, 17 ms a normal launch) with
    0 involuntary context switches in the slow ones and no system time. The slow
    launches did not take longer on the host: 7.8 to 19.7 ms against 17.5 ms.
  - *A link error.* The card's AER counters (correctable errors: RxErr, BadTLP, Timeout
    and the rest; non-fatal) read 0 on both cards (counts since boot); the link is 16 GT/s x8.
  - *Idle time before the launch* (two cores, 25 launches after each wait): gap over
    100 k in 7 of 24 (no wait), 4 of 25 (10 ms), 6 of 25 (100 ms), 11 of 25 (500 ms).
    Noise is about 9 points a block; at most a weak rise.
- **What is left:** a delay of a fixed length, most often 0.476 ms (642.5 k cycles at
  1350 MHz), inserted between two host stores in the launch, in a share of the launches
  that changes over time. The host spends its time in stores to the card with the
  thread running, so a store (or the path behind it) waits. Which wait it is I did not
  find. Not tried: another host, a pinned thread, another PCIe slot, fast dispatch.

What this means for a measurement: time a push from the cores' own starts (the per-core
time in `push.py`'s output, `core_data`), or take the median of several pushes. A span
from the first start includes the launch's stalls.

## Running it

On a card through the hang watcher, with tt-metal's environment set and the repository's
`src` on the path:

    TT_VISIBLE_DEVICES=0 python spread.py --cores 12 --reps 40
    TT_VISIBLE_DEVICES=0 python spread.py --cores 2 --reps 100 --clock-out clock.json
    TT_VISIBLE_DEVICES=0 python spread.py --cores 2 --reps 25 --sleeps 0,10,100,500

It prints one JSON line a launch (the start offsets, the host's wall and CPU time, the
context switches, the other card's holders) and a summary per block.
`../host_push/push.py` prints `core_start`, `core_data` and `start_spread` for each
push.

## Files

- `kernels/stamp.cpp`: stamps the wall clock, then, as the host chooses, nothing, spins,
  writes 16 bytes to the host channel, or writes a block (offsets from 40 MiB, checked).
- `spread.py`: the launches and the per-launch measurements; `sample_clock.py`: the
  child process that samples `tt_aiclk` with time stamps.
