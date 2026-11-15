# Code placed above 1 MB of L1: where the extra cycles come from

A loop of overlay code ran about 4 cycles an iteration slower when its slot sat in L1
at or above 1 MB (0x100000) than below it, with instruction for instruction the same
code. This experiment takes the loop out of the runtime: a handful of loops written in
assembly are copied by a one-shot kernel to chosen L1 addresses, the instruction cache
is invalidated, and each is run and timed with the core's wall clock. Nothing in `src/`
changes.

## Result

On the reader core (data-movement core 0) of core (0, 0), Blackhole p150a, slow
dispatch. Cycles are the core's wall-clock cycles; the table gives the cycles an
iteration of a warm loop (repeats 2 to 4 of each run; the first repeat starts with the
cache invalidated). The kernel's own text is at 0xa4d0 and runs the same loops in place
at the low rate. The table comes from the jobs that filled the region around each loop
with nops (below); the repeats of a run never differed by more than 0.01 cycles an
iteration, except where a row says so.

| Loop (instructions an iteration, taken branches) | below 1 MB (0x90000) | at or above (0x120000) | extra |
|---|---|---|---|
| `tiny`: `addi`, `bnez` back (2, 1) | 2.001 | 6.001 | +4.0 |
| `medium`: 60 `addi`, the loop (62, 1) | 62.015 | 66.011 | +4.0 |
| `straight`: 16 `addi`, a never-taken branch, the loop (19, 1) | 19.004 | 23.003 | +4.0 |
| `untaken`: 16 never-taken branches, the loop (18, 1) | 18.004 | 22.003 | +4.0 |
| `jumps`: 16 jumps, each to the next instruction, the loop (18, 17) | 18.020 | 86.003 | +68.0 = 17 x 4 |
| `indirect`: 8 times `lla`, `jalr` to the next instruction, the loop (26, 9) | 26.004 | 62.003 | +36.0 = 9 x 4 |
| `far`: a jump forward over 2 KiB of nops, the loop branch back (3, 2) | 3.019 | 18.07 to 18.17 | +15 = 2 x 7.5 |
| `large`: 600 `addi`, the loop (602, 1) | 602.15 | 613.3 | +11.2 |
| `memory`: 8 times `lw`, `addi`, `sw` in a buffer, the loop (26, 1), data low | 82.71 | 88.59 | +5.9 |
| the same, data at 0x120000 and code below | 82.71 | | 0 |

What follows from it:

- **The extra is paid by taken branches and jumps, 4 cycles each, and only when the
  target is at or above 0x100000.** `jumps` and `indirect` add 4 per taken transfer
  (17 and 9); `straight` and `untaken` add 4 in all, the loop branch alone: a branch
  that is never taken costs nothing. Below 1 MB a jump, even an indirect `jalr`, costs
  its one cycle.
- **It is the target's address.** A loop with its `addi` at p and its `bnez` at p + 4
  back to p runs at 2.001 for p from 0xFFFE8 to 0xFFFFC (for 0xFFFFC the `bnez` itself
  is at 0x100000) and at 6.001 for p = 0x100000 and 0x100004.
- **Straight-line code is nearly unaffected:** 600 instructions in a row cost about 7
  cycles more above 1 MB (1.2 %), beyond the one taken branch. A jump to a line other
  than the next costs more than 4 (`far`: about 7.5 each).
- **Data is not slower.** The `memory` loop with its buffer at 0x120000 and its code
  below 1 MB runs at the low rate; with its code above, the extra is the loop branch and
  the instruction stream, not the loads and stores.
- **It does not depend on the jump's distance** (4 bytes, 2 KiB, or through a register),
  on alignment (the 12-byte loop at 16 offsets of 4 bytes from 0x120000 and 7 from
  0x90000: always 6.001 and 2.001, also when the loop straddles a 64-byte line), on the
  words that follow the loop (nops, zeros, `0xDEADBEEF`, all ones: 62.015 and 66.011
  every time), or on the address inside the upper part (the tiny loop, in a first sweep:
  6.001 at every address from 0x100000 to 0x170000 and 2.001 at every one from 0x90000
  to 0xf0000).
- **What it is, by inference:** the instruction fetch after a redirect waits longer for
  code in the upper half megabyte of L1, as if the instruction cache did not serve it,
  while the sequential fetch behind a taken branch hides most of the wait. The registers
  that would show it (the cache and prefetch controls) were not read.

For code that is placed at run time this means: keep an overlay's slot below 0x100000
(the runtime does), and expect about 4 cycles for every taken branch and call in a
body that has to run above it.

## Where the runtime puts code

A Tensix core's L1 (1,536 KiB) as a run lays it out, before and after the change that
made the overlay slots the run's first circular buffer:

| Region | Placed by | Where |
|---|---|---|
| firmware, mailboxes | tt-metal | below 0xA080 |
| kernels (the text of all five threads) | tt-metal's kernel config buffer, sized by `Session(kernel_bytes=...)` | 0xA080 up to 0x1B480 (default 70,656 B), 0x1F640 for the ray tracer's 87,488 B |
| overlay slots (a program's overlays, the service overlay, programs held as overlays) | the runtime | before: just under 0x100000 when the circular buffers ended below that, else the allocator's top-down choice; now: at the bottom of the L1 left, 0x1B480 by default |
| circular buffers | tt-metal, from the bottom of the L1 left, in the order the run declares them | before: from 0x1B480; now: above the slots |
| control block, overlay area, transfer buffer, L1 tensors | tt-metal's allocator, top-down | the top of L1 (data only) |

The kernels never reach 1 MB unless `kernel_bytes` is near it. The slots did: every run
holds the service overlay (the reader that serves the host's reads and writes of a live
run), and a run whose buffers take more than about 900 KiB had it placed above 1 MB:
the dense product at 8192 x 8192 on two cards logged its 7,424-byte slot at 0x16a000
and 0x172000. The programs under `projects/` declare no overlays of their own, so the
service overlay was the only code of theirs above 1 MB. Now a load whose slots would
end above 0x100000 raises `km.SlowCodeError`, unless the session was opened with
`slow_code=True`.

`placement.py` runs example 01's program with one more buffer of 960 KiB (1,020,032
bytes of circular buffers a core), checks 200 DOT commands, then reads a 1 MiB DRAM
tensor five times while the run is live, each read a PUSH served by the service overlay
from the slots. Card 0, 2026-11-14, one run each:

| | slots | DOT checked, wrong | PUSH cycles, 5 reads | reads wrong |
|---|---|---|---|---|
| before (a4f7d6c2) | 0x166000 | 200, 0 | 97,432 97,736 97,640 97,562 97,732 | 0 |
| after | 0x1b480 | 200, 0 | 97,434 97,459 97,478 97,638 97,599 | 0 |

The slots move below 1 MB; the PUSH does not get measurably faster (mean 97,620 against
97,522 cycles, 0.1 %): moving 1 MiB from DRAM to the host's memory is bound by the
transfer, not by the service loop's branches. Example 01 (`mapcollect.py --jobs 20
--count 8`) after the change: slots at 0x1b480, every job right, the ten integer jobs
bit-exact.

## Not found, not tried

- The first two jobs with the 60- and 600-instruction loops (a first version of the
  kernel that did not fill the region with nops) measured the 60-instruction loop 4 to 8 cycles an iteration slower on some
  placements than every later job at the same address (0x90000 and 0x120000), and the
  600-instruction loop 20 more above 1 MB. The fill word does not matter (above), and
  none of this came back in the eight jobs after them; I cannot explain it, and no
  number of those two jobs is in the table.
- Only the reader core. The writer core cannot write the instruction cache's invalidate
  register (a job that tried it left that core stuck until the next device open), and
  the compute cores were not run.
- Jumps farther than 2 KiB, and a call from a body above 1 MB into one below.
- The AICLK was not sampled in these jobs (it read 1350 MHz in every sample of the
  sibling experiment `start_spread` on the same card); the numbers are cycles of one
  core, not seconds.

## Running it

On a card through the hang watcher, with tt-metal's environment set and the repository's
`src` on the path (`--card` is the card the job owns, `TT_VISIBLE_DEVICES` must name it):

    TT_VISIBLE_DEVICES=0 python loops.py --card 0 \
        --runs tiny@0x90000,tiny@0x120000,jumps@0x90000,jumps@0x120000

A run is `blob@code`, `blob@code:data` or `blob@code:data/fill` (code 0 runs the loop
in place); the blobs are listed in `loops.py`. It prints one JSON line with the cycles
and the cycles an iteration of each repeat.

`placement.py` the same way: `TT_VISIBLE_DEVICES=0 python placement.py --card 0`
(`--buffer` KiB, `--reads`, `--pages` of 4 KiB a read).

## Files

- `kernels/loops.cpp`: the loops as assembly in one section, the copy, the invalidation
  of the core's own instruction cache, the timing; the region around a loop is filled
  with nops (or the fill word) from 4 KiB below it.
- `loops.py`: reserves the regions in every core's L1 through the session (`empty` with
  an address), launches the kernel, prints the cycles an iteration.
- `placement.py`: example 01's program with a large buffer, where its slots land, its
  DOT check and the PUSH cycles of reads while it runs.
