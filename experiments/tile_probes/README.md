# Tile probes: the vector unit and the compute RISC-V cores of one Tensix tile

Five small measurements on one Blackhole Tensix core, made for the design of a ray walk
that keeps 32 rays in the 32 lanes of the vector unit (SFPU) and a small tree in L1:

1. the cost of predicated vector instructions (all lanes, one lane, no lane, 8 nested
   conditions);
2. the "is any lane still enabled" test through the `tt_cfg_qstatus` CSR, against an
   OR-reduction read through `Dst`;
3. a gather: each of the 32 lanes loads its own 1, 2 or 4-word record from a 256-record
   table in L1, copied by the pack RISC-V core through the `Dst` memory window (scalar
   loop, a scheduled scalar loop, and the core's vector extension);
4. the inverted form: one box node against 32 rays (slab test), with the 32 hit flags
   read back by the core;
5. the speed of a compute RISC-V core on its own, alone and three at once;
6. one full walk of a 255-node box tree for 32 to 1024 rays, in a ray-first and a
   node-first form;
7. one 64-byte NoC read submitted by a compute RISC-V core from another tile's L1;
8. what the cost model of a tile compiler needs and the sources leave open: the cost of
   pushing one Tensix instruction word, replay and MOP loops against pushing, a
   load-macro chain against the same work one instruction at a time, the code the SFPI
   compiler emits, which numbering of `SFPCONFIG`'s combine modes the hardware uses, and
   loop bodies around 512 bytes on a compute core (probes 8 to 10, 12, 13), the matrix
   unit's issue cost and Dst read-after-write stall (11), and how the matrix unit (14) and
   the packer (15) round exact ties (section 8).

Every result value is checked against a CPU reference by the driver.

## How it runs

One program on core (0, 0), launched through the session (`Session.launch`) in slow
dispatch:

| part | file |
|---|---|
| reader (RISCV_1): the input page from DRAM into CB 0 | `kernels/reader.cpp` |
| compute: one probe per build (define `PROBE`), output into CB 16 | `kernels/probe.cpp` |
| vector-unit probes 0 to 4, run by the pack thread (TRISC2) | `kernels/vec.hpp` |
| RISC-V loop probe 5, run by the TRISCs selected in a mask | `kernels/rv.hpp` |
| tree walk probe 6, run by the pack thread | `kernels/walk.hpp` |
| NoC read probe 7, run by the pack thread | `kernels/nocread.hpp` |
| instruction-cost probes 8 to 10 and 13, run by the pack thread | `kernels/costs.hpp` |
| matrix-unit probes 11 and 14 (unpack, math and pack threads) | `kernels/fpu.hpp` |
| loop-size probe 12, run by the TRISCs selected in a mask | `kernels/icache.hpp` |
| writer (RISCV_0): CB 16 to DRAM once compute pushed it | `kernels/writer.cpp` |
| shared layout of the input and output words | `kernels/probe.hpp` |
| driver: builds the program, runs it, checks against the CPU reference | `probes.py` |
| card job through the hang watcher | `run.sh` |

- **Timing.** The kernel reads the tile's wall clock (`RISCV_DEBUG_REG_WALL_CLOCK_L`),
  which counts AICLK cycles: the smoke probe shows it equal to `mcycle` over a 300,000
  cycle spin. AICLK is read from the driver's sysfs (`tt_aiclk`) before and after each
  run: 1350 MHz in every run here, so 1 cycle = 0.741 ns. Vector-unit spans are taken
  between two `tensix_sync()` calls (the thread's Tensix instructions have finished),
  so they include one synchronisation (10 to 12 cycles when idle).
- **Raw instructions.** The vector code is raw Tensix instructions (`TTI_*`), not the
  SFPI compiler, so every instruction is accounted for. Repeated bodies come from the
  replay buffer.
- **Dst window.** `Dst` is set to 32-bit mode; the pack core sees it at `0xFFBD8000`
  with format 0 (uint32), and the vector unit loads and stores it in the INT32 mode.
  Broadcast `l` of the 4-row group at `Dst` address `a` is window byte `64 a + 8 l`. The
  smoke probe checks the round trip bit for bit (denormals, -0, infinities, NaN, all
  ones) and that `LReg[15]` holds `2 x lane`.
- **Ordering.** Window stores are followed by a load of the last word written (same
  memory region, processed in order) before the vector unit is told to load them.
- **Runtime arguments.** The nonce, then the probe parameters `p0` to `p6`. The pack
  thread writes the nonce last, then pushes the output page.
- **Output words.** Layouts beyond the shared header (`kernels/probe.hpp`):
  - probe 2 (`anycc`), data: words 0 to 119 hold the `qstatus` bits per recording pass (3)
    and body (40); `128 + K m` the trip counts of method `m`; `160 + m` the "any" of a
    check alone; `256 + 32 v` the L1 per lane per loop (the 5 methods, then the fixed 32
    bodies); `512 + 40 i + k` the bit after body `k` with a wait of `kDelays[i]` cycles
    past a fence.
  - probe 4 (`slab`), input: ray fields at `32 k + l` for `k < 7`, nodes at
    `512 + 6 n + c` as fp32 bits (the low half holds bf16 for `kBf16`). Timing: per
    repetition one sweep over `n` nodes, then one readout of all `n` hit masks. Data:
    the hit mask of node `n` at word `n`; L6 of lane `l` at `64 + 32 n + l`.

## Running

```bash
export TP_ENV=<script that sets up tt-metal and sets $PY>
export HANGWATCH=<hang watcher> LOCK_DIR=<lock directory> CARD=0
./run.sh smoke-1 smoke
./run.sh pred-1 pred --reps 1000 --k 5
./run.sh anycc-1 anycc --k 5
./run.sh gather-w4-1 gather --words 4 --reps 64 --variant 1   # 0 plain, 1 scheduled, 2 RVV
./run.sh slab-1 slab --nodes 64 --reps 10 --slab-mode f32 --variant 1
./run.sh riscv-1 riscv --mask 7 --n 100000 --n-float 10000
./run.sh walk-1 walk --rays 256 --ray-set camera --form b   # --form a --copy scalar
./run.sh nocread-1 nocread
./run.sh costs-1 sfpi,push,macro,sfpcfg --k 5   # one session, one JSON line each
./run.sh icache-1 icache --mask 2 --icache-passes 1000
```

A comma list of probes runs in one session and stops at the first one that does not
finish cleanly. Probe 11 (`fpu`) takes `--first N --count M`, the configurations to run
(section 8).

Without the hang watcher: `python probes.py <probe> [options]` with tt-metal's Python
environment and `TT_METAL_SLOW_DISPATCH_MODE=1`. Each run prints one JSON line;
`nonce_ok` and `kernel_error == 0` say the kernel finished cleanly, and every probe
reports its count of wrong values. Variant 2 sets `ComputeConfig.enable_trisc2_rvv`.

## Results

tt-metal v0.81.1, one Blackhole card, AICLK 1350 MHz. In sections 1 to 7 each number
is the median over the in-kernel repetitions of three separate runs; the three runs
agreed to within the min/max given. Host load average 2.4 to 12 during the runs.

### 1. Predication

1000 replays of 32 `SFPMAD` (four independent registers, `x = x * 1 + 1`), minus the
12-cycle empty span.

| lanes enabled | cycles per instruction | ns |
|---|---|---|
| all 32 | 1.000 | 0.741 |
| 1 (lane 13) | 1.000 | 0.741 |
| none | 1.000 | 0.741 |
| 24, inside 8 nested conditions | 1.000 | 0.741 |
| all 32, one dependent chain | 2.000 | 1.48 |

Entering and leaving 8 nested conditions (8 x `SFPPUSHC` + `SFPIADD` with condition,
8 x `SFPPOPC`, 24 instructions): 24.0 cycles, i.e. 3 cycles per level. Disabled lanes
keep their values and enabled lanes have the exact count in every run.

### 2. "Any lane still enabled"

Broadcast i's counter starts at 2i and the body subtracts 2 with a condition, so lane i
leaves after body i + 1; the loop must end after exactly 32 bodies.

| check after each body | bodies run | cycles per check (loop minus 32 unchecked bodies) |
|---|---|---|
| `tensix_sync()`, then read `qstatus` bit 11 | 32 | 20.8 |
| `fence`, then read bit 11 | **33** (one late) | 16.8 |
| poll `bstatus` SFPU busy bit, then read | 32 here, **33** in an earlier build | 40.6 |
| `fence`, poll busy bit, then read | 32 | 31.3 |
| counters to `Dst`, core reads 32 lanes (plain loop) and ORs | 32 | 554 |

- The bit read right after the push, with or without a fence, still shows the state
  before the last body (body 32 read as "enabled" in every recording pass). Read after
  a fence and two `mcycle` reads it was right in every case, so for a 2-instruction
  body the stale window is a few cycles; it grows with whatever is queued ahead.
- The busy-bit poll without a fence can run before the pushed instructions reach the
  coprocessor and see "idle": it is not a reliable wait on its own.
- `tensix_sync()` (wait until this thread's Tensix instructions finished) is both
  exact and the cheapest exact wait: 18 cycles for sync plus CSR read with an idle
  unit, 20.8 per check inside the loop. Bit 24 always equalled bit 11.
- The same 32-lane read through `Dst` done well costs 143 cycles (scheduled loads,
  probe 3 "indices out": `SFPSTORE`, sync, 32 window loads) or 99 cycles (vector
  extension, probe 4 readout, without the store and sync). The `qstatus` check is 5 to 27
  times cheaper. The flash-attention pattern (pack the tile, then `read_tile_value`)
  was not built here.

### 3. Gather by copy through the `Dst` window

Per 32-lane step: the lanes' indices go to `Dst` (`SFPSTORE`) and to the core, the core
copies each lane's record words from the L1 table into the lane's window slots, then
`SFPLOAD` brings them into `LReg`s. Cycles per step (ns in brackets):

| words per record | plain loop | scheduled loop | vector extension (T2) |
|---|---|---|---|
| 1 | 1097 (813) | 499 (370) | 239 (177) |
| 2 | 1578 (1169) | 759 (562) | 337 (250) |
| 4 | 2279 (1688) | 1080 (800) | 533 (395) |

| part of the step | plain | scheduled |
|---|---|---|
| indices out (`SFPSTORE`, sync, 32 window loads) | 575 | 143 |
| copy, 1 / 2 / 4 words | 504 / 986 / 1685 | 342 / 602 / 918 |
| `SFPLOAD` x W and sync | 14 to 19 | 14 to 19 |

- The plain loop is what the compiler makes of a straightforward loop: each window
  load or table load is used by the very next store, so every load's latency is
  exposed (about 17 cycles per window load). The scheduled loop submits 8 loads
  before using them.
- The vector variant: one stride-8 load (`vlse32`) for the 32 indices, then per word
  one indexed load (`vluxei32`) from the table and one stride-8 store (`vsse32`) into
  the window, LMUL 8. About 98 cycles per record word plus about 140 fixed. It worked
  on the first attempt. Only its total is a clean number (the compiler moved the
  middle timestamp).
- All loaded values were correct (8 index sets per run, the last 8 steps checked).

### 4. The inverted form: one node, 32 rays

Rays (inverse direction, `-origin x inverse`, `tmax`) in `Dst`; per node and axis two
`SFPMAD` give the slab distances, three `SFPSWAP` give min / max and update near and
far; one `SFPMAD` gives `far - near`, stored per lane (hit: sign clear).

| node fields | cycles per node test (32 rays) | ns |
|---|---|---|
| fp32 from L1 (core loads 6 words, pushes 12 `SFPLOADI`) | 116.6 | 86.4 |
| bf16 from L1 (6 `SFPLOADI`) | 98.7 | 73.1 |
| fixed at compile time (vector unit bound: 2 + 3 x 13 + 3 cycles) | 44.4 | 32.9 |

| reading the 32 hit flags into a 32-bit mask (unit finished) | cycles per node | ns |
|---|---|---|
| plain loop | 627 | 465 |
| scheduled loop (8 loads ahead) | 186 | 138 |
| vector extension (`vlse32`, `vmsge`, `vsm`) | 99 | 74 |

Flags: exact-valued data (all values multiples of 1/4, inverse directions powers of
two) bit-exact, 64 nodes x 32 rays, about 20 % hits; random float data against a
float64 reference: 0 wrong of 2048 in each of 3 seeds (30 to 33 % hits).

### 5. Compute RISC-V cores

Cycles per loop iteration, 100,000 iterations (soft float 10,000); instruction counts
from the compiled code.

| loop | instructions per iteration | T0 | T1 | T2 | T0, T1, T2 at once |
|---|---|---|---|---|---|
| add / xor chain | 5 | 5.00 | 5.00 | 5.00 | 5.00 each |
| multiply chain (`mul`, `add`) | 4 | 5.00 | 5.00 | 5.00 | 5.00 each |
| L1 load, sequential, used 2 instructions later | 6 | 12.0 | 12.0 | 13.0 | 12.0 each |
| L1 store, sequential | 5 | 5.00 | 5.00 | 5.00 | 5.00 each |
| L1 load, add, store | 7 | 13.2 | 13.2 | 13.1 | 13.2 each |
| software float `x = x * a; x = x + b` (`__mulsf3`, `__addsf3`) | 2 calls | 153.0 | 153.0 | 153.0 | 153.0 each |

One cycle = 0.741 ns, so the integer loops run at 1 instruction per cycle (a dependent
`mul` costs one more cycle), a sequential L1 load costs about 8 cycles of exposed
latency, and one software-float multiply plus add costs about 113 ns. The three cores
running at the same time do not slow each other in these loops. All checksums match
the CPU reference.

### 6. One walk of a box tree: ray-first against node-first

The tree: 128 leaf boxes (corners on a 1/4 grid in [-16, 16]), split at the median of
the longest centroid axis into a complete binary tree of 255 nodes, numbered breadth
first (children have larger numbers than their parent). A node record is 8 words in
L1: box min and max (fp32), child 0, child 1; a leaf carries `0x80000000 | leaf` in
both child words. Per ray the input gives `1 / direction` and `-origin / direction`
per axis and a best distance (1e30); the result is the entry distance of the closest
leaf box hit and that leaf's index. Two ray sets:

- **camera** (coherent): a pinhole at (0.3, 0.2, -24) looking along +z through an
  8, 16 or 32 wide grid of pixels;
- **random** (incoherent): origins on the 1/4 grid in [-12, 12], directions with
  power-of-two components aimed roughly at the centre.

The two forms (`kernels/walk.hpp`):

- **A, ray-first.** One packet of 32 rays in `Dst` at a time. The core keeps one stack
  per lane in L1 (depth 32) and a bitmask of live lanes. Per step it pops one node per
  live lane, copies the 32 node records (8 words each) into 8 `Dst` slots through the
  window (vector extension: one `vluxei32` + `vsse32` per word; or a grouped scalar
  loop), the vector unit runs the slab test against the lane's own box with `far`
  starting at the ray's best (so boxes beyond the best are culled), leaf lanes update
  best and leaf index under their condition, internal hit lanes are read back as a
  mask (vector extension) and the core pushes child 1 then child 0. The packet ends
  when no lane is live; the core checks that with `tensix_sync()` and `qstatus` bit 11
  after enabling exactly the live lanes, and compares it with its own count.
- **B, node-first.** Up to 16 packets (512 rays) resident in `Dst` at once (1024 rays
  run as 2 batches). The bins are one 32-bit lane mask per (node, packet) in L1
  (255 x 16 words). One sweep over the nodes in number order; for each node and each
  packet with a nonzero bin, the core enables the bin's lanes and pushes the box as
  12 `SFPLOADI` immediates read from the record in L1; the vector unit tests (leaf:
  updates best and leaf index under condition; internal: stores `far - near`). After
  the node's packets, one `tensix_sync()`, the hit masks are read with the vector
  extension and ORed into both children's bins. Every loop is limited (stack depth,
  4096 steps per packet); a breach ends the kernel with error code 2.

Correctness, every run: the best distance of every ray that hits is bit-equal to the
CPU reference (planes computed in float64 and rounded to float32, which is exact for
the random set by construction and came out equal for the camera set too); no ray
disagrees on hit or miss; no leaf index is wrong. Where two leaf boxes give the same
best distance (boxes sharing a face on the grid, or a ray starting inside two boxes
so both give 0) either index is accepted; the "ties" column counts those rays. The
CSR end test never disagreed with the core's count.

Cycles for the whole walk (median of 3 runs; the 3 runs were within 0 to 51 cycles of
each other). "Node tests" counts ray-node tests (lanes enabled, summed); "steps" are
32-lane steps in A and (node, packet) tests in B.

| rays, set | form | cycles | per ray | per ray-node test | steps | node tests | ties |
|---|---|---|---|---|---|---|---|
| 32 camera | A, vector-ext copy | 82,297 | 2572 | 120.3 | 43 | 684 | 0 |
| | A, scalar copy | 88,645 | 2770 | 129.6 | 43 | 684 | |
| | B | 91,621 | 2863 | 99.4 | 227 | 922 | |
| 32 random | A, vector-ext copy | 99,009 | 3094 | 110.5 | 49 | 896 | 1 |
| | A, scalar copy | 106,295 | 3322 | 118.6 | 49 | 896 | |
| | B | 90,138 | 2817 | 72.7 | 221 | 1240 | |
| 256 camera | A, vector-ext copy | 581,994 | 2273 | 112.2 | 288 | 5188 | 2 |
| | A, scalar copy | 624,569 | 2440 | 120.4 | 288 | 5188 | |
| | B | 315,471 | 1232 | 46.7 | 738 | 6762 | |
| 256 random | A, vector-ext copy | 896,026 | 3500 | 135.2 | 508 | 6630 | 9 |
| | A, scalar copy | 971,522 | 3795 | 146.5 | 508 | 6630 | |
| | B | 563,021 | 2199 | 63.9 | 1782 | 8818 | |
| 1024 camera | A, vector-ext copy | 2,323,116 | 2269 | 111.1 | 1146 | 20908 | 4 |
| | A, scalar copy | 2,494,004 | 2436 | 119.3 | 1146 | 20908 | |
| | B | 1,031,760 | 1008 | 37.4 | 2486 | 27616 | |
| 1024 random | A, vector-ext copy | 3,494,085 | 3412 | 134.0 | 1968 | 26068 | 25 |
| | A, scalar copy | 3,786,433 | 3698 | 145.3 | 1968 | 26068 | |
| | B | 2,160,198 | 2110 | 63.1 | 6970 | 34246 | |

Where the cycles go (same runs; A with the vector-extension copy). Setup: rays into
`Dst`, results out, stacks or bins cleared. Core work: A pops, pushes and lane
bookkeeping; B scanning bins and ORing hit masks into the children. Copy: A's node
records into the window. Vector: test pushed, run and waited for (B includes pushing
the 12 box immediates and the lane mask per test). Readout: hit masks out of `Dst`.
End test: A's `tensix_sync()` plus `qstatus` per step.

| rays, set | form | setup | core work | copy | vector | readout | end test |
|---|---|---|---|---|---|---|---|
| 32 camera | A | 3285 | 32,401 | 36,020 | 3569 | 4429 | 1959 |
| | B | 9548 | 22,311 | 0 | 43,894 | 15,769 | 0 |
| 32 random | A | 3284 | 42,716 | 41,040 | 4067 | 5047 | 2220 |
| | B | 9548 | 22,192 | 0 | 42,781 | 15,517 | 0 |
| 256 camera | A | 26,245 | 243,240 | 241,255 | 23,904 | 29,664 | 13,175 |
| | B | 32,612 | 87,354 | 0 | 130,752 | 64,650 | 0 |
| 256 random | A | 26,252 | 322,394 | 425,424 | 42,164 | 52,324 | 22,970 |
| | B | 32,614 | 116,388 | 0 | 299,268 | 114,648 | 0 |
| 1024 camera | A | 105,006 | 974,688 | 960,069 | 95,118 | 118,038 | 52,440 |
| | B | 117,888 | 281,663 | 0 | 422,945 | 209,096 | 0 |
| 1024 random | A | 105,022 | 1,268,263 | 1,648,053 | 163,344 | 202,704 | 88,946 |
| | B | 117,889 | 439,704 | 0 | 1,155,990 | 446,428 | 0 |

What the numbers say (measured unless marked inferred):

- **Which wins.** B is faster from 256 rays on: 1.84x (256) and 2.25x (1024) on the
  camera set, 1.59x and 1.62x on the random set. At 32 rays they are close: A takes
  10 % fewer cycles on the camera set, B 9 % fewer on the random set.
- **A is bound by the core.** Pops, pushes and the record copy are 83 to 85 % of A's
  cycles; the vector unit is 4 to 5 % (83 cycles per step). A step costs about 2030
  cycles (camera) or 1780 (random), of which the copy alone is 838 (vector extension,
  8 words x 32 lanes) or 986 (scalar). The vector-extension copy saves 7 to 8 %
  of the total.
- **B is bound by pushing the boxes and reading masks.** The vector category is 41 to
  54 % of B's cycles at 166 to 193 cycles per (node, packet) test; that is mostly the
  core pushing 12 box immediates and the mask, since the test itself is about 60
  vector-unit cycles (inferred from the instruction count and probe 4's 44.4 cycles
  for a compile-time box). Readouts are 17 to 21 % (probe 4: 99 cycles per mask).
- **B does 30 to 38 % more ray-node tests,** because the breadth-first sweep reaches
  every internal node of a ray before any leaf, so no box is culled by a best
  distance found earlier; A's depth-first stacks cull boxes beyond the best once a
  leaf is hit. B still wins because its step is about 5 times cheaper: 310 to 430
  cycles all in per (node, packet) test against 1780 to 2030 per A step. A's steps
  carry more lanes (13.1 to 18.3 of 32 against 4.1 to 11.1 in B), but not enough to
  make up for it; per ray-node test B costs 37 to 64 cycles from 256 rays on against
  111 to 135 for A.
- **Coherence.** The camera set halves B's cost per ray at 1024 rays (1008 against
  2110 cycles) and helps A much less (2269 against 3412).

### 7. A NoC read from a compute core

The pack core (TRISC2) of core (0, 0) reads 64 bytes from the input page of core
(1, 0), which that core's reader has filled, on NoC 0 with command buffer 1. The
program's data-movement kernels submit no NoC 0 reads while it runs. The core saves
the 8 command-buffer registers it writes, sets target (peer coordinate, address),
return (its own coordinate from `NOC_ID_LOGICAL`, an L1 buffer), length 64 and the
read control word, starts the command, polls the NoC's read-response counter
(`NIU_MST_RD_RESP_RECEIVED`) with a step limit, then restores the registers. A
"where" word is written before every step so a hang would show how far it got.

One run (the job was allowed once):

| result | value |
|---|---|
| this core / peer at NoC 0 | (1, 2) / (2, 2) |
| response counter | 50551 to 50552 |
| polls of the counter until it moved | 18 |
| cycles from starting the command to the counter moving | 209 (155 ns) |
| 16 words read | all equal to the peer's input |

The kernel finished normally and a smoke run on the same card right after was clean.
The 209 cycles is one sample, for neighbouring tiles, with an idle NoC; the counter is
shared by the core's data-movement firmware, so this only works while the
data-movement kernels submit no reads on that NoC.

### 8. Instruction costs for a tile compiler

One run of each probe (tt-metal v0.81.1, AICLK 1350 MHz before and after every run, so
one cycle is 0.741 ns; host load average 2.1). A span is taken between two
`tensix_sync()` calls; the empty span costs 10 cycles in every repetition and is
subtracted. Each figure is the median of 5 in-kernel repetitions, which agreed to
within 0.5 %. Every value the instructions computed was checked (none wrong).

**Pushing an instruction word** (probe 8, the pack thread, 4096 `SFPLOADI` a span):

| how the word reaches the instruction buffer | cycles a word |
|---|---|
| `TTI_` form (encoded into the core's instruction stream), loop of 16 | 1.128 |
| the same, loop of 64 | 1.035 |
| a constant word stored by the core (`instrn_buffer[0] = w`), loop of 16 / 64 | 1.131 / 1.036 |
| a word built at run time (add, mask to 16 bits, add, store) | 4.14 |
| a word loaded from a 64-word table in L1, then stored | 9.37 |

A store of a ready word costs what the `TTI_` form costs, one cycle; the difference
between the loops of 16 and 64 is the loop's own two instructions. A word that is built
or loaded costs the core instructions that make it.

**Replay and MOP against pushing** (probe 8, a body of 16 `SFPMAD` on 4 registers,
4096 in a span):

| form | cycles a `SFPMAD` |
|---|---|
| pushed by the core (`TTI_`, loop of 16) | 1.128 |
| `REPLAY` of the 16 recorded, pushed 256 times | 1.004 |
| `REPLAY` of 4, pushed 1024 times | 1.004 |
| MOP over the `REPLAY` of 16, or of 4 | 1.002 |
| MOP of two `SFPMAD` (loop op 0 and loop op 1) | 1.002 |

The vector unit takes one `SFPMAD` a cycle (the 2-cycle result latency is hidden by
four registers in rotation), so replay or a MOP saves only the core's loop overhead,
11 % against a loop of 16.

**Load-macro chain** (probe 9, 64 vectors of `Dst`, each `x = 1 x 1 + x` loaded,
computed and stored back, 1024 rows a span):

| form | cycles a row |
|---|---|
| `SFPLOAD`, `SFPMAD`, `SFPSTORE` for one row at a time (replay of 3) | 4.03 |
| four rows interleaved (replay of 12: 4 loads, 4 MADs, 4 stores) | 3.04 |
| one `SFPLOADMACRO` a row (template `SFPMAD` at delay 0, store at delay 2) | 1.046 |

One row at a time stalls a cycle on the MAD result; interleaving reaches one
instruction a cycle; the macro does the load, the MAD and the store of a row in one
issue slot, three times the best plain form.

**What SFPI emits** (probe 13; read from the compiled pack-thread kernel, `-O3`; each
function also has one `SFPLOAD` per input and one `SFPSTORE`):

| C++ (`sfpi::vFloat`) | vector instructions between the loads and the store |
|---|---|
| `a * b + c` | 1: `SFPMAD` |
| `a - b` | 1: `SFPADD` with its negate mode |
| `a * 1.5f + 0.25f` | 3: two `SFPLOADI` (the constants, as bfloat16) and `SFPMAD` |
| `v_if (a < b) { r = c; } v_endif;` | 3: `SFPGT` (sets the lane flags), `SFPMOV`, `SFPENCC` |

All four gave the C++ values on all 32 lanes on the card.

**`SFPCONFIG` combine modes** (probe 10): starting from `0x0F0` in the macro control
register, writing `0x0CC` with mode 2 gave `0x0FC`, mode 4 `0x0C0`, mode 6 `0x03C`; the
lane configuration register gave `0x0001` mode 2 `0x0100` = `0x0101` and `0x0101` mode 4
`0x0001` = `0x0001`. So mode 2 is OR, 4 is AND, 6 is XOR. SFPI's names
`SFPCONFIG_MOD1_AND` (2) and `SFPCONFIG_MOD1_OR` (4) say the opposite; use the numbers.
Both registers were saved before and read back equal after the probe restored them.

**Loop bodies around 512 bytes** (probe 12, the math core, TRISC1): a body of N
dependent adds plus a decrement and a branch, aligned to 64 bytes, one warm-up pass,
then 1000 timed passes:

| body bytes | 264 | 488 | 504 | 512 | 520 | 552 | 776 | 1032 | 1928 | 2088 |
|---|---|---|---|---|---|---|---|---|---|---|
| cycles a pass | 66.0 | 122.0 | 126.0 | 128.0 | 130.0 | 138.0 | 194.0 | 258.0 | 482.0 | 522.0 |

N + 2 cycles a pass at every size: one instruction a cycle, and no step at 512 bytes or
anywhere up to 2088 bytes for straight-line code.

**The order of the three cores in the matrix-unit probes.** Probes 11, 14 and 15 follow
the order of a matrix-product compute kernel on all three cores: a `Dst` section is
acquired, the unpack core hands a tile to the source registers, the math core issues its
instructions and then releases both source registers, the section is committed, and the
pack core packs it to a buffer and reads that buffer. No core waits for a word that
another writes. An earlier version unpacked a whole 32 x 32 tile (four faces, each a
hand-over of a source bank, with two banks per source register) and made the math core
wait for the unpack core's done word before releasing a bank: the unpack stalled at the
third face for a bank only the math core could free, and the program never ended. The
instruction words were never reached.

**The matrix unit** (probe 11, one core, one run of each; `Dst` in FP32, SrcA and SrcB
one bf16 tile of 1 + 2^-6, HiFi4 setting, a body of 24 words, replayed 64 times a span,
three spans; the empty span is 10 cycles and is subtracted; the pack read back every
`Dst` word and all were as the model says, 128 x K words each):

| instruction, K blocks in rotation | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| `MVMUL` (cycles a word) | 6.00 | 3.00 | 2.00 | 1.50 | 1.20 | 1.00 | 1.00 |
| `ELWMUL` | 6.00 | | | | | | 1.00 |
| `ELWADD` (with accumulate) | 6.00 | | | | | | 1.01 |
| `MOVA2D` (8 rows) | 1.00 | | | | | | 1.00 |
| `MOVB2D` (4 rows) | | | | | | | 1.00 |

- Throughput is one word a cycle for all five. A single `MVMUL` pushed alone, after the
  hand-over, takes 8 cycles from push to the sync's return (span 18, empty span 10).
- A word that adds into a `Dst` block that the instruction 1 to 5 words before it also
  wrote waits until 6 cycles after that word: a block written again after K words costs
  max(1, 6 / K) cycles a word, so rotating over 6 or more blocks hides it. `ELWMUL` and
  `ELWADD` do the same. The moves do not read `Dst` and show no stall.
- The `Dst` block is chosen by the counter and an address mode that steps it (8 rows,
  4 for `MOVB2D`); the row field of the word was not used.
- Phases: one `MVMUL` is one fidelity phase, and a phase costs one word; the number of
  phases is the number of words. The pools (`GAPOOL`, `GMPOOL`) were not measured.
- `ELWADD` with its accumulate bit adds SrcA + SrcB to `Dst` (2.03125 for two values of
  1.015625); `ELWMUL` adds their product; the moves copy.

**Ties** (probes 14 and 15, one run each). Exact sums built to lie half way between two
neighbours:

- `MVMUL` (SrcA = SrcB = a tile whose products sum to 1 + 2^-24, half an FP32 unit):
  of four such sums, three came out as 1.0 and -1.0 (the sum rounded toward zero, which
  here is also to even) and one, with the two terms in two multiplier lanes, as
  1.0000001192 (up). No one rule of round-to-nearest-even or ties-away fits all four, so
  the matrix unit does not round the exact sum once. With stochastic rounding of the
  matrix unit enabled, seven further blocks matched the nearest block at those four sums
  in all 28 cases, and 13 of the 896 words of the seven blocks differed from it.
- The packer, FP32 to BF16, on 512 ties and 512 exact values: round to nearest sent all
  512 ties to the larger magnitude (256 of them to the even neighbour: away from zero,
  not to even) and kept all 512 exact values. With stochastic rounding, ties went to the
  larger magnitude 329 times and to the smaller 183 times, and 6 of the 512 exact values
  moved to the larger magnitude.
- Values reached `Dst` whole: the tie inputs have no bit below 2^-10, which SrcA's TF32
  keeps. Ties just above or below a half were not tested.

## Limits

- One core, one card, slow dispatch; apart from the single read of probe 7 nothing here
  measures NoC, DRAM or more than one tile. The walk uses one fixed tree (128 leaves, balanced); deeper or unbalanced
  trees, and child ordering by distance in form A, were not tried.
- The vector-extension variants need `enable_trisc2_rvv` (TRISC2 only).
- Probe 2 compares the CSR with a `Dst`-window read, not with the packer path.
