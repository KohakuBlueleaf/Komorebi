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
5. the speed of a compute RISC-V core on its own, alone and three at once.

Every result value is checked against a CPU reference by the driver.

## How it runs

One program on core (0, 0), launched with `ttnn.generic_op` in slow dispatch:

| part | file |
|---|---|
| reader (RISCV_1): the input page from DRAM into CB 0 | `kernels/reader.cpp` |
| compute: one probe per build (define `PROBE`), output into CB 16 | `kernels/probe.cpp` |
| vector-unit probes 0 to 4, run by the pack thread (TRISC2) | `kernels/vec.hpp` |
| RISC-V loop probe 5, run by the TRISCs selected in a mask | `kernels/rv.hpp` |
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
  Lane `l` of the 4-row group at `Dst` address `a` is window byte `64 a + 8 l`. The
  smoke probe checks the round trip bit for bit (denormals, -0, infinities, NaN, all
  ones) and that `LReg[15]` holds `2 x lane`.
- **Ordering.** Window stores are followed by a load of the last word written (same
  memory region, processed in order) before the vector unit is told to load them.

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
```

Without the hang watcher: `python probes.py <probe> [options]` with tt-metal's Python
environment and `TT_METAL_SLOW_DISPATCH_MODE=1`. Each run prints one JSON line;
`nonce_ok` and `kernel_error == 0` say the kernel finished cleanly, and every probe
reports its count of wrong values. Variant 2 sets `ComputeConfig.enable_trisc2_rvv`.

## Results

tt-metal v0.81.1, one Blackhole card, AICLK 1350 MHz. Each number is the median over
the in-kernel repetitions of three separate runs; the three runs agreed to within the
min/max given. Host load average 2.4 to 12 during the runs.

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

Lane i's counter starts at 2i and the body subtracts 2 with a condition, so lane i
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
  exposed (about 17 cycles per window load). The scheduled loop issues 8 loads
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

## Limits

- One core, one card, slow dispatch; nothing here measures NoC, DRAM or more than one
  tile.
- The vector-extension variants need `enable_trisc2_rvv` (TRISC2 only).
- Probe 2 compares the CSR with a `Dst`-window read, not with the packer path.
