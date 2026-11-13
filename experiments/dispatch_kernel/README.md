# Dispatch kernel measurements

What the dispatch kernel (`src/komorebi_native/x280/kernel/`) costs on a Blackhole
p150a: commands per microsecond, the host's stores into the card, loads, stores and
fences per memory of the X280, and the stream engine's share of the kernel's loop.
`measure.py` runs on the harness of the kernel's card tests
(`tests/hardware/dispatch_kernel`: its host module, its stand-in worker and its run
frame); build them with that folder's `build.py`, which also builds `memcost.c` here.

`python measure.py <case> [options]` prints one JSON line. Under the hang watcher:
`SCRIPT=<this folder>/measure.py ../../tests/hardware/dispatch_kernel/run.sh <tag> <case>
[options]`.

| case | what it measures |
|---|---|
| `limit` | The largest active image cluster 0 accepts, read with the runtime's own check, and the kernel image's size |
| `bench` | `--n` NOP records at `--depth` to teams of `--team` cores (0 = all), then EXIT: microseconds per command, latency, idle and full passes; every event and every core count is checked |
| `store` | No kernel: `--n` bursts of `--lines` 64-byte stores into LIM (or, with `--target l1`, a Tensix core's free L1) and one fence each, for every idle gap in `--gaps-us`, keep-busy period in `--busy-us` (a second thread storing one line per period; 0 = none) and number of burst positions in `--spread`; `--read` reads LIM through the runtime before each burst. Reports the time in the stores and in the fence of each burst |
| `memcost` | `memcost.c` alone (no grid): nanoseconds per load, store and fence on the stack (cluster local memory, uncached), in cached DRAM of this cluster's bank (through the Memory Port, in a DRAM buffer the harness owns), and in registers |
| `stream-bench` | The kernel's pass period with `--streams` streams that only loop, idle for `--seconds` (`--n 0`), or with `--issuers` streams each issuing `--n` records one at a time on its own `--team` cores; then, with `--compute`, a stream of 2 x `--compute` micro-ops alone. Pass period from the result block, time per record from the streams, free-to-reissue from the test image's trace |

`bench` measures in two ways:

- **Kernel only**, with `--preload`: all `--n` + 1 commands are stored in the host ring
  before the kernel starts. `--n` + 1 must fit in `--host-depth` (at most 64).
- **Host-fed**: the host stores every command that fits in the ring, up to `--batch`
  commands, then fences once. `--settle-us` makes it wait that long after a fence
  before it stores the next batch: on the host these numbers were taken on, a store
  into the card's window issued within about 0.5 microseconds after a fence often
  stalls for about 12 microseconds (the `store` case shows it).

`us_per_command_kernel` is the kernel's time from the first command it takes to the
last event, divided by the number of commands. `us_per_command_host` is the host's time
from the first store to the last event read, divided the same way.
`us_store_per_command` is the share of that host time spent in ring stores and fences.

## Results

One p150a, AICLK 1,350 MHz, X280 timer at 50 MHz, host load 4 to 11, 2026-11-13. The
X280 runs at 1000 MHz, the session's and this script's default (`--pll`); the last
column is the same measurement at the 1750 MHz opt-in (`--pll 1750`), taken earlier
with the same command unless the row says otherwise. Every `measure.py` line logs the
clock it set.

| what | command | 1000 MHz | 1750 MHz (opt-in) |
|---|---|---|---|
| `limit` | `limit` | not a timing: 126,976 bytes; the kernel image is 27,664 bytes | |
| `bench` host-fed, depth 16, team 1, 100,000 records, batch 64 | `bench --n 100000 --depth 16 --team 1 --batch 64 --host-depth 64` | 1.239 microseconds per record (1.228 in a run beside the 1750 one) | 0.641 (0.726 in that run) |
| `bench` kernel only, 63 preloaded | `bench --preload --n 63 --depth 16 --team 1 --host-depth 64` | 1.263 | 0.669 |
| `memcost`, nanoseconds per operation | `memcost` | uncached local memory load 30.0, store 6.4, store and fence 30.0; cached DRAM load 5.0, store 4.0; `fence ir` or `fence ow` with nothing outstanding 2.0; timer read 18.0 | 17.6, 3.7, 17.1; 2.9, 2.3; 1.1; 10.3 |
| `store` (the host's stores; the X280 clock plays no part) | `store --gaps-us 0 1 --n 200` | a 64-byte store into the window issued right after a fence stalls for about 12 microseconds in most bursts (median 12.5 per burst of 2); after a 1 microsecond gap a line costs about 0.18 microseconds | the same |
| `stream-bench`, pass period of core 0, no stream | `stream-bench --n 0 --seconds 1` | 60 ns | 32 |
| the same with one or four streams that only loop | `stream-bench --n 0 --streams 1` (or `4`) `--seconds 1` | 86 ns | 49 |
| one or four issuing streams, 1,000 records each on its own core | `stream-bench --n 1000 --issuers 1` (or `4`) `--team 1` | pass 76 / 279 ns; 3.5 / 7.7 microseconds per record; free to next issue 2.4 microseconds (median, one stream) | pass 89 / 159 ns; 2.3 microseconds per record (one stream); 1.34 |
| one issuing stream beside three looping ones | `stream-bench --n 1000 --issuers 1 --streams 3 --team 1` | free to next issue 23.9 microseconds (median) | 12.8 |
| a stream that only computes | `stream-bench --n 0 --compute 1000000` | 7.6 million micro-ops a second (the same with `--compute 100000`) | 18 million (rounds not recorded) |
| `stream-bench` before the engine moved to core 1 (one loop) | an earlier kernel | not re-measured | `bench` as above: 0.669 and 0.675 microseconds per record while the stream interpreter shared core 0's loop; 34 ns with no stream; 316 ns with one issuing stream, 525 ns with four; 3.1 microseconds with one looping stream, 11.7 with four |
