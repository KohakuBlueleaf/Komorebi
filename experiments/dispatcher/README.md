# Dispatcher measurements

What the dispatcher (`src/komorebi_native/x280/kernel/`) costs on a Blackhole
p150a: commands per microsecond, the host's stores into the card, loads, stores and
fences per memory of the X280, and the stream engine's share of the kernel's loop.
`measure.py` runs on the harness of the kernel's card tests
(`tests/hardware/dispatcher`: its host module, its stand-in worker and its run
frame); build them with that folder's `build.py`, which also builds `memcost.c` here.

`python measure.py <case> [options]` prints one JSON line. Under the hang watcher:
`SCRIPT=<this folder>/measure.py ../../tests/hardware/dispatcher/run.sh <tag> <case>
[options]`.

| case | what it measures |
|---|---|
| `limit` | The largest active image cluster 0 accepts, read with the runtime's own check, and the kernel image's size |
| `bench` | `--n` NOP commands at `--depth` to groups of `--group` cores (0 = all), then EXIT: microseconds per command, latency, idle and full passes; every event and every core count is checked |
| `store` | No kernel: `--n` bursts of `--lines` 64-byte stores into LIM (or, with `--target l1`, a Tensix core's free L1) and one fence each, for every idle gap in `--gaps-microseconds`, keep-busy period in `--busy-microseconds` (a second thread storing one line per period; 0 = none) and number of burst positions in `--spread`; `--read` reads LIM through the runtime before each burst. Reports the time in the stores and in the fence of each burst |
| `memcost` | `memcost.c` alone (no grid): nanoseconds per load, store and fence on the stack (cluster local memory, uncached), in cached DRAM of this cluster's bank (through the Memory Port, in a DRAM buffer the harness owns), and in registers |
| `stream-bench` | The kernel's pass period with `--streams` streams that only loop, idle for `--seconds` (`--n 0`), or with `--submitters` streams each submitting `--n` commands one at a time on its own `--group` cores; then, with `--compute`, a stream of 2 x `--compute` micro-ops alone. Pass period from the result block, time per command from the streams, free-to-resubmit from the test image's trace |

`bench` measures in two ways:

- **Kernel only**, with `--preload`: all `--n` + 1 commands are stored in the host ring
  before the kernel starts. `--n` + 1 must fit in `--host-depth` (at most 64).
- **Host-fed**: the host stores every command that fits in the ring, up to `--batch`
  commands, then fences once. `--settle-microseconds` makes it wait that long after a fence
  before it stores the next batch: on the host these numbers were taken on, a store
  into the card's window submitted within about 0.5 microseconds after a fence often
  stalls for about 12 microseconds (the `store` case shows it).

`microseconds_per_command_kernel` is the kernel's time from the first command it takes to the
last event, divided by the number of commands. `microseconds_per_command_host` is the host's time
from the first store to the last event read, divided the same way.
`microseconds_store_per_command` is the share of that host time spent in ring stores and fences.

## Results

One p150a, AICLK 1,350 MHz, X280 timer at 50 MHz, host load 4 to 11, 2026-11-13. The
X280 runs at 1000 MHz, the session's and this script's default (`--pll`); the last
column is the same measurement at the 1750 MHz opt-in (`--pll 1750`), taken earlier
with the same command unless the row says otherwise. Every `measure.py` line logs the
clock it set.

| what | command | 1000 MHz | 1750 MHz (opt-in) |
|---|---|---|---|
| `limit` | `limit` | not a timing: 126,976 bytes; the kernel image is 27,664 bytes | |
| `bench` host-fed, depth 16, group 1, 100,000 commands, batch 64 | `bench --n 100000 --depth 16 --group 1 --batch 64 --host-depth 64` | 1.239 microseconds per command (1.228 in a run beside the 1750 one) | 0.641 (0.726 in that run) |
| `bench` kernel only, 63 preloaded | `bench --preload --n 63 --depth 16 --group 1 --host-depth 64` | 1.263 | 0.669 |
| `memcost`, nanoseconds per operation | `memcost` | uncached local memory load 30.0, store 6.4, store and fence 30.0; cached DRAM load 5.0, store 4.0; `fence ir` or `fence ow` with nothing outstanding 2.0; timer read 18.0 | 17.6, 3.7, 17.1; 2.9, 2.3; 1.1; 10.3 |
| `store` (the host's stores; the X280 clock plays no part) | `store --gaps-microseconds 0 1 --n 200` | a 64-byte store into the window submitted right after a fence stalls for about 12 microseconds in most bursts (median 12.5 per burst of 2); after a 1 microsecond gap a line costs about 0.18 microseconds | the same |
| `stream-bench`, pass period of core 0, no stream | `stream-bench --n 0 --seconds 1` | 60 ns | 32 |
| the same with one or four streams that only loop | `stream-bench --n 0 --streams 1` (or `4`) `--seconds 1` | 86 ns | 49 |
| one or four submitting streams, 1,000 commands each on its own core | `stream-bench --n 1000 --submitters 1` (or `4`) `--group 1` | pass 76 / 279 ns; 3.5 / 7.7 microseconds per command; free to next submit 2.4 microseconds (median, one stream) | pass 89 / 159 ns; 2.3 microseconds per command (one stream); 1.34 |
| one submitting stream beside three looping ones | `stream-bench --n 1000 --submitters 1 --streams 3 --group 1` | free to next submit 23.9 microseconds (median) | 12.8 |
| a stream that only computes | `stream-bench --n 0 --compute 1000000` | 7.6 million micro-ops a second (the same with `--compute 100000`) | 18 million (rounds not recorded) |
| `stream-bench` before the engine moved to core 1 (one loop) | an earlier kernel | not re-measured | `bench` as above: 0.669 and 0.675 microseconds per command while the stream interpreter shared core 0's loop; 34 ns with no stream; 316 ns with one submitting stream, 525 ns with four; 3.1 microseconds with one looping stream, 11.7 with four |

## Ring slots and depth

How much a command costs with the ring and DONE tables compiled for 16, 32 and 64
slots (`KM_RING_SLOTS` in `km_abi.h`, the dispatcher image and the runtime rebuilt for
each), and with 4, 16 and 48 of them in use (the program's `depth`). One p150a, AICLK
1,350 MHz, X280 at 1000 MHz, host load 4 to 7, 2026-11-14. Microseconds per command
from the kernel's first command to its last event: `bench` host-fed (`bench --n 100000
--depth D --group G --batch 64 --host-depth 64`, groups of 2 and 12 cores and the whole
grid), and `examples/01_mapcollect` with NOP jobs (`--handler nop --jobs 10000 --count C
--depth D --no-check`) and with its random MAP groups (`--jobs 1000 --groups random
--depth D`, the most commands in flight in brackets).

| depth | what | table 16 | table 32 | table 64 |
|---|---|---|---|---|
| 4 | `bench`, 2 / 12 / all cores | 1.181 / 1.181 / 1.223 | 1.184 / 1.185 / 1.223 | 1.186 / 1.187 / 1.224 |
| 4 | example 01 NOP, 2 / 12 / 120 cores | 1.350 / 1.358 / 1.360 | 1.353 / 1.350 / 1.357 | 1.356 / 1.354 / 1.361 |
| 4 | example 01 random MAP | 1.817 [4] | 1.737 [4] | 1.764 [4] |
| 16 | `bench` | 1.228 / 1.229 / 1.272 | 1.240 / 1.239 / 1.281 | 1.264 / 1.488 / 1.290 |
| 16 | example 01 NOP | 1.413 / 1.430 / 1.441 | 1.414 / 1.430 / 1.455 | 1.426 / 1.433 / 1.466 |
| 16 | example 01 random MAP | 1.766 [12] | 1.787 [12] | 1.798 [11] |
| 48 | `bench` | | | 1.432 / 1.464 / 1.466 |
| 48 | example 01 NOP | | | 1.644 / 1.635 / 1.645 |
| 48 | example 01 random MAP | | | 2.138 [30] |

At 4 in flight the 64-slot table costs under 0.5 % more than the 16-slot one (the
random rows move by 4 % between runs either way), so the table has 64 slots. What a
command costs grows with the depth in use, not with the table: the dispatcher's sweep
and the cores' scan go over `depth` slots, about 7 % more per NOP command at 16 and
21 % at 48. NOP commands never have more than 3 in flight (the dispatcher is the
bound); MAP jobs reach 30 at depth 48. `bench --preload --n 63` gives the same order,
0.03 to 0.06 microseconds above `bench`. The `bench` 12-core row at table 64, depth 16
(1.488) is one run against 1.264 and 1.290 beside it.
