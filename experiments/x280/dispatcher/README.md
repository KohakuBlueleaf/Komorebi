# Dispatcher measurements

What the dispatcher (`src/komorebi_native/x280/kernel/`) costs on a Blackhole
p150a: commands per microsecond, the host's stores into the card, loads, stores and
fences per memory of the X280, and the stream engine's share of the kernel's loop.
`measure.py` runs on the kernel's card harness in this folder (`dispatch_card.py`,
`graph_card.py`, its host module in `host/`, its stand-in worker in `kernels/` and its
run frame `run.sh`; [HARNESS.md](HARNESS.md)); build them with `build.py`, which also
builds `memcost.c`.

`python measure.py <case> [options]` prints one JSON line. Under the hang watcher:
`SCRIPT=<this folder>/measure.py ./run.sh <tag> <case>
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

`microseconds_per_command_kernel` is the kernel's time from the first command it sends to
cores or to a thread to the last event, divided by the number of commands; a command the
dispatcher serves itself (the reset list a load hands it, a ping, an inspect) does not
open that window. `microseconds_per_command_host` is the host's time
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
| `limit` | `limit` | not a timing: 126,976 bytes; the kernel image was 27,664 bytes at these timings (56,480 at 0.3.0) | |
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

The `bench` rows again on 2026-11-19, on commit 3e4c3a95: one p150a (card 1), firmware
bundle 19.13.1, table 64, two runs each, the first started at a host load of 3.1 to 3.7,
the second at 4.1 to 5.5. Since the rows above, the kernel's rare code has its own
section and its loop's rare branches are marked unlikely, and the cores' pass stops at the
ring bound the dispatcher keeps. Microseconds per command by the kernel's clock (host-fed:
the host's span agrees within 0.0001).

| what | command | 1000 MHz | 1,350 MHz |
|---|---|---|---|
| host-fed, depth 16, group 1 | `bench --n 100000 --depth 16 --group 1 --batch 64 --host-depth 64` | 1.231 / 1.229 | 0.928 / 0.916 |
| kernel only, 63 preloaded | `bench --preload --n 63 --depth 16 --group 1 --host-depth 64` | 1.274 / 1.272 | 0.961 / 0.966 |
| host-fed, depth 4, groups of 2 / 12 / all | `bench --n 100000 --depth 4 --group G --batch 64 --host-depth 64` | 1.183 / 1.180; 1.180 / 1.183; 1.214 / 1.219 | 1.375 / 0.879; 0.878 / 0.881; 1.303 / 1.300 |
| host-fed, depth 16, groups of 2 / 12 / all | `bench --n 100000 --depth 16 --group G --batch 64 --host-depth 64` | 1.225 / 1.226; 1.228 / 1.229; 1.269 / 1.265 | 0.924 / 0.930; 0.935 / 0.937; 1.513 / 1.510 |

At 1,350 MHz a command to all cores costs more than at 1000 MHz (1.30 and 1.51 against
1.21 and 1.27), most likely because the faster dispatcher leaves more commands waiting
on every core and a core's pass pays for each of its own (not measured here). The 1.375 at depth 4, groups of 2,
is one run against 0.879 beside it.

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
and the cores' scan go over `depth` slots. On 2026-11-14 that was about 7 % more per
NOP command at 16 and 21 % at 48, NOP commands never had more than 3 in flight (the
dispatcher was the bound) and MAP jobs reached 30 at depth 48.

`bench --preload --n 63` gives the same order, 0.03 to 0.06 microseconds above `bench`.
The `bench` 12-core row at table 64, depth 16 (1.488) is one run against 1.264 and 1.290
beside it.

The example's rows again on 2026-11-19, after the dispatcher's window was set to open at
its first command to cores (it had counted the host's work between the load and the first
submit) and with the example's tail writing a tensor of its own: one p150a (card 1),
firmware bundle 19.13.1, AICLK 1,350 MHz, table 64, commit 55eea5dd, host load 2.65 to
3.04 at the runs' starts. The X280 clock was set by opening the session as
`km.Session(pll_mhz=M)` around the example's own command lines above (NOP: `--handler
nop --jobs 10000 --count C --depth D --no-check`; MAP: `--jobs 1000 --groups random
--depth 48`, seed 1). Microseconds a command, two runs each: by the dispatcher's clock
(its first command to cores to its last event), then by the host's span (the first
submit to the end of the final wait); the most commands in flight in brackets.

| depth | what | X280 1000 MHz, dispatcher | host | X280 1,350 MHz, dispatcher | host |
|---|---|---|---|---|---|
| 4 | NOP, 2 cores | 1.430 / 1.428 [3] | 1.493 / 1.495 | 1.130 / 1.133 [3] | 1.196 / 1.199 |
| 4 | NOP, 12 cores | 1.435 / 1.435 [3] | 1.502 / 1.502 | 1.132 / 1.136 [3] | 1.197 / 1.202 |
| 4 | NOP, 120 cores | 1.453 / 1.454 [3] | 1.520 / 1.522 | 1.130 / 1.133 [3] | 1.197 / 1.200 |
| 16 | NOP, 2 cores | 1.495 / 1.498 [3] | 1.560 / 1.565 | 1.173 / 1.163 [3] | 1.238 / 1.229 |
| 16 | NOP, 12 cores | 1.509 / 1.507 [3] | 1.576 / 1.573 | 1.165 / 1.167 [3] | 1.231 / 1.232 |
| 16 | NOP, 120 cores | 1.506 / 1.508 [3] | 1.572 / 1.579 | 1.168 / 1.163 [4] | 1.235 / 1.230 |
| 48 | NOP, 2 cores | 1.644 / 1.643 [3] | 1.714 / 1.709 | 1.982 / 1.980 [27] | 2.048 / 2.048 |
| 48 | NOP, 12 cores | 1.643 / 1.642 [3] | 1.709 / 1.708 | 1.983 / 1.983 [27] | 2.050 / 2.051 |
| 48 | NOP, 120 cores | 1.643 / 1.652 [3] | 1.710 / 1.719 | 1.981 / 1.980 [27] | 2.048 / 2.045 |
| 48 | random MAP | 2.003 / 2.002 [11] | 2.156 / 2.175 | 2.072 / 2.062 [34, 33] | 2.241 / 2.224 |

Every MAP run checked its tail (jobs 962 to 999) with no job wrong. At depth 48 with the
X280 at 1,350 MHz, NOP commands reached 27 in flight and cost more than at 1000 MHz
(1.98 against 1.64), so there the dispatcher was not the bound; what was is not
measured here.

## With the ring bound

The cores' pass now stops at the ring bound the dispatcher keeps in their control block
(see [the kernel's README](../../../src/komorebi_native/x280/kernel/README.md#the-ring-bound)):
it is raised in steps of 8 before a command lands at or above it and lowered a step when
no busy slot is within two steps under it and it has held 20 microseconds. The table
above again with it, on 2026-11-19: one p150a (card 1), AICLK 1,350 MHz, table 64, the
same command lines (NOP on 12 cores, `--count 12`), commit 3e4c3a95. The first run of
each row started at a host load of 3.3 to 3.7, the second at 5.4 to 8.0 (one row's first
at 3.7, its second at 8.0); the two agree within 0.02 except where both are given.

| depth | what | X280 1000 MHz, dispatcher | host | X280 1,350 MHz, dispatcher | host |
|---|---|---|---|---|---|
| 4 | NOP, 12 cores | 1.353 / 1.351 [3] | 1.422 / 1.417 | 1.075 / 1.075 [3] | 1.141 / 1.141 |
| 4 | random MAP | 1.755 / 1.775 [4] | 1.942 / 1.959 | 1.717 / 1.720 [4] | 1.888 / 1.891 |
| 16 | NOP, 12 cores | 1.429 / 1.432 [3] | 1.496 / 1.499 | 1.107 / 1.116 [4, 3] | 1.173 / 1.184 |
| 16 | random MAP | 1.734 / 1.856 [11, 12] | 1.920 / 2.018 | 1.648 / 1.645 [16] | 1.813 / 1.815 |
| 48 | NOP, 12 cores | 1.559 / 1.557 [3] | 1.626 / 1.624 | 1.211 / 1.212 [8] | 1.279 / 1.279 |
| 48 | random MAP | 1.837 / 1.834 [10] | 2.000 / 2.009 | 1.703 / 1.688 [20] | 1.870 / 1.864 |
| 64 | NOP, 12 cores | 1.615 / 1.614 [4] | 1.682 / 1.681 | 1.265 / 1.261 [13] | 1.334 / 1.328 |
| 64 | random MAP | 1.885 / 1.868 [10] | 2.065 / 2.046 | 1.677 / 1.677 [18, 17] | 1.861 / 1.852 |

Every MAP run checked its tail with no job wrong. At 1,350 MHz NOP commands at depth 48
cost 1.98 microseconds before the bound (the table above) and 1.21 with it; at depth 64,
2.60 in a run of the same day before it (host load 13) and 1.26 with it.

What the bound leaves: the dispatcher's own sweep of the DONE table still goes over every
slot up to the depth, so NOP commands at depth 48 and 64 still cost more than at depth 16
(the cores' scan no longer does). Bounding the sweep as well lowers that but lets more
MAP commands into flight; the two are to be weighed together.
