# A dispatcher on one Tensix core, fed live

The test feeder (`src/komorebi/tools/feeder/`) made live: core 119 (the grid's last
core) plays the dispatcher while the host sends commands and reads one event a command,
with the X280 dispatcher's protocol with the host (`src/komorebi_native/x280/kernel/README.md`,
"Protocol with the host"). Cores 0 to 118 run the map/collect example's kernels, as under
the feeder. What one Tensix core costs as the dispatcher, beside the X280's `bench`
(`experiments/x280/dispatcher/`).

| part | file |
|---|---|
| The loop: take, slots, sweep, events, the stop flag | `kernels/dispatcher.cpp` (a copy of `feeder.cpp`, made live) |
| The second mover (V3): one rectangle's multicast, handed over in a word pair | `kernels/helper.cpp` |
| Arguments, the live block in L1, the host lines, the results | `kernels/live.h` |
| Host driver: the program (as `fed.py` builds it), the feed, the checks; one JSON line a run | `live.py` |

## How it works

- **Commands in.** A ring of `--host-depth` 64-byte lines (a power of two, at most 64)
  in the live block, an L1 area of every core of which only the dispatcher's is used. The
  host writes command i (from 1) into line i % host_depth with i in the four landing words,
  once the consumed count shows i - host_depth. It writes each run of consecutive lines that
  does not wrap in one call of the device module's `write_l1` (tt-metal's
  `WriteToDeviceL1`), and the read count the same way. The dispatcher takes a line only
  when all four landing words show its next number (read after `invalidate_l1_cache()`),
  copies it and then publishes the consumed count.
- **Events out.** For every finished command one 16-byte event {index, request, status,
  ticks} goes to the session's region of the host channel (offset 0: the counts line
  {consumed, told}, then the event ring of `--event-depth`, at most 128) with one
  `noc_async_write_pcie`. The counts line goes once a pass when either count moved. The
  host reads the ring in place: the events (each index before its fields), then the index
  column again, and keeps an event only when both reads give its number. A command is
  taken only while (pushed - read + busy slots + 1) fits the event ring.
- **As the feeder**: slots, sequence numbers, EXCLUSIVE, RESULT, EXIT, the deadline and the
  stop flag. A group outside the grid or flag bit 0 is answered with `KM_EVENT_REJECTED`
  (3, 2). An idle limit (`--idle`, nothing taken nor busy) writes the stop flag and ends the
  run, so a host that stops feeding cannot leave the card spinning.
- **Forms of the command line** (`--form`): `unicast`, the feeder's loop of 119 writes
  (U); `multicast`, one rectangle a panel of consecutive NoC 0 columns, the dispatcher's
  panel holding the sender (plain multicast, receivers not counting it) (M); `loopback`,
  the same with `noc_async_write_multicast_loopback_src`; `apart`, three rectangles
  without the sender.
- **Processors** (`--mover`): `brisc` (V1, the feeder's processor, NoC 0); `ncrisc` (V2,
  NoC 1); `both` (V3: the loop on BRISC, `helper.cpp` on NCRISC multicasting rectangle 1
  on NoC 1 when the loop writes sequence << 8 | slot into the word pair, the loop waiting
  for it in the second word); `both-ncrisc` the other way. `--pace N` puts N empty cycles
  between two polls of every polling loop (the helper's, the loop's wait for it, an idle
  pass of the loop).
- `--parts` builds the loop with wall-clock reads around each part (take, slot, write,
  sweep, event, and an empty look at the ring); they add about 10 % to a command.

`python live.py <case> [options]`, from the repository root on a card in slow dispatch:

| case | what it measures |
|---|---|
| `bench` | `--n` NOP commands on `--group` cores (0: every worker), `--exclusive`, host-fed in batches of at most `--batch`, then EXIT; `--runs` in one device open |
| `preload` | `--n` NOP commands and EXIT written before the launch (all must fit the ring): the dispatcher only |
| `latency` | one command at a time: submit to ack on the device; from the host, write to consumed count, to the event's count (`told`), to the event read |
| `map` | `--n` MAP jobs of the map/collect example on random groups, with `mapcollect_fed.py`'s checks (`jobs_bad`, `ranks_bad`) |

Every run checks every event (none lost, doubled, stale (an index over an earlier event's
fields) or inconsistent (a DONE whose sequence bits are not its request's)) and every
worker's run count.

## Results

One Blackhole p150a (card 1), AICLK 1,350 MHz (sysfs, every job), slow dispatch, tt-metal
v0.81.1, host driver in Python 3.13, host load 40 to 74 on 96 cores, 2026-11-19.
Microseconds a command on the device: the dispatcher's first take to its last event,
divided by the commands; the host's (write to last event read) agrees within 0.01 us in
every host-fed row. NOP commands on core 0, depth 16, batch 64, host_depth 64,
event_depth 128 unless the row says otherwise; in brackets the most commands in flight.
V3: loop on BRISC, helper on NCRISC, `--pace 16`.

| what | command | V1 U | V1 M | V2 U | V2 M | V3 M |
|---|---|---|---|---|---|---|
| B1 host-fed, 100,000 | `bench --n 100000` | 4.282 to 4.293 (5 runs) [1] | 3.567, 3.567 [16]; 1.100 [3] with `--parts` | 4.303, 4.295 [1] | 1.043, 1.038 [4] | 3.566, 3.572 [16] |
| B2 dispatcher only, 63 preloaded | `preload --n 63` | 4.366 [1] | 2.680 [16] | 4.374 [1] | 1.087 [4] | 2.872 [16] |
| B3 EXCLUSIVE NOP on all 119, 10,000 | `bench --n 10000 --group 0 --exclusive` | 6.115 | 3.175 | 6.123 | 3.057 | 2.984 |
| B4 one command on 8 cores, submit to ack on the device (median of 1,000) | `latency --n 1000 --group 8` | 3.721 | 2.395 | 3.722 | 2.284 | 2.010 |
| B4 from the host: write to the event's count in host memory / to the event read by the driver | the same | 5.62 / 18.2 | 4.00 / 16.3 | 5.74 / 17.9 | 4.04 / 16.5 | 3.71 / 15.9 |
| B1 at depth 4 | `bench --n 100000 --depth 4` | 4.310 [1] | 0.922 [3] | 4.225 [1] | 0.981 [3] | 1.243 [4] |
| B2 at depth 4 | `preload --n 63 --depth 4` | 4.385 [1] | 0.939 [3] | 4.311 [1] | 0.995 [3] | 1.238 [4] |

The X280 dispatcher beside, `experiments/x280/dispatcher/measure.py bench` on the same
card the same day (`--n 100000 --group 1 --batch 64 --host-depth 64`, and `--preload --n
63`), AICLK 1,350 MHz during every open (sysfs), at most 2 in flight in every run; host
load 0.7 (1000 MHz) and 10 (1350 MHz):

| X280 clock | B1 depth 16 | B2 depth 16 | B1 depth 4 | B2 depth 4 |
|---|---|---|---|---|
| 1000 MHz (the script's default) | 1.349 | 1.391 | 1.293 | 1.336 |
| 1350 MHz (`--pll 1350`, the session's default) | 1.003 | 1.041 | 0.962 | 1.003 |

At host load 3 to 5 this experiment's rows came out within 0.02 us of the table's: V1 U
4.286 / 4.368, V1 M 3.558 [16] / 2.678 [16], V2 M 1.045 [4] / 1.086 [4], V1 M at depth 4
0.942 [4] / 0.933 [3] (B1 / B2).

- **B5 events.** 400,000 commands in four runs of 100,000 in one device open (V1 U, depth
  16; and V1 M, depth 4): 400,004 events each, 0 stale, 0 inconsistent, 0 lost, 0 doubled.
- **B6 under load.** `map --n 1000 --depth 16 --seed 7` (random groups, 64 tiles a job, 32
  output tensors): V1 U jobs_checked 1000, jobs_bad 0, ranks_bad [], 4.346 us a job [4];
  V1 M the same checks, 2.562 us a job [16].
- **The host's write.** One `write_l1` of one 64-byte line costs the host 1.93 us (median
  of 10,001 calls, `bench --n 10000 --single`); a run of 18 lines 3.4 us. The 5 us
  threshold for a write-combining window in the host module is not reached.
- **Multicast forms** (B1 with `--parts`, V1): two rectangles with the sender inside its
  panel's (plain multicast, 59 receivers) 1.100 us [3]; the same with `loopback_src` (60)
  1.107 [3]; three rectangles without the sender 1.363 [3]. A rectangle may hold the
  sending core in both forms; every worker's run count was right in each.
- **V3 at depth 16** with `--pace 0` 3.562 [16]; the loop on NCRISC (`both-ncrisc`) 3.557
  [16], 1.215 [4] at depth 4. The loop waits 29 to 36 ticks a command for the helper.

### Ticks of each part of the loop

`bench --n 100000 --depth 4 --parts`, ticks a command (sums over the run divided by the
commands). Take: the landing words and the copy of the line; slot: the checks and the
free slot; write: the command line to the workers, flushed; sweep: the busy slots' ack
words; event: the event's and the counts line's writes to the host.

| | us a command | take | slot | write | sweep | event | passes a command |
|---|---|---|---|---|---|---|---|
| V1 U | 4.305 [1] | 278 | 97 | 5,035 | 169 | 190 | 1.04 |
| V2 U | 4.302 [1] | 278 | 97 | 5,009 | 169 | 209 | 1.05 |
| V1 M | 1.056 [3] | 279 | 107 | 450 | 226 | 302 | 1.17 |
| V2 M | 1.116 [3] | 280 | 108 | 451 | 220 | 403 | 1.06 |
| V3 M | 1.011 [3] | 278 | 108 | 493 (36 of it waiting for the helper) | 232 | 196 | 1.14 |

Write is issue to flushed (`noc_async_writes_flushed`: the requests have left the core),
with the copy of the line into its slot and the sequence numbers; five_processors' 0.547
us for one multicast is issue to acknowledged (`noc_async_write_barrier`), a round trip
the loop does not wait for. A pass of the loop polls L1 3.4 (V1, V2) to 4.9 (V3) times:
the ring line's landing words once, each busy slot's ack word once, the read count only
when the event ring looks full, and in V3 the helper's done word.

V3 a line (`--parts`, depth 4, loop on BRISC / on NCRISC), ticks: the loop's token to the
helper's look 62 / 54, the helper's multicast issued and flushed 51 / 54, its done word to
the loop's look 58 / 47, the loop's wait after its own multicast 72 / 57; the helper polls
the word pair 19 times a line (16 empty cycles apart). The hand-over costs about what one
multicast's issue saves.

### Where a worker's time goes past 3 in flight

V1 M, NOP commands, `bench --n 20000 --form multicast` with the options of each row; us a
command [most in flight].

| what | options | us a command |
|---|---|---|
| depth 1, 2, 3, 4 | `--depth D` | 1.808 [1], 1.034 [2], 0.900 [3], 0.902 [3] |
| depth 6, 8, 12, 16 | `--depth D` | 1.624 [6], 2.027 [8], 2.794 [12], 3.614 [16] |
| depth 32, 64 | `--depth D` | 6.621 [32], 12.486 [64] |
| (i) all on core 0, depth 16 | `--depth 16` | 3.630 [16] |
| (ii) 16 single cores in turn | `--depth 16 --spread 16` | 0.967 [4] |
| (iii) 14 disjoint groups of 8 in turn | `--depth 16 --group 8 --spread 14` | 0.965 [4] |
| (iv) every command on all 119, not exclusive (overlapping groups are accepted) | `--depth 16 --group 0` | 3.627 [16] |
| the dispatcher's cap on commands in flight | `--depth 16 --in-flight 3` / `--depth 64 --in-flight 3` / `--depth 64 --in-flight 16` | 1.058 [3] / 1.779 [3] / 4.190 [16] |

Inside core 0 (`--n 2000 --depth 16 --trace 2000`, the runtime's per-command trace and
the dispatcher's submit and ack times; medians, us; in flight 1 / 4 / 16 by
`--in-flight 1`, `4`, none): line written to the reader's take 0.966 / 4.222 / 55.64;
from one take on core 0 to the next 2.154 / 1.386 / 3.636; take to the writer's done
0.445 / 0.456 / 0.456; done to the ack sent 0.340 / 0.344 / 0.341; ack sent to the
dispatcher seeing it 0.111 / 0.151 / 0.317; reader passes a take 6 / 1 / 1.

A command's own work is the same at any number in flight; the time between two takes
grows, in one pass of the reader's scan (`km_worker.hpp:416`, `km::scan` in
`km_command.hpp:139`): every slot whose sequence is newer than the core's last for it is
copied whole (`km_command.hpp:153`, 16 volatile loads and stores) and checked landed,
and only a command of another group is marked done there (`km_command.hpp:161`). A
core's own pending commands are copied again on every pass until each runs: about 250
ticks for each, every take (the slope from 6 to 64 in flight). Each pass also reads every
slot's sequence and last word up to the depth, about 20 ticks a slot (depth 64 against
16 at the same 3 or 16 in flight). Spread over cores, no core holds more than one of its
own, so the dispatcher stays the bound.

### What bounds a command

- U: the 119 unicast writes, 42 ticks a write on either mover (five_processors X3(c): 31
  to 32 on NCRISC, 40 to 44 on BRISC; this loop loads each core's address from L1).
- M: about 1,360 ticks a command on one core: the line copy and its checks (take, slot)
  28 %, the two multicasts 33 %, the sweep 17 %, the two PCIe writes 22 %.
- Past 3 in flight the workers bound it: a NOP command takes 1.24 us at 4 in flight, 2.03
  at 8, 3.57 at 16 (V1 M, depth 4, 8, 16), so a faster loop that fills the ring gets fewer
  commands through (V3, V1 M without `--parts`).
- The host: a Python poll of the event ring costs about 15 us at this load; the driver
  keeps up with 0.9 us a command only because it reads up to 64 events a poll.
