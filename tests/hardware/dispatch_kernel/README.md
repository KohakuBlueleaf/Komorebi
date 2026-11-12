# Dispatch kernel tests on a card

These tests run the dispatch kernel (`src/komorebi_native/x280/kernel/`) on X280
cluster 0 of a Blackhole p150a and feed it from the host. A stand-in Tensix worker runs
on every core of the grid. It implements only what the kernel needs: the start-up
check-in, taking a record by its landing words, team membership, completion through the
team's leader, the stop word, and EXIT. Each core counts what it ran. The host reads
the counters back only after the X280 runtime is closed.

| part | file |
|---|---|
| Host module: X280 runtime, LIM layout, host command ring, event ring, result block | `host/dispatch_test.cpp` |
| Stand-in worker, one data-movement kernel per core | `kernels/worker.cpp` |
| Its test segments, argument function and counter indices | `kernels/dk_test.h` |
| Driver, one case per run | `dispatch_card.py` |
| Measuring image: cost of loads, stores and fences per memory | `x280/memcost.c` |
| Builds the kernel image, the measuring image and the host module | `build.py` |
| One run under a hang watcher | `run.sh` |

## Cases

The kernel keeps its state in this cluster's DRAM bank (cached); every case that runs
the kernel gives it 4 KiB of a DRAM buffer the test owns.

The kernel takes a command only while its event has room in the event ring
(`--event-depth`, 128 by default, a power of two like `--host-depth`). For that the host
module stores the number of events it has read into the kernel's local memory with every
batch of commands, and while it waits whenever the kernel could be holding a command for
it (command i can wait only while i minus that number exceeds the ring). The kernel's
idle limit is off unless `--idle-s` sets one.

`python dispatch_card.py <case> [options]` prints one JSON line. `ok` is true when the
case's checks hold.

| case | what it does | `ok` when |
|---|---|---|
| `limit` | Reads the largest active image cluster 0 accepts, using the runtime's own check | (no `ok`: it reports `max_image_bytes` and `image_bytes`) |
| `ping` | Sends one `KM_SEG_PING`, then EXIT to the whole grid, then `KM_SEG_STOP` | the PONG carries the PING's command number; every core finished |
| `issue` | Like `ping`, plus one CHECK record to the whole grid | the event carries the record's command number; no argument errors |
| `teams` | `--n` CHECK records with random teams at depth 16, fed in batches | every event arrives once; every core's run count, key sum and key mix match |
| `disjoint` | A DELAY of `--delay-us` on the first half of the grid, then `--n` CHECK records on the second half | the second team's events all arrive before the first's; two slots were busy at once; every core's counts match |
| `stall` | `KM_SEG_DEADLINE` with `--deadline-us`, then a record whose member `--core` waits for the stop word | `KM_EV_TIMEOUT` for that slot and command; every core gave up on the stop word; only `--core` did not report the record |
| `stall-leader` | Like `stall`, with the team's leader (core 0) as the core that waits | the same, with core 0 the one that did not report |
| `fill` | 16 DELAY records of `--delay-us` on 16 disjoint teams, then `--n` CHECK records with random teams | all 16 slots were busy at once; every event and every core count is right |
| `exit-busy` | A DELAY on half the grid, then at once an EXIT whose team names one core | EXIT waited for the DELAY (one held command), then reached every core |
| `after-exit` | EXIT, then a CHECK record and a PING | the record is refused with reason "after EXIT"; the PING answers; no error |
| `malformed` | Four bad records (segment 240, flag bit 0, count 0, a team past the last core), then a good one | four `KM_EV_REFUSED` events with their reasons; the good record runs; the session goes on |
| `stop-busy` | A DELAY on every core, then at once `KM_SEG_STOP` | the kernel ends with `KMD_ERR_STOP_BUSY`; every core gave up on the stop word |
| `runtime-stop` | A DELAY of 0.5 s on every core; the host asks the X280 runtime to stop the kernel | the kernel ends with `KMD_ERR_STOPPED`; every core gave up on the stop word |
| `excl` | DELAY A on the first half, DELAY C (twice as long) on the second half but one core, then a `KM_F_EXCL` CHECK B on the two middle cores, which overlaps A only | B is held until A completes and does not wait for C: events in the order A, B, C |
| `verify` | The test image built with `KMD_VERIFY_LINES`, under load: `--n` commands, mostly NOP records whose header, team and argument words are functions of the command number, with a PING every 7th and a refused record (segment 245) every 11th | the kernel checked every record and found no word wrong; every event's kind, reason and ticks are what its command must produce |
| `bench` | `--n` NOP records at `--depth` to teams of `--team` cores (0 = all), then EXIT | every event and every core count is right; reports microseconds per command |
| `compile-error` | The worker is built with `DK_COMPILE_ERROR`, so it fails to compile | the exception is that build failure, and the X280 image was never started (or, with `--image-first`, was stopped) |
| `store` | No kernel: `--n` bursts of `--lines` 64-byte stores into LIM (or, with `--target l1`, a Tensix core's free L1) and one fence each, for every idle gap in `--gaps-us`, keep-busy period in `--busy-us` (a second thread storing one line per period; 0 = none) and number of burst positions in `--spread`; `--read` reads LIM through the runtime before each burst | always (it reports the time in the stores and in the fence of each burst, as percentiles and raw) |
| `memcost` | Runs `x280/memcost.c` alone (no grid): nanoseconds per load, store and fence on the stack (cluster local memory, uncached), in cached DRAM of this cluster's bank (through the Memory Port, in a DRAM buffer the test owns), and in registers | the image returned 0 |

The harness always launches the grid before it starts the image, so a compile error is
raised before any image exists. `--image-first` reverses the order, to test that the
error path stops a running image.

`bench` measures in two ways:

- **Kernel only**, with `--preload`: all `--n` + 1 commands are stored in the host ring
  before the kernel starts. `--n` + 1 must fit in `--host-depth` (at most 64).
- **Host-fed**: the host stores every command that fits in the ring, up to `--batch`
  commands, then fences once.
  `--settle-us` makes it wait that long after a fence before it stores the next batch.
  It exists because, on the host these tests were written on, a store into the card's
  window issued within about 0.5 microseconds after a fence often stalls for about 12
  microseconds; with `--settle-us 1` the stores cost about a fifth as much. It is off by
  default so that the plain numbers stay comparable with earlier runs.

`us_per_command_kernel` is the kernel's time from the first command it takes to the
last event, divided by the number of commands. `us_per_command_host` is the host's time
from the first store to the last event read, divided the same way.
`us_store_per_command` is the share of that host time spent in ring stores and fences.

## Build and run

You need:

- this repository installed (`pip install -e .`);
- a tt-metal source tree configured with `TT_METAL_BUILD_X280=ON` (which needs
  `TT_METAL_BUILD_TESTS=ON`) and built, for `build.py`;
- a Python environment with that tree's `ttnn`, and `torch` and `numpy`, which
  `dispatch_card.py` imports;
- one Blackhole p150a whose X280 local memory has been primed by the X280 runtime's
  preparation step: the harness opens the runtime with `RequirePrepared` and does not
  prime it.

```
python build.py --tree <tt-metal>
python dispatch_card.py ping
python dispatch_card.py teams --n 1000 --seed 1
python dispatch_card.py disjoint --n 8 --delay-us 2000
python dispatch_card.py stall --core 37 --deadline-us 50000
python dispatch_card.py fill --n 200
python dispatch_card.py verify --n 100000 --host-depth 64 --batch 64 --settle-us 1
python dispatch_card.py bench --preload --depth 16 --n 63 --host-depth 64
python dispatch_card.py bench --depth 16 --n 100000 --host-depth 64 --batch 64
```

- `--tree` defaults to `TT_METAL_HOME`.
- `--define NAME=VALUE` passes a define to the image build.
- `--image <path>` runs another image.
- `--host-depth` is 16 by default; the bench examples use 64, the most the kernel takes.

On a shared machine, `run.sh <tag> <dispatch_card.py args>` runs one case under a hang
watcher. The watcher is a tool of the host these tests were written on, not part of this
repository. It holds the card lock and kills the job on a time limit. Set:

- `KMR_CARD_ENV`: an environment script that sets `$PY`, the Python to run;
- `HANGWATCH`: the watcher's path;
- `LOCK_DIR`: the directory of its stop and reset lock files;
- optionally `CARD` and `LIMIT`.

Logs go to `<repo>/.logs/`, which git ignores.
