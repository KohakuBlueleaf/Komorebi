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
| Driver, one case or the regression set per run | `dispatch_card.py` |
| The graph executor's cases on the same harness (`SCRIPT=graph_card.py` for `run.sh`) | `graph_card.py` |
| Builds the kernel image, the test image (`KMD_VERIFY_LINES` and `KMD_TRACE`), the measuring image of `experiments/dispatch_kernel` and the host module | `build.py` |
| One run under a hang watcher | `run.sh` |

## Cases

The kernel keeps its state in this cluster's DRAM bank (cached); every case that runs
the kernel gives it `KMD_STATE_BYTES` (32 KiB) of a DRAM buffer the test owns.

The kernel takes a command only while its event has room in the event ring
(`--event-depth`, 128 by default, a power of two like `--host-depth`). For that the host
module stores the number of events it has read into the kernel's local memory with every
batch of commands, and while it waits whenever the kernel could be holding a command for
it (command i can wait only while i minus that number exceeds the ring). The kernel's
idle limit is off unless `--idle-s` sets one.

`python dispatch_card.py <case> [options]` prints one JSON line. `ok` is true when the
case's checks hold. `python dispatch_card.py set` runs the regression set, the cases
below marked "set", each on a small input, in one device open: a JSON line per case and
one for the set, whose `ok` is true when every case's is. It takes about 13 seconds.

| case | what it does | `ok` when |
|---|---|---|
| `ping` (set) | Sends one `KM_SEG_PING`, then EXIT to the whole grid, then `KM_SEG_STOP` | the PONG carries the PING's command number; every core finished |
| `issue` | Like `ping`, plus one CHECK record to the whole grid | the event carries the record's command number; no argument errors |
| `teams` (set) | `--n` CHECK records with random teams at depth 16, fed in batches | every event arrives once; every core's run count, key sum and key mix match |
| `disjoint` (set) | A DELAY of `--delay-us` on the first half of the grid, then `--n` CHECK records on the second half | the second team's events all arrive before the first's; two slots were busy at once; every core's counts match |
| `stall` (set) | `KM_SEG_DEADLINE` with `--deadline-us`, then a record whose member `--core` waits for the stop word | `KM_EV_TIMEOUT` for that slot and command; every core gave up on the stop word; only `--core` did not report the record |
| `stall-late` (set) | Depth 4: on the last core a record that waits for the stop word under a deadline ten times `--deadline-us`, then on the other cores a record whose member `--core` waits for the stop word and two records behind it | `KM_EV_TIMEOUT` names the stalled record (its deadline passed first) and its slot 1, not the lower slot 0 of the record issued before it with the later deadline; all four slots were busy at the finish |
| `stall-leader` | Like `stall`, with the team's leader (core 0) as the core that waits | the same, with core 0 the one that did not report |
| `fill` (set) | 16 DELAY records of `--delay-us` on 16 disjoint teams, then `--n` CHECK records with random teams | all 16 slots were busy at once; every event and every core count is right |
| `exit-busy` (set) | A DELAY on half the grid, then at once an EXIT whose team names one core | EXIT waited for the DELAY (one held command), then reached every core |
| `after-exit` (set) | EXIT, then a CHECK record and a PING | the record is refused with reason "after EXIT"; the PING answers; no error |
| `malformed` (set) | Four bad records (segment 240, flag bit 0, count 0, a team past the last core), then a good one | four `KM_EV_REFUSED` events with their reasons; the good record runs; the session goes on |
| `stop-busy` (set) | A DELAY on every core, then at once `KM_SEG_STOP` | the kernel ends with `KMD_ERR_STOP_BUSY` and counts the DELAY abandoned; every core gave up on the stop word |
| `runtime-stop` | A DELAY of 0.5 s on every core; the host asks the X280 runtime to stop the kernel | the kernel ends with `KMD_ERR_STOPPED` and counts the DELAY abandoned; every core gave up on the stop word |
| `excl` (set) | DELAY A on the first half, DELAY C (twice as long) on the second half but one core, then a `KM_F_EXCL` CHECK B on the two middle cores, which overlaps A only | B is held until A completes and does not wait for C: events in the order A, B, C |
| `queue` (set) | The test image: `--n` DELAY records of up to `--delay-us`, 70% with `KM_F_QUEUE` (counts from 1 to the whole grid, `first` random), the rest with a given team of 1 to 16 cores, half of them with `KM_F_EXCL`; first two queued records with count 0 and one above the grid | the two are refused with reason 3, and any refused for a full queue (reason 9) did not run; replaying the kernel's trace, every queued record got the lowest team free at its issue and its `KM_EV_DONE` names that team's first core, no `KM_F_EXCL` record overlapped a team in flight, more than one queued record was in flight at once; every core ran exactly the traced teams |
| `verify` (set) | The test image built with `KMD_VERIFY_LINES`, under load: `--n` commands, mostly NOP records whose header, team and argument words are functions of the command number, with a PING every 7th and a refused record (segment 245) every 11th | the kernel checked every record and found no word wrong; every event's kind, reason and ticks are what its command must produce |
| `result` (set) | `--n` records of the worker's result segment with `KM_F_RESULT`, teams of `--team` cores (1 if 0) rotating over the grid, streamed a million at a time | every `KM_EV_DONE` event carries the result of its own command (`dk_result` of its key), the kernel counted `--n` results, every core ran exactly its records |
| `upload` (set) | The `KMD_VERIFY_LINES` image: the whole upload region written twice through the host window with a pattern of whole lines, then `KMD_SEG_CHECK_UPLOAD` over the region, single lines, a middle range, a wrong seed, one changed word, and two ranges outside the region | the ready block names the region after the layout; every check finds exactly the differing words (none, 16, or the one changed word at its index); the two outside ranges are refused with reason 5 |
| `streams` (set) | The test image: `KM_SEG_RUN` and `KM_SEG_CANCEL` refusals and range errors with four looping streams at once, every reference stream of `experiments/stream_checks/streams` that issues no record, then a stream of `--n` result records one at a time | every refusal and error is the expected one; every reference's events equal the simulator's; the records reach the grid in the simulated order with the simulated teams; a PING answers after each part |
| `replay` (set) | The test image: every reference stream that issues records and does not rest on the simulator's passes (a deadline or cancel in passes, a record never acknowledged), its segments set to `DK_SEG_RESULT`, each simulated on this grid under several completion models | where all models agree, the stream's events and its teams in issue order equal the simulator's; otherwise its record counts and its last event do; every queued record got the lowest team free at its issue; every core ran exactly the traced teams; a PING answers after each |
| `stream-deadline` (set) | A stream whose second record has a member waiting for the stop word, header deadline `--deadline-us` | one `KM_EV_STREAM_ERROR` with reason 8 at that `ISSUE`; the kernel ends with the deadline error; every core gave up on the stop word |
| `engine-busy` (set) | A host DELAY of `--delay-us` on the grid's first half, then a stream with one `KM_F_EXCL` record on that half and a stream of `--n` records one at a time on the other half | the second stream ends before the DELAY's `KM_EV_DONE` and the first after it, both with their halt values: the blocked record was refused busy and sent again (the engine sent more records than the two streams have), and did not hold up the other stream |
| `seq-events` (set) | A host record, a stream of `--n` records one at a time, a second host record | the host records' `KM_EV_DONE` events carry their first cores and the low 16 bits of sequence numbers 1 and n + 2: the stream's records are counted in the same sequence and the host can tell |
| `engine-cost` | Three streams in turn, each `--n` records on core 0 timed by the stream: ISSUE and WAIT 0 of a `--delay-us` DELAY; the same ISSUE with `KM_F_EXCL` and no wait, refused busy while the one before runs; SUBMIT of DELAY 0 jobs | a measurement: microseconds per record of each stream and the kernel's count of refusals; ok when every stream ended and the kernel finished clean |
| `exit-flush` (set) | A stream of `--n` records one at a time and a looping stream on core 1 when STOP ends the kernel; the host then writes a pattern over the kernel's state buffer in DRAM, and a second kernel answers a PING and exits | the first stream ended, the kernel ended without error and stopped the engine, and the pattern reads back with no word changed: neither core left a line of the state dirty in the cache |
| `compile-error` | The worker is built with `DK_COMPILE_ERROR`, so it fails to compile | the exception is that build failure, and the X280 image was never started (or, with `--image-first`, was stopped) |

`python graph_card.py <case>` runs the graph executor's cases; `set` runs both in one
device open (about 5 seconds). The worker's `DK_SEG_SUM` answers `a0 + count`, the
result word the reference graphs assume.

| case | what it does | `ok` when |
|---|---|---|
| `graphs` | The test image: two graphs of a long DELAY running, then a third `RUN`, `a2 = 3`, a kept `RUN` of an unknown req, both cancelled; a graph over a running stream's range, an unaligned one; seven kept graphs, `kept_graph`, its drop, an eighth, `kept_table_full`, every drop; every saved reference of `experiments/graph_checks/graphs` that ends by itself, its templates on `DK_SEG_SUM`; the documented loop kept and run again by its req with another binding | every refusal, error and drop is the expected one; every graph's events (rounds, end with registers 0 and 1, error with reason and node) equal the simulator's on this grid; the engine's records in the trace are each graph's simulated teams (a queued record's count); every core ran exactly those; a PING answers after each part |
| `graph-deadline` | A graph whose second record has member `--core` waiting for the stop word, header deadline `--deadline-us` | one `KM_EV_GRAPH_ERROR` with reason 8 at that node; the kernel ends with the deadline error; every core gave up on the stop word |

The harness always launches the grid before it starts the image, so a compile error is
raised before any image exists. `--image-first` reverses the order, to test that the
error path stops a running image.

Throughput, host store cost, the memory cost per operation and the stream engine's pass
are measured by `experiments/dispatch_kernel/measure.py` on this harness.

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
python dispatch_card.py set
python dispatch_card.py ping
python dispatch_card.py teams --n 1000 --seed 1
python dispatch_card.py stall --core 37 --deadline-us 50000
python dispatch_card.py verify --n 100000 --host-depth 64 --batch 64 --settle-us 1
```

- `--tree` defaults to `TT_METAL_HOME`.
- `--define NAME=VALUE` passes a define to the image build.
- `--image <path>` runs another image.
- `--host-depth` is 16 by default; 64 is the most the kernel takes.

On a shared machine, `run.sh <tag> <dispatch_card.py args>` runs one case under a hang
watcher. The watcher is a tool of the host these tests were written on, not part of this
repository. It holds the card lock and kills the job on a time limit, or sooner when the
log gains no line for a third of it: the harness prints a line before and after every
launch and readback. Set:

- `KMR_CARD_ENV`: an environment script that sets `$PY`, the Python to run;
- `HANGWATCH`: the watcher's path;
- `LOCK_DIR`: the directory of its stop and reset lock files;
- optionally `CARD`, `LIMIT` (seconds, default 120), `STALL` (at most a third of
  `LIMIT`) and `OP_TIMEOUT` (`TT_METAL_OPERATION_TIMEOUT_SECONDS`, default 30, so a
  blocked host call raises before the watcher acts).

The stand-in worker bounds every wait, NoC barriers included, by `--limit`, and checks
that every address the host hands it is 16-byte aligned.

Logs go to `<repo>/.logs/`, which git ignores.
