# Runtime harness

A tool for hunting defects in the Tensix runtime without a card. The headers in
`src/komorebi_native/include/komorebi/device/` and the kernels of
`examples/01_mapcollect` are compiled unchanged against `stub/`, a stand-in for the
tt-metal calls they use, and run as host threads on a simulated grid. The scenarios
force failure paths on that grid and check what the kernels do. The harness is run by
hand. Seven of its scenarios pin known behaviour in
`tests/core/test_runtime_harness.py`.

| file | what it holds |
|---|---|
| `stub/` | the tt-metal calls, each saying what it models and what it leaves out |
| `harness.hpp` | what a scenario sets and reads: grid, jobs, faults, the log |
| `sim_impl.hpp`, `sim.cpp` | the scheduler, L1 images, NoC delivery and the grid's run |
| `stub_calls.cpp` | the calls behind the stub headers |
| `dispatcher.cpp` | the dispatcher model and the grid's set-up |
| `checks.hpp`, `checks.cpp` | the checks every scenario ends with, and the runner |
| `pinned.cpp` | the seven scenarios the tests run |
| `scenarios.cpp`, `item_kernels.cpp` | the other scenarios, and a scenario's own compute kernels |
| `run.py` | builds the programs and runs the scenarios |

## Running it

    python experiments/runtime_harness/run.py                  # every scenario, one form
    python experiments/runtime_harness/run.py --rows --tsan    # also row leaders, TSAN
    python experiments/runtime_harness/run.py exit_partial_team
    python experiments/runtime_harness/run.py --tree <other checkout>

The program is built into `build/runtime_harness/` and reused while the sources are
unchanged. `--build-only --pinned` builds the tests' program instead: the seven pinned
scenarios, at `-O0`.

Each compute thread is compiled as its own object, as it is its own program on the
card. The build renames the runtime's namespace per thread (`-Dkm=km_unpack`, and so
on), and the compute stub's functions are `static`. Without that, the linker keeps one
thread's copy of an inline function for all three. At `-O0`, where little is inlined,
that made pack run unpack's buffer calls. Each scenario prints `ok` or `FAIL`, followed by any failed checks, kernels
that did not return, and broken stub rules. `--tree` builds another checkout's runtime
headers and example kernels against this harness.

## What it models

- A grid of 1 to 16 cores. Each core has its own L1 image and a thread for each of its
  five kernels (reader, writer, unpack, math, pack).
- A dispatcher that issues records the way the dispatch kernel does:
  - it holds a `KM_F_EXCL` record while an overlapping team is in flight;
  - it sweeps the acknowledgements;
  - it stops the grid on a missed deadline.
- NoC writes delivered in 16-byte chunks, each with its own delay, or dropped.
- Circular buffers with the hardware's wrap rules:
  - compute pointers wrap at or past the limit;
  - data-movement pointers wrap only at the limit.

  An item that passes a buffer's end is reported with the buffer, the side and the
  record.
- A clock that advances only at scheduling points. One thread runs at a time in a fixed
  order, so every run gives the same result.
- Faults: hold a kernel at a named call for a number of cycles, or set a core's stop
  word.

Every scenario ends with these checks:
- every kernel returned, with none of its writes still in flight;
- each acknowledged record was run once by each member of its team and completed
  through its tree;
- no write landed outside what a core receives;
- after a clean run, every kernel recorded leaving on the EXIT.

It cannot show:
- timing on the card;
- the real NoC's ordering;
- the L1 data cache;
- reads in flight: a read lands at the call;
- the vector unit: a tile is added in single precision.

With one thread running at a time, the thread sanitizer checks the harness, not races
between kernels.

## Scenarios

| scenario | what is forced |
|---|---|
| `smoke` | a few records on four cores, no fault |
| `stop_during_long_compute` | the grid is stopped while compute is deep in a long record |
| `writer_gives_up` | math is held: the writer's wait for the output gives up |
| `leader_gives_up` | a member's writer is held before its report: the leader gives up |
| `reader_gives_up` | unpack is held: the reader's wait for input room gives up |
| `token_wait_after_stop` | the stop word is set while the reader waits for a token page |
| `no_output_waits_for_compute` | pack held as it takes a record with no output |
| `wall_clock_across_a_carry` | a clock that moves at every register read, across the low word's carry |
| `result_word_from_the_leader` | `KM_F_RESULT` records, with and without a result given, and one without the flag |
| `slot_out_of_range` | a record's header names slot 20 |
| `slot_not_ring_slot` | a record in ring slot 0 whose header names slot 3 |
| `exit_partial_team` | an EXIT naming two of four cores, with late increments |
| `scan_across_wrap` | sequence numbers that run across the 32-bit wrap |
| `smallest_unless_part_landed` | the older record is still landing when a newer one has landed |
| `dropped_chunk_times_out` | a record chunk is lost: the deadline stops the grid |
| `idle_zero_means_no_limit` | a long gap between records with idle limit 0 |
| `idle_expiry_is_a_give_up` | a long gap with a nonzero idle limit |
| `mb_stop_between_ok_and_seq` | the stop word set between the two mailbox words of a token |
| `mb_math_held_while_unpack_aborts` | math held while unpack's token wait gives up |
| `mb_pack_held_while_input_wait_aborts` | pack held while unpack's input wait gives up |
| `mb_abort_then_good_token` | an input wait gives up, then a good token arrives |
| `mb_exit_during_input_wait` | the EXIT token arrives while unpack waits for input |
| `items_straddle_the_end` | 5 or 6 items of 5 pages plus one of 1 page per record, in 15 pages |
| `wrappers_refuse_straddling_items` | each bounded buffer wrapper asked for an item past the end |
| `excl_waits_for_overlapping_team` | `KM_F_EXCL` records on a disjoint and an overlapping team |
| `completion_every_team` | every team of an 8-core grid, rows of 3, random chunk delays |
| `random_jobs` | 10^4 records with random teams, sizes and delays on 6 cores |

The `mb_` scenarios run at mailbox depths 1 and 4.

## Defects it found

Each was found while the runtime was being written. The fix is named by its commit
subject on the main branch.

- When the grid was stopped, compute was stranded in its unbounded waits. Fixed by
  "Device: bounded compute waits for the token, input pages and output space".
- After a kernel gave up, the core went on reporting later records, which were
  acknowledged with wrong outputs. Fixed by "Device: a give-up stops the tile, and a
  tile that gave up acknowledges nothing".
- The reader's first wait returned on the stop word without handing an EXIT to the
  writer and compute. Fixed by "Device: the reader's first give-up hands the EXIT off as
  well".
- A record's header slot was trusted. A record naming slot 20 was run by both cores,
  and one increment landed in the start-up area. Fixed by "Device: a record runs only
  from its own slot; EXIT completes as every worker".
- An EXIT was completed by the team it named: the leader acknowledged after one of three
  increments. Fixed by the same commit.
- An idle limit of 0 made every core leave at once. Fixed by "Device: idle limit 0 means
  no limit, and it is the default".
- Pointer-to-`uint32_t` casts in the device headers did not compile on a 64-bit host.
- Replay of a card hang. Records of five 5-page items and one 1-page item in a 15-page
  buffer leave the second record's first item across the buffer end. The harness
  reports it at record 2. The wrappers refuse such an item since "Device: a buffer
  wrapper gives up on an item that would pass the buffer end".
- Replay of a card finding. A record with no output was acknowledged while compute
  was still in its body: the writer waited only for its own body. Fixed by "Device: a
  member completes only after its compute body returned".
- A card finding: a feeder deadline passed long before its time. The 64-bit wall clock
  was read through the high word that every core of a tile latches, so a read at the
  low word's carry could come out 2^32 cycles ahead. Fixed by "Device: the wall clock
  reads the live high word on both sides of the low word".

## Result

At the last change to this folder:
- all 27 scenarios pass in the one-level form;
- 3 scenarios pass in the row-leader form;
- all 27 pass under the thread sanitizer.

Times on the shared development host, from a cold build:
- one form: about 17 s, of which 13 s is `random_jobs`;
- all three forms: about 56 s;
- the tests' program: about 1.1 s to build.
