# Runtime harness

A tool for hunting defects in the Tensix runtime without a card. The headers in
`src/komorebi_native/include/komorebi/device/` and the kernels of
`examples/01_mapcollect` are compiled unchanged against `stub/`, a stand-in for the
tt-metal calls they use, and run as host threads on a simulated grid. The scenarios
force failure paths on that grid and check what the kernels do. The harness is run by
hand.

| file | what it holds |
|---|---|
| `stub/` | the tt-metal calls, each saying what it models and what it leaves out |
| `harness.hpp` | what a scenario sets and reads: grid, jobs, faults, the log |
| `sim_impl.hpp`, `sim.cpp` | the scheduler, L1 images, NoC delivery and the grid's run |
| `stub_calls.cpp` | the calls behind the stub headers |
| `dispatcher.cpp` | the dispatcher model and the grid's set-up |
| `checks.hpp`, `checks.cpp` | the checks every scenario ends with, and the runner |
| `pinned.cpp`, `pinned_main.cpp` | the pinned scenarios, run by hand (`--pinned`; no test runs them): seven here, four of the dynamic ones |
| `scenarios.cpp`, `item_kernels.cpp` | the other scenarios, and a scenario's own compute kernels |
| `messages_dispatch.cpp`, `messages_scenarios.cpp` | the submitter's two-pass end of a dynamic run, and the scenarios that run the generic dynamic kernels with the hop example |
| `run.py` | builds the programs and runs the scenarios |

`messages_scenarios.cpp` writes the forward example's arguments in the order its kernels
read them (`km_messages.hpp`'s `kMessagesCArg*` words, then the example's own): a word
added there, or a change in what one means, is made in `messages_config` too.

## Running it

    python experiments/checks/runtime_harness/run.py                  # every scenario, one form
    python experiments/checks/runtime_harness/run.py --rows --tsan    # also row leaders, TSAN
    python experiments/checks/runtime_harness/run.py exit_partial_group
    python experiments/checks/runtime_harness/run.py --tree <other checkout>

The program is built into `build/runtime_harness/` and reused while the sources are
unchanged. `--build-only --pinned` builds the pinned program instead: the eleven pinned
scenarios, at `-O0`, with the harness's own eight files compiled as two units (those on
`sim_impl.hpp` and the rest) so that each header family is parsed once: about 3.8 s of
compiler time instead of 7.4 s.

Each compute thread is compiled as its own object, as it is its own program on the
card. The build renames the runtime's namespace per thread (`-Dkm=km_unpack`, and so
on), and the compute stub's functions are `static`. Without that, the linker keeps one
thread's copy of an inline function for all three. At `-O0`, where little is inlined,
that made pack run unpack's buffer calls. A link's transport is built as NoC writes
(`KM_WIRE`): the stub has no fabric headers. Each scenario prints `ok` or `FAIL`, followed by any failed checks, kernels
that did not return, and broken stub rules. `--tree` builds another checkout's runtime
headers and example kernels against this harness.

## What it models

- A grid of 1 to 16 cores. Each core has its own L1 image and a thread for each of its
  five kernels (reader, writer, unpack, math, pack).
- A dispatcher that submits commands the way the dispatcher does:
  - it holds a `KM_FLAG_EXCLUSIVE` command while an overlapping group is in flight;
  - it sweeps the acknowledgements;
  - it stops the grid on a missed deadline, or, with `give_up`, copies the group's live
    words (an inspect), writes the slot's abort word on every core, and frees the slot
    once no core of the group runs the command (past `give_up_grace`: the stop).
- NoC writes delivered in 16-byte chunks, each with its own delay, or dropped.
- Circular buffers with the hardware's wrap rules:
  - compute pointers wrap at or past the limit;
  - data-movement pointers wrap only at the limit.

  An item that passes a buffer's end is reported with the buffer, the side and the
  command.
- A clock that advances only at scheduling points. One thread runs at a time in a fixed
  order, so every run gives the same result.
- Faults: hold a kernel at a named call for a number of cycles, or set a core's stop flag.

Every scenario ends with these checks:
- every kernel returned, with none of its writes still in flight;
- each acknowledged command was run once by each core of its group and completed
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
| `smoke` | a few commands on four cores, no fault |
| `stop_during_long_compute` | the grid is stopped while compute is deep in a long command |
| `writer_gives_up` | math is held: the writer's wait for the output gives up |
| `leader_gives_up` | a core's writer is held before its report: the leader gives up |
| `reader_gives_up` | unpack is held: the reader's wait for input room gives up |
| `token_wait_after_stop` | the stop flag is set while the reader waits for a token page |
| `no_output_waits_for_compute` | pack held as it takes a command with no output |
| `wall_clock_across_a_carry` | a clock that moves at every register read, across the low word's carry |
| `result_word_from_the_leader` | `KM_FLAG_RESULT` commands, with and without a result given, and one without the flag |
| `slot_out_of_range` | a command's header names slot 20 |
| `slot_not_ring_slot` | a command in ring slot 0 whose header names slot 3 |
| `exit_partial_group` | an EXIT naming two of four cores, with late increments |
| `scan_across_wrap` | sequence numbers that run across the 32-bit wrap |
| `smallest_unless_part_landed` | the older command is still landing when a newer one has landed |
| `dropped_chunk_times_out` | a command chunk is lost: the deadline stops the grid |
| `idle_zero_means_no_limit` | a long gap between commands with idle limit 0 |
| `idle_expiry_is_a_give_up` | a long gap with a nonzero idle limit |
| `mb_stop_between_ok_and_sequence` | the stop flag set between the two mailbox words of a token |
| `mb_math_held_while_unpack_aborts` | math held while unpack's token wait gives up |
| `mb_pack_held_while_input_wait_aborts` | pack held while unpack's input wait gives up |
| `mb_abort_then_good_token` | an input wait gives up, then a good token arrives |
| `mb_exit_during_input_wait` | the EXIT token arrives while unpack waits for input |
| `items_straddle_the_end` | 5 or 6 items of 5 pages plus one of 1 page per command, in 15 pages |
| `wrappers_reject_straddling_items` | each timed buffer wrapper asked for an item past the end |
| `excl_waits_for_overlapping_group` | `KM_FLAG_EXCLUSIVE` commands on a disjoint and an overlapping group |
| `completion_every_group` | every group of an 8-core grid, rows of 3, random chunk delays |
| `random_jobs` | 10^4 commands with random groups, sizes and delays on 6 cores |
| `give_up_stall_ends_the_run` | rank 1 waits for the stop flag only: it never leaves the command, and the deadline ends the run |
| `give_up_reader_port` | rank 1's reader waits for a port arrival that never comes: given up at the deadline, the next commands on its cores are right |
| `give_up_compute_input` | rank 1's compute waits for a second input tile that never comes: the same |
| `give_up_writer_output` | rank 1's writer waits for a second output tile that never comes: the same |
| `messages_smoke` | one message is forwarded once between two cores; the run ends quiet |
| `messages_random` | 60 messages of 12 forwards on 4 cores with two bins: bins evicted, packets filled and split; against the replay |
| `messages_idle_flapping` | an idle time of 50 cycles and chunk delays up to 3000: lines go idle and busy again many times, the end comes only after the last consume |
| `messages_one_core` | every forward of every message to rank 0, whose queue holds the whole population |
| `messages_quota` | one core sends until `send` returns false |
| `messages_gap` | the first packet commit to DRAM is lost: the reader reports the gap with core and position, the grid stops |
| `messages_overflow` | more messages than the queue holds: the overflow is reported |
| `messages_stall` | a core stalls in its body: the run stops and names it |

The `mb_` scenarios run at mailbox depths 1 and 4.

## Defects it found

Each was found while the runtime was being written. The fix is named by its commit
subject on the main branch.

- When the grid was stopped, compute was stranded in its waits without a timeout. Fixed by
  "Device: timed compute waits for the token, input pages and output space".
- After a kernel gave up, the core went on reporting later commands, which were
  acknowledged with wrong outputs. Fixed by "Device: a give-up stops the tile, and a
  tile that gave up acknowledges nothing".
- The reader's first wait returned on the stop flag without handing an EXIT to the
  writer and compute. Fixed by "Device: the reader's first give-up hands the EXIT off as
  well".
- A command's header slot was trusted. A command naming slot 20 was run by both cores,
  and one increment landed in the start-up area. Fixed by "Device: a command runs only
  from its own slot; EXIT completes as every worker".
- An EXIT was completed by the group it named: the leader acknowledged after one of three
  increments. Fixed by the same commit.
- An idle limit of 0 made every core leave at once. Fixed by "Device: idle limit 0 means
  no limit, and it is the default".
- Pointer-to-`uint32_t` casts in the device headers did not compile on a 64-bit host.
- Replay of a card hang. Commands of five 5-page items and one 1-page item in a 15-page
  buffer leave the second command's first item across the buffer end. The harness
  reports it at command 2. The wrappers reject such an item since "Device: a buffer
  wrapper gives up on an item that would pass the buffer end".
- Replay of a card finding. A command with no output was acknowledged while compute
  was still in its body: the writer waited only for its own body. Fixed by "Device: a
  core completes only after its compute body returned".
- A command given up while math waited for unpack with a limit: math's wait ended
  first and math left, and unpack, held meanwhile, then blocked for ever in its copy
  of the command. Math and pack now wait for
  unpack without a limit and show the wait in their live word; unpack's own waits are
  the timed ones. Fixed by "Runtime: inspect a running core, give up one command at
  its deadline".
- A card finding: a feeder deadline passed long before its time. The 64-bit wall clock
  was read through the high word that every core of a tile latches, so a read at the
  low word's carry could come out 2^32 cycles ahead. Fixed by "Device: the wall clock
  reads the live high word on both sides of the low word".

## Result

At the last change to this folder:
- all 39 scenarios pass in the one-level form;
- 3 scenarios pass in the row-leader form;
- all 39 pass under the thread sanitizer.

Times on the shared development host, from a cold build:
- one form: about 17 s, of which 13 s is `random_jobs`;
- all three forms: about 74 s;
- the tests' program: about 1.1 s to build.
