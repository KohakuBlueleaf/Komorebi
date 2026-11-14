# Dispatcher

The dispatcher runs on X280 cluster 0. Core 0 takes commands from its sources,
submits commands to the Tensix grid, sweeps the grid's acknowledgement words and reports
every finished command to its source as an event. Core 1 runs the stream engine, which
is one of those sources. `km_dispatch.h` holds the exact layouts;
`komorebi/abi/km_abi.h` holds the commands, the events and the handler numbers.

| file | part |
|---|---|
| `main.c` | start-up, the loop, taking a command, the finish; core 1's entry |
| `ring.h` | a source's command ring and event ring, the same code for every source |
| `event_ring.h` | the host buffer: the host's event ring, the state words, the ready block |
| `grid_port.h` | the windows onto every core's control block, the command store, the stop flag |
| `completion.h` | busy slots, acknowledgement sweep, deadlines, group overlap, group allocator |
| `queue.h` | the job queue: jobs from every source until a slot and a group are free |
| `engine_port.h` | core 0's side of the engine: admitting streams, its posts, stopping it |
| `work_port.h` | core 0's side of a work cluster's ring: its window onto the thread's events |
| `engine.h` | core 1: the engine's loop, the upload check, the interpreter (`km_stream.h`) |
| `graph_port.h` | core 0's side of the graphs: the graph table, admitting `RUN`, `CANCEL`, their ends |
| `graph.h` | core 1: up to two graphs on the plain-C executor (`../engines/km_graph_engine.c`) |
| `dynamic.h` | dynamic runs: the two passes over the status lines, the end command |
| `result.h` | the result block in LIM |
| `kernel.h` | the kernel's state, which lives in the cluster's DRAM bank |

## Sources

A source is anything that sends core 0 commands: the host (source 0), the stream
engine on core 1 (source 1) and the threads on work clusters 1 to 3 (sources 2 to 4). Each has a command ring and an event ring, served by the
same code (`ring.h`) under the protocol below; only their memory differs. The host's
command ring is in LIM and its event ring in host memory; the engine's two rings are in
the state area in the cluster's DRAM bank (`KMD_STATE_RING`, `KMD_ERING_*`), cached
and coherent between the cluster's cores. A work cluster's ring is split so that each
side reads only its own LIM: the thread writes its command lines and its read line into
this cluster's LIM after the layout (cluster c's at `work_cmds + (c - 1) *
KMD_WORK_RING_BYTES(work_depth)`), through a window of its own; core 0 writes the
consumed count, the event count and the events into the work cluster's LIM
(`work_noc_addr[c - 1]`, `KMD_WORK_*`) through window `KMD_WIN_WORK + c - 1`. Neither part passes
through a cache. Each pass of core 0's loop sweeps the busy slots, then takes at most
one command from each source in the order host, engine, one work ring; the engine is
looked at only while it has a stream, a command in flight or a waiting command, and work
ring c - 1 on the passes whose number is c - 1 modulo 4 (each ring every fourth pass, at
most one ring a pass).
A command's `KM_EVENT_DONE` or `KM_EVENT_TIMEOUT` goes to the source that sent it. Its auxiliary holds
in its low 8 bits the first core of the group that ran the command (DONE; for `KM_FLAG_QUEUE`,
the one the allocator chose) or the slot (TIMEOUT), and above them the low 16 bits of the
command's sequence number (`KM_EVENT_SEQUENCE_SHIFT`, `KM_EVENT_SEQUENCE_MASK`). The kernel numbers every
source's commands in one sequence; with these bits a source finds the number of each of its
commands while fewer than 32,768 commands of other sources are submitted between two of its
own; a queued command's number may fall up to 2,048 behind its source's order, as the
queue lets later jobs go first. A `KM_EVENT_TIMEOUT` names the command whose deadline passed first.

Nothing the kernel writes through the cached port stays dirty at exit: core 1 flushes its
part of the state area before it reports stopped and core 0 flushes the whole area before
the result (a core the runtime kills never gets there).

## Protocol with the host

These rules are part of the protocol; a host that relies on anything else is wrong.

- **Commands.** Command `i` (numbered from 1) is one 64-byte line in slot
  `i % host_depth` of the host ring. The host writes `i` into `sequence` and the three
  landing words; the kernel takes the line only when all four show `i`. The host may
  write command `i` once the consumed count in the host buffer has reached
  `i - host_depth`.
- **Ring depths** (`host_depth`, `event_depth`) are powers of two, so the slot numbering
  survives the wrap of the 32-bit counters. The kernel rejects other values at start.
- **Event room.** The host writes the number of events it has read into word 0 of the
  64-byte line at `KMD_LIM_READ` (the rest of the line is zero), as a whole line; the
  reference host stores it after each batch of commands, under the same fence. The
  kernel takes a command that ends in an event only while the unread events, one per
  busy slot holding a host command, one per running stream and one more fit the event
  ring, so it never overwrites an event the host
  has not read. Command `i` can wait for this only while
  `i - 1 - (count the host wrote)` reaches `event_depth`.
- **One store per event.** The kernel writes each 16-byte event with a single vector
  store. Separate stores from the X280 to host memory may be seen by the host in a
  different order than they were submitted, even with a store fence between them: an event
  written as two 8-byte halves, the index half last, was read with a new index over the
  old fields 2 times in 400,000 events, against 0 times in 400,000 with the single store
  (the harness's `verify` case under load, four runs of 100,000 commands each). The host
  reads `index`, the other fields, then `index` again, and keeps the event only when
  both reads give its number.
- **Rejected commands.** A bad command is consumed and answered with `KM_EVENT_REJECTED`, its
  reason (`KMD_REJECT_*`) in the status's auxiliary; the session goes on. Reasons: 1 a handler
  from 240 to 249, which this kernel does not serve (`KM_HANDLER_LOAD`, 240, and
  `KM_HANDLER_COPY`, 245, from the host or a work cluster, and `KMD_HANDLER_INSPECT`,
  243, from the host excepted, below; 241 and 242 from the host); 2 flag bit 0 set; 3 a
  group outside
  the grid (`count == 0 || first > cores || count > cores - first`, with no sum that
  could wrap; with `KM_FLAG_QUEUE` only the count is checked); 4 any command after EXIT; 5 a
  range of the upload region that is not whole lines inside it; 6 `KM_HANDLER_RUN` while
  four streams run; 7 `KM_HANDLER_CANCEL` of a req no running stream has; 8
  `KM_HANDLER_DEADLINE` of 0; 9 a `KM_FLAG_QUEUE` command while the queue is full (the host
  may retry it); 11 a work cluster's command that would wait; 12 a `KM_FLAG_QUEUE` command
  of the program's dynamic handler; 13 (`KMD_REJECT_PROGRAM`) a command, from any source,
  whose group has a core that does not hold its program (a queued one: unless every core
  holds it; never a COPY, which every program's loop serves), in a run with the program
  table (below), and a bad `KM_HANDLER_LOAD`. A
  rejected command's event has ticks 0.
- **The program table.** With `programs` 1 in the argument block (`KMD_ARGS_VERSION` 3),
  LIM at `KMD_LIM_PROGRAMS` holds a byte per core: the program field it holds (0xFF none;
  the layout grew by these 128 bytes). At the start the kernel
  makes it a pair of 64-bit masks per program field (`holds`); `take()` checks every
  command but NOP and EXIT against it (`program_held`: the group's bits, at most a few
  instructions; a run without the table, `programs` 0, skips the check). A
  `KM_HANDLER_LOAD` from the host or a work cluster's thread (a group switches program:
  its program field the new program, `a0` the program's record in DRAM) is exclusive:
  it goes only when the job queue is empty and no slot in flight shares a core with its
  group; the host's waits in its ring, never rejected for a busy group, a thread's is
  rejected busy (11) with the wake owed, as its other commands (`kmx_load` sends it
  again). It moves the group's bits to its program field when it is submitted (the
  cores take their commands in order, so a later command for the new program finds it
  held), whichever source sent it. A load with a program field of 64 or more, a queued
  one, or one in a run without the table is rejected with 13; from a stream or a graph
  with 1.
- **Ready block.** Before the state word turns READY, the kernel writes `kmd_ready_t`
  into 32-bit words 4 to 7 of the host buffer (`KMD_HOST_READY`) as one 16-byte store:
  word 4 the mark `KMD_READY_MARK` (0x594452), word 5 the upload region's LIM address,
  word 6 its size in bytes, word 7 `KMD_LIM_BYTES`. The host zeroes its buffer before
  the run and waits for the mark in word 4, not for the state word: the two are separate
  stores, so READY may be seen first. Once word 4 holds the mark, words 5 to 7 are valid.
- **Upload region.** It follows the layout (`KMD_LIM_UPLOAD`) and is `upload_bytes` of
  the argument block long: a multiple of 64 that ends inside LIM (the reference host
  gives it the rest of the runtime's available LIM, 707,840 bytes on card 1). The host
  writes it in whole 64-byte lines through its window and fences before the command
  that uses them; the kernel neither zeroes nor copies it.
- **Sequence numbers.** The host's command number is the event's `req`; the command the
  grid sees carries the kernel's own sequence number instead, in `sequence` and the landing
  words. It increases by one per command submitted, but skips 0 and the value the slot's
  acknowledgement word still holds from the slot's previous command, so a stale
  acknowledgement can never match. The host needs none of this; only a worker reads
  `sequence`.
- **Result word.** A command with `KM_FLAG_RESULT` is acknowledged by its group's leader with
  `{ack, result}` written as one 8-byte NoC write into the slot's acknowledgement line;
  the kernel reads the pair with one 8-byte load, keeps the result, and puts it in the
  `ticks` field of the command's `KM_EVENT_DONE` event. Such an event has no time; every
  other `KM_EVENT_DONE` event's ticks are the command's time from submit to acknowledgement.
- **EXIT** goes to every core whatever its group field says, and only once no slot is
  busy. A command with `KM_FLAG_EXCLUSIVE` waits until every job queued before it, by any
  source, has left the job queue and no busy slot's group overlaps its own: when the
  kernel takes the command it notes the queue's occupied places, and each job leaving
  the queue (submitted or dropped) clears its place. A command of the engine or a work
  cluster that was rejected busy counts from the take of its next attempt. The check
  holds only at its own submit, so a later command without the flag may still overlap
  it. A waiting host command holds up every host command behind it. This check
  is the only order of exclusive commands: the host stores them as soon as its ring has
  room, so back-to-back exclusive NOP commands on 120 cores come 2.3 us apart at 1000
  MHz (the host's reference, "Exclusive commands").
- **Job queue.** A command with `KM_FLAG_QUEUE` takes its count from the group word and
  ignores `first` (and `KM_FLAG_EXCLUSIVE`). It is taken into the job queue, which holds up to
  `KM_QUEUE_JOBS` (64) jobs of every source in arrival order, and answered with its
  `KM_EVENT_DONE` once it has run; a full queue rejects the host's (reason 9) and the
  engine's and a work cluster's (reason 11, below). The queue submits into the same
  completion slots as every command: at most the program's depth (64 at most) commands of
  all sources are in flight at once over the whole grid, queued ones included. Each
  pass ends with the queue's turn: of the first `KM_QUEUE_LOOK` (16) jobs
  as they are at its start, each that finds a free slot and room is submitted, the group
  allocator giving it the lowest `first` whose cores overlap no busy slot's group (the
  overlap test of `KM_FLAG_EXCLUSIVE`). A job that does not fit does not hold up the ones
  behind it, but every job submitted past a waiting one counts against it, and once a
  skipped job has been passed over `KM_QUEUE_PASSES` (1,024) times the turn ends before
  the next job: no more than 1,024 jobs ever overtake one. A job's deadline starts at
  its submit. The grid sees the chosen group and no `KM_FLAG_QUEUE`. EXIT waits for an empty
  queue; a finish drops what is still queued. Each queued job holds a place in its
  source's event room.
- **Streams.** `KM_HANDLER_RUN` (a0 offset, a1 bytes in the upload region) starts the
  stream there: a range that is not inside the region and line aligned, one that
  overlaps a running stream's, or a req a running stream has ends it at once with
  `KM_EVENT_STREAM_ERROR` reason 1 (core 0 checks these); otherwise core 0 hands it to the
  engine, which runs the same check as `stream.check` and ends a failing stream the same
  way. Up to four streams run on core 1, each at most `KM_STEP_OPS` micro-ops per engine
  pass in turn, and none waits inside a pass. An SUBMIT puts the command into the engine's
  command ring (after a `KM_HANDLER_DEADLINE` when the stream's deadline changed); core 0
  submits it in the engine's turn like a host command, and its event goes back to the
  engine, not to the host. The stream stays at the SUBMIT until core 0 has taken the
  command (in a later engine pass), so its commands reach core 0 in its own order. `KM_EVENT_STREAM`, `_END` and `_ERROR` carry the stream's req,
  the auxiliary `reason | pc << 8` for an error; the engine sends them to core 0 as
  `KMD_HANDLER_POST` commands, which core 0 forwards to the host. A ENQUEUE sends the command
  with `KM_FLAG_QUEUE` and the stream's tag in the group's `first` half (the kernel
  chooses `first`; such a command never goes to a work cluster); it waits while the
  queue's length (`KMD_ERING_QUEUED`) and the engine's jobs not yet taken in
  (`KMD_ERING_TAKEN`) fill the queue. At a stream's error or cancel the engine sends
  `KMD_HANDLER_DROP` (247) with the tag, and core 0 drops that stream's queued jobs, each
  answered to the engine with `KM_EVENT_REJECTED` reason 10. `KM_HANDLER_CANCEL` ends a
  stream with reason 10; its commands in flight still free their slots. A command of a
  stream that misses its deadline ends the stream with reason 8 and the kernel with it.
  EXIT waits until no stream runs.
- **Graphs.** `KM_HANDLER_RUN` with a2 = 1 (a0 offset, a1 bytes, a3 and a4 the binding
  block) or a2 = 2 (a0 the req of a kept graph) runs a graph on core 1 beside the
  streams (`docs/graphs.md`, "In the dispatcher"): up to two at once, eight kept.
  Core 0 rejects a2 above 2 with reason 1, a third graph with 6, a2 = 2 of a req no
  kept graph has with 7, and ends a bad range at once with `KM_EVENT_GRAPH_ERROR` reason
  1; core 1 checks the rest. `KM_HANDLER_CANCEL` of a running graph's req ends it with
  reason 10, of a kept one that is not running drops it (the same event). A graph's
  commands go back to the engine like a stream's; `KM_EVENT_GRAPH`, `_END` and `_ERROR`
  go to the host as `KMD_HANDLER_POST` commands, an end carrying in a3 whether the graph is
  kept. EXIT waits until no graph runs.
- **Commands from the engine.** The engine may send commands, `KM_HANDLER_DEADLINE`,
  `KMD_HANDLER_POST` (248) and `KMD_HANDLER_DROP` (247); STOP, EXIT, RUN and CANCEL from it are
  rejected with reason 1, and so are `KMD_HANDLER_POST` and `KMD_HANDLER_DROP` from the host.
  Core 0 never holds an engine command: a command that cannot be submitted now (no free
  slot, a busy overlapping group under `KM_FLAG_EXCLUSIVE`, a full queue) is consumed and
  answered `KM_EVENT_REJECTED` reason 11 (`KMD_REJECT_BUSY`) at once, the rejection stored
  before the consumed count. After a busy rejection core 0 owes the engine one
  `KMD_EVENT_WAKE` (0x12), which it pushes when a slot next frees (any source's command) or
  the queue submits a job; its room in the engine's event ring is kept from the rejection
  on. Core 1 does the waiting: a rejected stream sends the same SUBMIT or ENQUEUE again
  after the next wake, the other streams go on meanwhile.
- **Commands from a work cluster.** The argument block's `work_*` words name the rings
  (`work_cmds` 0: none; ring c - 1 in use when `work_noc_addr[c - 1]` is nonzero; one
  depth and one event depth for all, `KMD_ARGS_VERSION` 2); the host zeroes its command lines before the start and the
  thread's event area before the thread starts. The thread may send commands,
  `KM_HANDLER_DEADLINE` and `KM_HANDLER_PING`; STOP, EXIT, RUN, CANCEL and the engine's commands
  from it are rejected with reason 1 (the run belongs to the host, which stops the
  thread before the kernel). A command of the thread that cannot be submitted now (a busy
  overlapping group under `KM_FLAG_EXCLUSIVE`, a full queue, no free slot) is consumed and
  rejected with reason 11 at once, with the same `KMD_EVENT_WAKE` (handle 0) owed as to the
  engine, so a thread never holds core 0 (`kmx_wait` passes wakes over, `kmx_poll`
  returns them); a thread that stops
  reading its events stops having its commands taken once its event ring is full, and
  nothing else waits for it. Looking at the ring every fourth pass costs core 0's idle
  pass 59 to 73 ns at 1000 MHz (32 to 41 ns at the 1750 opt-in; the experiment
  `x280_threads/ring_cost.py`); a STEP and SUM round of `examples/07_work_cluster`, two
  commands submitted and waited for in turn, takes about 53 microseconds from the thread
  (50 at 1750).
- **What streams cost the sweep.** Measured on a p150a at 1000 MHz (pass period of
  core 0 = run time / passes; the 1750 opt-in's figures in brackets): 60 ns with no
  stream [32], 86 ns with one or four streams that only loop [49], 76 ns with one
  stream submitting one command at a time [89], 279 ns with four [159]. One submitting stream
  takes 3.5 microseconds per command [2.3], 2.4 microseconds (median) from a command's
  free to the stream's next submit over the ring [1.34]; with three looping streams
  beside it, 23.9 [12.8]. A host command at depth 16 to one core costs 1.24
  microseconds host-fed [0.641], 1.26 with all 64 commands preloaded [0.669]. The
  commands are in `experiments/dispatcher/README.md`.
- **Dynamic runs.** The argument block's `messages_handler` names the program's dynamic
  handler (0: none). A command of it is a dynamic run's `RUN`; one runs at a time. A
  second one from the host waits until the first and its end command have completed;
  from the engine or a work cluster it is rejected busy (reason 11) with the wake. Each
  core of the run writes its 16-byte status line `{run, state, sent, consumed}` into
  LIM at `KMD_LIM_MESSAGES` + 16 x core (the layout grew by these 2 KiB, to 0x2000). While the run is in flight, every pass of core 0 reads the group's lines;
  once two passes, the second after the first, find every line idle and unchanged and
  the sums of sent and consumed equal, core 0 submits the end command
  (`KM_HANDLER_MESSAGES_END`, a0 the run's sequence) to the group in a free slot under the
  `RUN`'s command and source. The end command's completion has no event: the source
  sees only the `RUN`'s `KM_EVENT_DONE`, which comes first. A line that says gave up sets
  the `RUN`'s deadline to now, so the session ends with `KMD_ERROR_DEADLINE` naming it.
  Core block words `KMD_H_MESSAGES_GAVE` (1 + the core that gave up), `KMD_H_MESSAGES_CHECKS`
  (two-pass checks) and `KMD_H_MESSAGES_QUIET` (mtime of the last quiet end, low word)
  report it; the host's `finish` returns the core block.
- **What a command costs a stream.** The harness's `engine-cost` case
  (`dispatch_card.py engine-cost --delay-microseconds 5 --n 1000`), 1,000 commands on one core
  timed by the stream itself (`TIME`), p150a, depth 4:

  | Stream | Microseconds per command, 1000 MHz | 1750 MHz (opt-in) |
  |---|---|---|
  | `SUBMIT`, then `WAIT 0`, a command of 5 us | 8.34 | 7.16 |
  | `SUBMIT` with `KM_FLAG_EXCLUSIVE` and no wait, a command of 5 us: rejected busy while the one before runs (999 rejections), sent again after the wake | 8.07 | 7.01 |
  | `ENQUEUE` of 1-core jobs of `DELAY 0`, then one `WAIT 0` | 2.58 | 1.46 |

  A busy rejection and its wake cost no more than waiting for the record's event: both
  add about 3.3 microseconds to the command's run at 1000 MHz (2.1 at 1750).
- **Finish.** `KM_HANDLER_STOP`, the runtime's stop, a missed deadline or a start-up failure
  ends the loop. Core 0 then sets the engine's stop flag and waits for core 1 to leave
  its loop (`KMD_ENGINE_STOP_TICKS`, 10 ms), forwarding its last posts; if it does not,
  the kernel ends with `KMD_ERROR_ENGINE`. Core 1 returns before core 0 does. The stop flag goes to the grid only while a slot is busy or before EXIT was submitted. The state
  and error words in the host buffer, the result block and the core block in LIM
  (`KMD_LIM_CORES`: each core's progress marker, the engine's state and counts) then
  hold the outcome. `KMD_H_ABANDONED` counts the commands in flight (busy slots 0 to 31 in
  `KMD_R_BUSY`, 32 to 63 in the core block's `KMD_H_BUSY_HIGH`) and
  the jobs still queued at the finish: none of them gets its `KM_EVENT_DONE`.
- **Inspect.** `KMD_HANDLER_INSPECT` (243) from the host, group `[first, first +
  count)`, reads each core's state words, the first `KM_LIVE_READ` bytes of its live
  words and, when the live words name a trace in L1, the trace entries of the four
  latest sequences, all from the control block, by aligned 64-bit loads through window
  `KMD_WIN_INSPECT` (22, aimed at each core in turn at the coordinates of the host's
  core table, `KMD_LIM_COORDS`). Every address is checked to lie in L1 above the
  firmware's part, and nothing else is read. One core is read per pass of the loop, so
  commands in flight go on meanwhile. The records (`KMD_INSPECT_RECORD`, 832 bytes
  each: a line `KMD_IR_*`, then the words) follow the block's header at
  `KMD_LIM_INSPECT` (`KMD_I_*`). After the last record the kernel writes the read's
  ticks, then the mark `KMD_INSPECT_MARK` last, and answers `KM_EVENT_PONG`. The
  command takes no slot and no sequence number. A second inspect waits in the ring
  while one runs. The host zeroes the layout up to the block's header
  (`KMD_LIM_ZEROED`) and writes the core table before the run. `KMD_LIM_BYTES` is
  0x1C2C0.
- **Giving up one command.** At its deadline, a host command of a program's handler
  (not a dynamic run's) is not fatal. The kernel first reads its group into the same
  block (kind `KMD_INSPECT_TIMEOUT`, with the slot and sequence). Then it writes the
  command's sequence into the slot's abort word (`KM_CONTROL_ABORT` + 4 x slot) on
  every core through the panel windows. Then, one core a pass, it reads the live
  sequence words (`KM_LIVE_SEQUENCE`, one per kernel) until no kernel of the group runs
  the command; then it frees the slot, counts the give-up in the core block's
  `KMD_H_GIVEN_UP` and sends `KM_EVENT_TIMEOUT`, in that order. The host tells this
  timeout from a fatal one by the count. The cores leave the command, clear its pages
  and take the next one (the runtime reference, "Giving up one command"). A core still
  in the command after `KMD_GIVE_UP_TICKS` (20 ms) marks the slot kept
  (`KMD_HANDLER_KEPT`), and the next deadline check ends the run with
  `KMD_ERROR_DEADLINE`. A command that completes while it is read or checked ends the
  inspect instead. A late acknowledgement of the slot's old sequence matches no
  command. Any other command that misses its deadline ends the run as before.
- **Abort.** The runtime's stop is the abort: core 0 looks for it every 1,024 passes
  (tens of microseconds), the poll that also checks deadlines, and finishes as
  above whatever is in flight. Only a core that has stopped running its loop (an
  access that never completes) does not answer it; the runtime then cannot restore
  the cluster and the card needs a reset.
- **Trap.** `main` (`trap.h`) points `mtvec` at the kernel's own handler for the run
  and restores the runtime's on return. An exception on core 0 or 1 ends in
  `kmd_trapped`, which does not trust the state in DRAM (its pointers may be what
  failed): core 0 rebuilds the windows from the argument block and the hello's
  control block address in LIM, stores the stop flag (unless EXIT left the grid),
  stops core 1, writes `KMD_ERROR_TRAP` (core, `mcause`, `mepc`, `mtval`) into the result
  block and the host buffer, flushes the state's lines and returns from `main`, so the
  runtime restores the cluster as after an abort. Core 1 notes its trap in the core
  block (`KMD_ENGINE_TRAPPED`); core 0 reports it at its finish. Interrupts, and a
  second exception inside the handler, go to the runtime's handler, which ceases the
  core as before.
- **Clock.** The session runs the X280 at 1000 MHz; 1750 is an explicit opt-in that
  loses loads when the Tensix grid is under full matrix load. At 1750 MHz, with every
  Tensix core at full matrix load (the card lowers its core voltage under it), core 0's
  loads returned wrong values (a pointer of the host ring read as 0, its base off by 12
  bytes) and the kernel trapped within seconds: 9 of 18 runs of the 8192-cubed matrix
  product, against 0 of 23 at 1000 MHz (the bisect below). One PLL clocks all four
  clusters, so thread programs on the work clusters run at the same clock. Before a
  session sets it, the runtime that opens the cluster leaves that PLL at its safe
  200 MHz; 1750 was never the card's own setting, only the session's earlier default.
  The session's log names the clock it set (`X280 clock set: 1000 MHz`).

  | Variant of the 8192-cubed run (a profile command, then 300 calls) | 1750 MHz | 1000 MHz |
  |---|---|---|
  | as the program runs it | 4 of 7 | 0 of 23 (3 with 700 calls) |
  | a 64 KiB gap per bank before the kernel's state | 1 of 3 | |
  | the state flushed to DRAM right after boot | 2 of 4 | |
  | the profile command's pages not read by the host | 2 of 4 | |
  | no profile command | 0 of 5 | |
