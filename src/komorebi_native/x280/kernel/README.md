# Dispatch kernel

The dispatch kernel runs on X280 cluster 0. Core 0 takes commands from its sources,
issues records to the Tensix grid, sweeps the grid's acknowledgement words and reports
every finished command to its source as an event. Core 1 runs the stream engine, which
is one of those sources. `km_dispatch.h` holds the exact layouts;
`komorebi/abi/km_abi.h` holds the records, the events and the segment numbers.

| file | part |
|---|---|
| `main.c` | start-up, the loop, taking a command, the finish; core 1's entry |
| `ring.h` | a source's command ring and event ring, the same code for every source |
| `event_ring.h` | the host buffer: the host's event ring, the state words, the ready block |
| `grid_port.h` | the windows onto every core's control block, the record store, the stop word |
| `completion.h` | busy slots, acknowledgement sweep, deadlines, team overlap, team allocator |
| `queue.h` | the job queue: jobs from every source until a slot and a team are free |
| `engine_port.h` | core 0's side of the engine: admitting streams, its posts, stopping it |
| `work_port.h` | core 0's side of a work cluster's ring: its window, the refusal of a record that would wait |
| `engine.h` | core 1: the engine's loop, the upload check, the interpreter (`km_stream.h`) |
| `graph_port.h` | core 0's side of the graphs: the graph table, admitting `RUN`, `CANCEL`, their ends |
| `graph.h` | core 1: up to two graphs on the plain-C executor (`../engines/km_graph_engine.c`) |
| `result.h` | the result block in LIM |
| `kernel.h` | the kernel's state, which lives in the cluster's DRAM bank |

## Sources

A source is anything that sends core 0 commands: the host (source 0), the stream
engine on core 1 (source 1) and the thread on a work cluster (source 2). Each has a command ring and an event ring, served by the
same code (`ring.h`) under the protocol below; only their memory differs. The host's
command ring is in LIM and its event ring in host memory; the engine's two rings are in
the state area in the cluster's DRAM bank (`KMD_STATE_RING`, `KMD_ERING_*`), cached
and coherent between the cluster's cores. A work cluster's ring is split so that each
side reads only its own LIM: the thread writes its command lines and its read line into
this cluster's LIM after the layout (`work_cmds`), through a window of its own; core 0
writes the consumed count, the event count and the events into the work cluster's LIM
(`work_noc_addr`, `KMD_WORK_*`) through window `KMD_WIN_WORK`. Neither part passes
through a cache. Each pass of core 0's loop sweeps the busy slots, then takes at most
one command from each source in the order host, engine, work cluster; the engine is
looked at only while it has a stream, a record in flight or a waiting command, the work
cluster's ring every fourth pass.
A record's `KM_EV_DONE` or `KM_EV_TIMEOUT` goes to the source that sent it. The aux of
`KM_EV_DONE` is the first core of the team that ran the record (for `KM_F_QUEUE`, the one
the allocator chose); a `KM_EV_TIMEOUT` names the record whose deadline passed first.

Nothing the kernel writes through the cached port stays dirty at exit: core 1 flushes its
part of the state area before it reports stopped and core 0 flushes the whole area before
the result (a core the runtime kills never gets there).

## Protocol with the host

These rules are part of the protocol; a host that relies on anything else is wrong.

- **Commands.** Command `i` (numbered from 1) is one 64-byte line in slot
  `i % host_depth` of the host ring. The host writes `i` into `seq` and the three
  landing words; the kernel takes the line only when all four show `i`. The host may
  write command `i` once the consumed count in the host buffer has reached
  `i - host_depth`.
- **Ring depths** (`host_depth`, `event_depth`) are powers of two, so the slot numbering
  survives the wrap of the 32-bit counters. The kernel refuses other values at start.
- **Event room.** The host writes the number of events it has read into word 0 of the
  64-byte line at `KMD_LIM_READ` (the rest of the line is zero), as a whole line; the
  reference host stores it after each batch of commands, under the same fence. The
  kernel takes a command that ends in an event only while the unread events, one per
  busy slot holding a host record, one per running stream and one more fit the event
  ring, so it never overwrites an event the host
  has not read. Command `i` can wait for this only while
  `i - 1 - (count the host wrote)` reaches `event_depth`.
- **One store per event.** The kernel writes each 16-byte event with a single vector
  store. Separate stores from the X280 to host memory may be seen by the host in a
  different order than they were issued, even with a store fence between them: an event
  written as two 8-byte halves, the index half last, was read with a new index over the
  old fields 2 times in 400,000 events, against 0 times in 400,000 with the single store
  (the harness's `verify` case under load, four runs of 100,000 commands each). The host
  reads `index`, the other fields, then `index` again, and keeps the event only when
  both reads give its number.
- **Refused commands.** A bad record is consumed and answered with `KM_EV_REFUSED`, its
  reason (`KMD_REFUSE_*`) in the status's aux; the session goes on. Reasons: 1 a segment
  from 240 to 249, which this kernel does not serve; 2 flag bit 0 set; 3 a team outside
  the grid (`count == 0 || first > cores || count > cores - first`, with no sum that
  could wrap; with `KM_F_QUEUE` only the count is checked); 4 any record after EXIT; 5 a
  range of the upload region that is not whole lines inside it; 6 `KM_SEG_RUN` while
  four streams run; 7 `KM_SEG_CANCEL` of a req no running stream has; 8
  `KM_SEG_DEADLINE` of 0; 9 a `KM_F_QUEUE` record while the queue is full (the host
  may retry it); 11 a work cluster's record that would wait. A refused command's event
  has ticks 0.
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
- **Sequence numbers.** The host's command number is the event's `req`; the record the
  grid sees carries the kernel's own sequence number instead, in `seq` and the landing
  words. It increases by one per record issued, but skips 0 and the value the slot's
  acknowledgement word still holds from the slot's previous record, so a stale
  acknowledgement can never match. The host needs none of this; only a worker reads
  `seq`.
- **Result word.** A record with `KM_F_RESULT` is acknowledged by its team's leader with
  `{ack, result}` written as one 8-byte NoC write into the slot's acknowledgement line;
  the kernel reads the pair with one 8-byte load, keeps the result, and puts it in the
  `ticks` field of the record's `KM_EV_DONE` event. Such an event has no time; every
  other `KM_EV_DONE` event's ticks are the record's time from issue to acknowledgement.
- **EXIT** goes to every core whatever its team field says, and only once no slot is
  busy. A record with `KM_F_EXCL` waits until no busy slot's team overlaps its own; the
  check holds only at its own issue, so a later record without the flag may still
  overlap it. A waiting host command holds up every host command behind it.
- **Job queue.** A record with `KM_F_QUEUE` takes its count from the team word and
  ignores `first` (and `KM_F_EXCL`). It is taken into the job queue, which holds up to
  `KM_QUEUE_JOBS` (64) jobs of every source in arrival order, and answered with its
  `KM_EV_DONE` once it has run; a full queue refuses the host's (reason 9) and the
  engine's (reason 11, below). Each pass ends with the queue's turn: of the first `KM_QUEUE_LOOK` (16) jobs
  as they are at its start, each that finds a free slot and room is issued, the team
  allocator giving it the lowest `first` whose cores overlap no busy slot's team (the
  overlap test of `KM_F_EXCL`). A job that does not fit does not hold up the ones
  behind it, but every job issued past a waiting one counts against it, and once a
  skipped job has been passed over `KM_QUEUE_PASSES` (1,024) times the turn ends before
  the next job: no more than 1,024 jobs ever overtake one. A job's deadline starts at
  its issue. The grid sees the chosen team and no `KM_F_QUEUE`. EXIT waits for an empty
  queue; a finish drops what is still queued. Each queued job holds a place in its
  source's event room.
- **Streams.** `KM_SEG_RUN` (a0 offset, a1 bytes in the upload region) starts the
  stream there: a range that is not inside the region and line aligned, one that
  overlaps a running stream's, or a req a running stream has ends it at once with
  `KM_EV_STREAM_ERROR` reason 1 (core 0 checks these); otherwise core 0 hands it to the
  engine, which runs the same check as `stream.check` and ends a failing stream the same
  way. Up to four streams run on core 1, each at most `KM_STEP_OPS` micro-ops per engine
  pass in turn, and none waits inside a pass. An ISSUE puts the record into the engine's
  command ring (after a `KM_SEG_DEADLINE` when the stream's deadline changed); core 0
  issues it in the engine's turn like a host record, and its event goes back to the
  engine, not to the host. The stream stays at the ISSUE until core 0 has taken the
  record (in a later engine pass), so its records reach core 0 in its own order. `KM_EV_STREAM`, `_END` and `_ERROR` carry the stream's req,
  the aux `reason | pc << 8` for an error; the engine sends them to core 0 as
  `KMD_SEG_POST` commands, which core 0 forwards to the host. A SUBMIT sends the record
  with `KM_F_QUEUE` and the stream's tag in the header's slot bits; it waits while the
  queue's length (`KMD_ERING_QUEUED`) and the engine's jobs not yet taken in
  (`KMD_ERING_TAKEN`) fill the queue. At a stream's error or cancel the engine sends
  `KMD_SEG_DROP` (247) with the tag, and core 0 drops that stream's queued jobs, each
  answered to the engine with `KM_EV_REFUSED` reason 10. `KM_SEG_CANCEL` ends a
  stream with reason 10; its records in flight still free their slots. A record of a
  stream that misses its deadline ends the stream with reason 8 and the kernel with it.
  EXIT waits until no stream runs.
- **Graphs.** `KM_SEG_RUN` with a2 = 1 (a0 offset, a1 bytes, a3 and a4 the binding
  block) or a2 = 2 (a0 the req of a kept graph) runs a graph on core 1 beside the
  streams (`docs/graphs.md`, "In the dispatch kernel"): up to two at once, eight kept.
  Core 0 refuses a2 above 2 with reason 1, a third graph with 6, a2 = 2 of a req no
  kept graph has with 7, and ends a bad range at once with `KM_EV_GRAPH_ERROR` reason
  1; core 1 checks the rest. `KM_SEG_CANCEL` of a running graph's req ends it with
  reason 10, of a kept one that is not running drops it (the same event). A graph's
  records go back to the engine like a stream's; `KM_EV_GRAPH`, `_END` and `_ERROR`
  go to the host as `KMD_SEG_POST` commands, an end carrying in a3 whether the graph is
  kept. EXIT waits until no graph runs.
- **Commands from the engine.** The engine may send records, `KM_SEG_DEADLINE`,
  `KMD_SEG_POST` (248) and `KMD_SEG_DROP` (247); STOP, EXIT, RUN and CANCEL from it are
  refused with reason 1, and so are `KMD_SEG_POST` and `KMD_SEG_DROP` from the host.
  Core 0 never holds an engine command: a record that cannot be issued now (no free
  slot, a busy overlapping team under `KM_F_EXCL`, a full queue) is consumed and
  answered `KM_EV_REFUSED` reason 11 (`KMD_REFUSE_BUSY`) at once, the refusal stored
  before the consumed count. After a busy refusal core 0 owes the engine one
  `KMD_EV_WAKE` (0x12), which it pushes when a slot next frees (any source's record) or
  the queue issues a job; its room in the engine's event ring is kept from the refusal
  on. Core 1 does the waiting: a refused stream sends the same ISSUE or SUBMIT again
  after the next wake, the other streams go on meanwhile.
- **Commands from a work cluster.** The argument block's `work_*` words name the ring
  (`work_cmds` 0: none); the host zeroes its command lines before the start and the
  thread's event area before the thread starts. The thread may send records,
  `KM_SEG_DEADLINE` and `KM_SEG_PING`; STOP, EXIT, RUN, CANCEL and the engine's commands
  from it are refused with reason 1 (the run belongs to the host, which stops the
  thread before the kernel). A record of the thread that cannot be issued now (a busy
  overlapping team under `KM_F_EXCL`, a full queue, no free slot) is consumed and
  refused with reason 11 at once, with the same `KMD_EV_WAKE` (ticket 0) owed as to the
  engine, so a thread never holds core 0 (`kmx_wait` passes wakes over, `kmx_poll`
  returns them); a thread that stops
  reading its events stops having its commands taken once its event ring is full, and
  nothing else waits for it. Looking at the ring every fourth pass costs core 0's idle
  pass 32 to 41 ns (the experiment `x280_threads/ring_cost.py`); a STEP and SUM round of
  `examples/07_work_cluster`, two records issued and waited for in turn, takes about
  50 microseconds from the thread.
- **What streams cost the sweep.** Measured on a p150a with the harness's
  `stream-bench` case (pass period of core 0 = run time / passes): 32 ns with no
  stream, 49 ns with one or four streams that only loop, 89 ns with one stream issuing
  one record at a time, 159 ns with four. One issuing stream takes 2.3 microseconds per
  record, 1.34 microseconds (median) from a record's free to the stream's next issue
  over the ring; with three looping streams beside it, 12.8. A host record at depth 16
  to one core costs 0.641 microseconds (host-fed; 0.669 with all 64 commands preloaded).
- **Finish.** `KM_SEG_STOP`, the runtime's stop, a missed deadline or a start-up failure
  ends the loop. Core 0 then sets the engine's stop word and waits for core 1 to leave
  its loop (`KMD_ENGINE_STOP_TICKS`, 10 ms), forwarding its last posts; if it does not,
  the kernel ends with `KMD_ERR_ENGINE`. Core 1 returns before core 0 does. The stop
  word goes to the grid only while a slot is busy or before EXIT was issued. The state
  and error words in the host buffer, the result block and the hart block in LIM
  (`KMD_LIM_HARTS`: each core's progress stamp, the engine's state and counts) then
  hold the outcome.
