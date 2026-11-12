# Dispatch kernel

The dispatch kernel runs on core 0 of X280 cluster 0. It takes commands from the host,
issues records to the Tensix grid, sweeps the grid's acknowledgement words and reports
every finished command to the host as an event. `km_dispatch.h` holds the exact layouts;
`komorebi/abi/km_abi.h` holds the records, the events and the segment numbers.

| file | part |
|---|---|
| `main.c` | start-up, the loop, taking a command, the finish |
| `host_ring.h` | the host command ring in the cluster's local memory (LIM) |
| `event_ring.h` | the event ring and the state words in host memory |
| `grid_port.h` | the windows onto every core's control block, the record store, the stop word |
| `completion.h` | busy slots, acknowledgement sweep, deadlines, team overlap, team allocator |
| `stream.h` | the upload check and the interpreter of uploaded streams (`km_stream.h`) |
| `result.h` | the result block in LIM |
| `kernel.h` | the kernel's state, which lives in the cluster's DRAM bank |

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
  busy slot and one more fit the event ring, so it never overwrites an event the host
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
  `KM_SEG_DEADLINE` of 0. A refused command's event has ticks 0.
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
  overlap it. A waiting command holds up every command behind it.
- **Queued records.** A record with `KM_F_QUEUE` takes its count from the team word and
  ignores `first`: the team allocator gives it the lowest `first` whose cores overlap no
  busy slot's team, using the same overlap test as `KM_F_EXCL`, and the record waits
  until such a team exists. The grid sees the chosen team and no `KM_F_QUEUE`. Until the
  job queue exists, a waiting queued record holds up the commands behind it like any
  other waiting command.
- **Streams.** `KM_SEG_RUN` (a0 offset, a1 bytes in the upload region) starts the
  stream there, after the same check as `stream.check`: a range that is not inside the
  region and line aligned, one that overlaps a running stream's, a req a running stream
  has, or a failed check ends it at once with `KM_EV_STREAM_ERROR` (reason 1 for the
  first three). Up to four streams run, each at most `KM_STEP_OPS` micro-ops per pass in
  turn, and none waits inside a pass. Their records go out like the host's, with the
  stream's own deadline, and end in no event; `KM_EV_STREAM`, `_END` and `_ERROR` carry
  the stream's req, the aux `reason | pc << 8` for an error. `KM_SEG_CANCEL` ends a
  stream with reason 10; its records in flight still free their slots. A record of a
  stream that misses its deadline ends the stream with reason 8 and the kernel with it.
  EXIT waits until no stream runs.
- **What streams cost the sweep.** The interpreter runs in the same loop as the
  acknowledgement sweep, so a pass that steps streams delays every acknowledgement.
  Measured on a p150a with the harness's `stream-bench` case (pass period = run time /
  passes): 34 ns with no stream; 316 ns with one stream issuing one record at a time,
  525 ns with four; 3.1 microseconds with one stream that only loops (64 micro-ops of
  about 49 ns each per pass), 11.7 microseconds with four. A host record at depth 16 to
  one core costs 0.673 microseconds with the interpreter built in, against 0.640 without.
- **Finish.** `KM_SEG_STOP`, the runtime's stop, a missed deadline or a start-up failure
  ends the loop. The stop word goes to the grid only while a slot is busy or before EXIT
  was issued. The state and error words in the host buffer, and the result block in LIM,
  then hold the outcome.
