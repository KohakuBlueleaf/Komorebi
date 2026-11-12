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
| `completion.h` | busy slots, acknowledgement sweep, deadlines, team overlap |
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
  from 240 to 251 that this kernel does not serve (`KM_SEG_RUN` and `KM_SEG_CANCEL`
  included, until the engines exist); 2 flag bit 0 set; 3 a team with count 0 or
  `first + count` above the cores; 4 any record after EXIT; 5 a range of the upload
  region that is not whole lines inside it. A refused command's event has ticks 0.
- **Ready block.** Before the state word turns READY, the kernel writes `kmd_ready_t`
  into words 4 to 7 of the host buffer as one store: the mark, the upload region's
  LIM address and size, and `KMD_LIM_BYTES`. The host waits for the mark, not for the
  state word, since the two are separate stores.
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
- **EXIT** goes to every core whatever its team field says, and only once no slot is
  busy. A record with `KM_F_EXCL` waits until no busy slot's team overlaps its own. A
  waiting command holds up every command behind it.
- **Finish.** `KM_SEG_STOP`, the runtime's stop, a missed deadline or a start-up failure
  ends the loop. The stop word goes to the grid only while a slot is busy or before EXIT
  was issued. The state and error words in the host buffer, and the result block in LIM,
  then hold the outcome.
