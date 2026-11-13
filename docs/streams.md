---
title: Micro-op streams
summary: The upload format, the Python builder, the check and the simulator for the small programs the dispatcher runs to decide which commands to submit.
tags:
  - reference
  - dispatch
  - streams
---

# Micro-op streams

A stream is a small program that runs on the card's dispatch cluster and decides which
commands to submit to the Tensix grid: which handler, on which group, with which
arguments, in which order and how many at once. The host builds it, uploads it once
and waits for one event, instead of sending one command per command.

Terms, defined on [the Tensix runtime](concepts/runtime.md) and
[the dispatcher](concepts/dispatch.md) pages: a *command* is the 64-byte unit sent
to the grid; a *handler* is the number of the code a command runs (1 to 239 belong to
the program); a *group* is a contiguous range of cores `[first, first + count)` out of
the grid's cores (119 on the card this is developed on; the simulator's `n_cores`); the
group's *leader* reports its completion with one word. The kernel keeps 16 *completion
slots*: a command occupies one from submit until its group's acknowledgement is swept up,
so at most 16 commands are in flight. The kernel loop runs in *passes*, described
[below](#the-kernel-pass-and-the-job-queue).

The code is `komorebi.engines.stream`; the shared definitions are in
`src/komorebi_native/include/komorebi/abi/km_stream.h`.

## Building a stream

```python
from komorebi.engines.stream import Stream

HANDLER_MAP = 3

s = Stream(deadline_us=200_000, max_steps=100_000, seed=1, req=1)
i, count = s.regs(2)
top = s.label()
s.rand(count)
s.mod(count, count, s.const(119))
s.addi(count, count, 1)
s.enqueue(HANDLER_MAP, count=count, args=[i, 77])
s.addi(i, i, 1)
s.jlt(i, s.const(1000), top)
s.wait(0)
s.halt(s.const(0))
upload = s.encode()
```

This submits 1,000 jobs, each on a random number of cores from 1 to 119, and ends when
all are done.

- `Stream(deadline_us, max_steps, seed, req)`: the deadline applies to every command the
  stream submits (0 uses the kernel's current default); the stream ends with an error
  after `max_steps` micro-ops; `seed` (not 0) starts the random generator; `req` is the
  request id of the stream's events.
- `s.regs(n)` gives `n` registers. There are 16; register 15 holds the generator state
  and starts as the seed, so 15 are free. Asking for more raises an error that says how
  many were asked for and how many are free.
- `s.const(value)` gives a register that is set once before the first micro-op; equal
  values share one register.
- `s.label()` is a label at the next micro-op; `s.forward()` and `s.bind(label)` make
  a label for a later position. Jumps take labels.
- `s.submit(handler, group=(first, count), args=[...], exclusive=False, result=False)`: `first`
  and `count` are integers or registers; `args` are up to ten integers, of which up to
  four may be registers. `exclusive=True` makes this one command wait at its own submit
  until every job queued before it (by this stream's `enqueue` or by any other source)
  has left the job queue and no busy group overlaps its group; it does not stop a later
  plain command from overlapping it.
- `s.enqueue(handler, count, args=[...], exclusive=False, result=False)`: the same through the
  job queue; `count` is an integer or a register and the allocator chooses `first`.
- `s.encode()` gives the upload bytes; `Stream.decode(bytes)` is its inverse and raises
  `StreamFormatError` for an upload the check rejects.

The builder rejects what the check would reject, and names the field: `max_steps`,
`seed`, `req` and `deadline_us` out of range, a `wait` count outside 0 to 4,294,967,295,
a program of more than 65,535 micro-ops, an argument that does not fit 32 bits, a group
field that does not fit 16 bits, a handler outside 1 to 239.

## Micro-ops

Registers are unsigned 32-bit and arithmetic wraps. The operand fields a micro-op reads
are the ones in its row; every other field and byte is ignored by the kernel and the
check.

| Micro-op | Meaning |
|---|---|
| `SET a, imm` | `a = imm` |
| `MOV a, b` | `a = b` |
| `ADD a, b, c` · `SUB` · `MUL` · `AND` | `a = b op c`; `MUL` keeps the low 32 bits |
| `ADDI a, b, imm` | `a = b + imm` |
| `SHR a, b, imm` | `a = b >> (imm & 31)`, logical |
| `MOD a, b, c` | `a = b % c`; `c` = 0 ends the stream with reason 7 |
| `RAND a` | on register 15: `x ^= x << 13; x ^= x >> 17; x ^= x << 5`; then `a = x` |
| `JMP imm` | go to micro-op `imm` |
| `JLT a, b, imm` | go there if `a < b`, unsigned |
| `JEQ a, b, imm` | go there if `a == b` |
| `JNZ a, imm` | go there if `a != 0` |
| `HALT a` | wait until no command or job of the stream is pending, then end; the status is `a` |
| `SUBMIT imm` | submit command template `imm` on its group |
| `ENQUEUE imm, a` | queue template `imm` as a job of `a` cores |
| `WAIT imm` | go on when at most `imm` of the stream's commands and jobs are pending; 0 = all done |
| `RESULT a` | `a` = result of the latest completed command that had `result=True` (0 before any) |
| `EVENT a` | post a progress event; its aux is the low 24 bits of `a` |
| `TIME a` | `a` = low 32 bits of the tick counter |

A command is pending from `SUBMIT`, or from `ENQUEUE`, until the grid acknowledges it. A
queued job carries the argument values the registers had when `ENQUEUE` ran, not the
values they have when the job is finally submitted.

`SUBMIT` and `ENQUEUE` fill a command from the template and the registers named in its
patch list, then wait if they cannot go on: `SUBMIT` waits while all 16 completion slots
are busy, and with `exclusive=True` also while a job queued before it is still waiting
in the job queue or any busy group overlaps its group. A group
is outside the grid when `count == 0 || first > n_cores || count > n_cores - first`
(no sum is formed, so nothing wraps); `SUBMIT` onto such a group is an error before any
waiting, and so is an `ENQUEUE` whose count is 0 or above the grid. A program counter that
reaches the end of the micro-ops is an error, as is a stream that has run `max_steps`
micro-ops; see [Errors](#errors).

A command submitted with `result=True` makes the group's leader return one 32-bit value with
its acknowledgement; `RESULT` reads it. When several results are swept together, the one
from the highest slot is the latest.

## The upload

All fields are little-endian.

| Part | Layout |
|---|---|
| Header | 64 bytes: `magic` 0x4B4D5354, `version` 1, `n_ops`, `n_commands`, `max_steps`, `deadline_us`, `req`, `seed`, then 32 zero bytes |
| Micro-ops | `n_ops` times 16 bytes: `op`, `a`, `b`, `c` (one byte each), `imm`, `imm2`, 4 zero bytes; then zero padding to 64 bytes |
| Templates | `n_commands` times 80 bytes: the 64-byte command, `arg[4]`, `reg[4]`, `first_reg`, `count_reg`, `n_patch`, 5 zero bytes; then zero padding to 64 bytes |

A template's command carries the handler, the flags (`exclusive` is bit 1 and `result`
bit 2 of the flag byte), the group if it is fixed and the integer arguments. The kernel
overwrites the sequence and landing words at submit. Patch `k` (for `k < n_patch`)
copies register `reg[k]` into argument `arg[k]` (0 to 9); a later patch of the same
argument wins. `first_reg` and `count_reg` are a register number, or 0xFF to take
`first` or `count` from the command. `ENQUEUE` takes the count from its own register and
ignores `first_reg` and `count_reg`; the submitted command never carries the queue flag.

Every byte the layout does not name (the padding, the header's last 32 bytes, unused
operand fields, `imm2`, the command's sequence and landing words and the slot bits of its
`hdr` word, the template's last 5 bytes, patch entries past `n_patch`) is ignored.

Opcodes: SET 1, MOV 2, ADD 3, ADDI 4, SUB 5, MUL 6, AND 7, SHR 8, MOD 9, RAND 10,
JMP 16, JLT 17, JEQ 18, JNZ 19, HALT 20, SUBMIT 32, ENQUEUE 33, WAIT 40, RESULT 41,
EVENT 48, TIME 49. Opcode 0 is invalid. A stream has at most 65,535 micro-ops.

The host writes the upload as whole 64-byte lines into the region the kernel publishes,
then sends the `KM_HANDLER_RUN` command (`a0` = offset in the region, `a1` = byte count).
`KM_HANDLER_CANCEL` (`a0` = request id) ends a running stream with reason 10. At most four
streams run at once; a fifth `KM_HANDLER_RUN` is rejected, and the host does not write into
the range of a running stream.

`check_run(upload, offset, region_size, active=(), active_reqs=())` gives what the kernel
must answer a `KM_HANDLER_RUN`: `(error, req)`, where `req` is the id for the error event
(the header's, or 0 if the header is not inside the region). The `RUN` is rejected with
reason 1 when the offset is not a multiple of 64, when the range is not inside the
region (tested as `offset > region_size || length > region_size - offset`), when the
range overlaps an `active` `(offset, length)` range, or when the header's `req` is an
active stream's. Otherwise the answer is that of `check`.

## The check

`check(upload, region_size)` is what the kernel does before it runs a stream. It returns
`None`, or `StreamError(reason, pc)`, the reason and program counter of the error event.
The checks run in this order and the first fault wins.

1. Header (reason 1, pc 0): at least 64 bytes and a multiple of 64; at most the region
   size (a stream as large as the region is accepted);
   magic and version; `seed` not 0; `n_ops` from 1 to 65,535; `n_commands` at most 65,535;
   `max_steps` not 0; the total size equals the header's layout exactly.
2. Each micro-op in order: an unknown opcode (reason 2); a register number of 16 or more
   in a field the micro-op uses (reason 4); a jump target of `n_ops` or more (reason 3);
   a template index of `n_commands` or more (reason 5), at the micro-op.
3. Each template in order (reason 5), at the pc of the first micro-op that uses it, or 0
   if none does: a handler outside 1 to 239; a flag outside exclusive and result; more
   than four patches; in the first `n_patch` entries an argument number above 9 or a
   register above 15; `first_reg` or `count_reg` neither 0xFF nor below 16.

A bad upload ends in one error event and the stream never starts. The event carries the
header's `req`, or 0 if the header is too short to read.

## Events and errors

Events are 16-byte `km_event_t` commands with `req` set from the header. The status word
is the kind in the low 8 bits and the aux above it.

| Kind | Posted | Aux |
|---|---|---|
| `KM_EV_STREAM` 4 | by `EVENT a` | low 24 bits of `a` |
| `KM_EV_STREAM_END` 5 | when `HALT` ends the stream | low 24 bits of the halt register |
| `KM_EV_STREAM_ERROR` 6 | on any error; the stream posts nothing after it | reason in the low 8 bits, the program counter in the 16 bits above |
| `KM_EV_TIMEOUT` 1 | after a stream's error, for a command it left in flight that passes its deadline; the grid is stopped | the slot |

### Errors

| Reason | Meaning | Program counter |
|---|---|---|
| 1 | bad header, or an upload of the wrong size | 0 |
| 2 | unknown opcode | the micro-op |
| 3 | jump target out of range, or the program counter ran past the last micro-op | the jump; or `n_ops` |
| 4 | register number out of range | the micro-op |
| 5 | bad template or patch, or template index out of range | the micro-op that uses it (0 if none) |
| 6 | step limit | the micro-op that would have been the `max_steps + 1`th |
| 7 | `MOD` by zero | the `MOD` |
| 8 | a command's deadline passed (the grid is also stopped); if several are late in one pass, the lowest slot's | the `SUBMIT` or `ENQUEUE` that made it |
| 9 | an `ENQUEUE` whose count is 0 or above the grid | the `ENQUEUE` |
| 10 | cancelled | the micro-op the stream had reached |
| 11 | an `SUBMIT` whose group is empty or leaves the grid | the `SUBMIT` |

After an error the stream's queued jobs are dropped. Commands it had already submitted stay
on the grid until they complete and nobody waits for them, but they keep their
deadlines: a late one stops the grid and posts a `KM_EV_TIMEOUT` event for its slot with
the ended stream's `req`.

Every command has a deadline, counted from its submit. The header's `deadline_us` sets it;
0 means the kernel's default deadline, which is a start-up setting and never zero (a
`KM_HANDLER_DEADLINE` command of 0 is rejected).

## The kernel pass and the job queue

The kernel loop runs in passes. In one pass it sweeps acknowledgements (freeing
completion slots, lowest slot first), checks deadlines, takes host commands, then gives
each submitter a turn: a stream runs for at most 64 micro-ops (a micro-op that has to wait
does not count and ends the turn); the queue submits what fits. The submitters are the
active streams in the order they started, then the queue. The first turn rotates: in
pass `p`, with `n` streams, the turns start at submitter `p % (n + 1)` and go on cyclically.
With one stream, the stream goes first in even passes and the queue in odd ones, so
neither can keep the other from freed slots.

A command takes the lowest free completion slot. The group allocator returns the lowest
`first` such that cores `[first, first + count)` overlap no busy slot's group.

The queue holds at most 64 waiting jobs, shared by every stream and by queued jobs from
the host; an `ENQUEUE` into a full queue waits (the kernel rejects the engine's job with
reason 11 and wakes it when there is room), and a queued command from the host into a
full queue is rejected with reason 9. Queued jobs run in the same 16 completion slots as
every other command, so at most 16 commands of all sources are in flight over the grid,
whatever their group sizes. Each pass
the queue looks at the first 16 waiting jobs in order and submits each that finds a free
slot and room. A job that does not fit does not block those behind it, but once 1,024
jobs have been submitted ahead of it nothing behind it is submitted until it fits, so a large
job is not starved.

What a queued command costs. Measured on one p150a, X280 at 1,000 MHz, 480 commands of
8 cores in windows of 48 submitted by the host and waited for at once, queued or on fixed
disjoint groups (`tests/hardware/host/session_cases.py queue-cost`), in microseconds per
command:

| | Plain, 8 cores | Queued, 8 cores | Plain, leader reads 100 pages | Queued, the same |
|---|---|---|---|---|
| host: `submit` | 2.77 | 2.69 | 2.51 | 2.53 |
| host: submit, store and event, per command | 10.08 | 9.59 | 17.65 | 17.82 |
| kernel: first command to last event, per command | 9.03 | 9.08 | 17.15 | 17.24 |
| kernel: submit to acknowledgement (event ticks) | 2.29 | 2.44 | 38.62 | 38.65 |
| most commands in flight | 3 | 2 | 4 | 4 |

The queue adds no time of its own that shows: a queued job is placed in the pass that
takes it when a slot and room are free, and its command then costs what a plain one
does. Host-fed commands arrive about 10 microseconds apart, so the kernel waits for
commands; a program that sees tens of microseconds per queued command is paying for its
commands' work, the free cores they wait for, or the host's waits between them.

## The simulator

```python
from komorebi.engines.stream import Stream, simulate

done = s.simulate(delay=lambda command: 1 + command.seq % 3)
# the same, from upload bytes
done = simulate(upload, n_cores=119, delay=0)
```

`simulate(upload, ...)` and `Stream.simulate(...)` run the check and then the stream as
the kernel would; an upload the check rejects ends in one error event at pass 0. They return `SimResult(commands, events, status, passes, error, halt,
stopped)`:

- `commands`: every command submitted, in order, as `SubmittedCommand(seq, slot, handler, first,
  count, flags, args, pass_index, pc, queued)`; `args` has ten words.
- `events`: `Event(kind, req, aux, pass_index)`, in order.
- `status`: `"ended"` (`halt` holds the status), `"error"` (`error` holds the reason and
  pc) or `"stuck"` (no end within `max_passes`). `stopped` is true when a late command
  stopped the grid.
- `passes`: the number of passes up to the one that posted the last event.

How completions arrive is a parameter. `delay` is an integer or a function of the
command: a command submitted in pass `p` is acknowledged by the sweep of pass
`p + 1 + delay`, and `None` means never. `result(command)` gives the value of a command
with `result=True`. `deadline_passes` makes a command late when it is still busy after
the sweep of pass `p + deadline_passes`; it replaces the header's deadline, while
`default_deadline_passes` (1,000,000 unless given; both at least 1) stands for a header
with `deadline_us` 0. `cancel_pass` delivers a cancel command in that pass. `n_cores` is
the grid size (119 by default); `ticks_per_pass` sets `TIME` (`pass * ticks_per_pass`);
`max_passes` ends a run that does not terminate (status `"stuck"`); `region_size`
is the size the check allows an upload (1 MiB by default).

The simulator runs one stream. Pass counts, and what `TIME` returns, depend on the
model and are not what a card shows; the order of commands and events is what a card
run is compared with where that order is determined (one command in flight), and the
sets of commands where it is not.

## Reference streams

`experiments/stream_checks/streams/` holds about ninety named streams as `.bin` uploads, each with
a `.json` beside it giving the parameters of the simulation and the simulator's answer:
the check result, every submitted command (`seq`, `slot`, submit `pass`, `pc`, `queued`,
`handler`, `first`, `count`, `flags`, `args`), the events, the status, the error, the halt
value and whether the grid was stopped. Each file's notes say how completions are
modelled. They exist so that another implementation, such as the kernel's interpreter,
can replay them, and they are chosen so that a deliberate slip in the simulator is
noticed: `experiments/stream_checks/mutants.py` applies about a hundred small patches to
it and requires every one to change some saved answer. `stream_reference.py` in the same
folder rewrites the files, and with `--check` compares them with what the simulator gives.
