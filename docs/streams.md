---
title: Micro-op streams
summary: The upload format, the Python builder, the check and the simulator for the small programs the dispatch kernel runs to decide which records to issue.
tags:
  - reference
  - dispatch
  - streams
---

# Micro-op streams

Status: the format, the builder, the check and the simulator are implemented and tested
on the CPU. The interpreter in the dispatch kernel is not part of this page's code yet;
the simulator is the definition it is tested against.

A stream is a small program that runs on the card's dispatch cluster and decides which
records to issue to the Tensix grid: which segment, on which team, with which
arguments, in which order and how many at once. The host builds it, uploads it once
and waits for one event, instead of sending one command per record.

The code is `komorebi.engines.stream`; the shared definitions are in
`src/komorebi_native/include/komorebi/abi/km_stream.h`.

## Building a stream

```python
from komorebi.engines.stream import Stream

SEG_MAP = 3

s = Stream(deadline_us=200_000, max_steps=100_000, seed=1, req=1)
i, count = s.regs(2)
top = s.label()
s.rand(count)
s.mod(count, count, s.const(119))
s.addi(count, count, 1)
s.submit(SEG_MAP, count=count, args=[i, 77])
s.addi(i, i, 1)
s.jlt(i, s.const(1000), top)
s.wait(0)
s.halt(s.const(0))
upload = s.encode()
```

This issues 1,000 jobs, each on a random number of cores from 1 to 119, and ends when
all are done.

- `Stream(deadline_us, max_steps, seed, req)`: the deadline applies to every record the
  stream issues (0 uses the kernel's current default); the stream ends with an error
  after `max_steps` micro-ops; `seed` (not 0) starts the random generator; `req` is the
  request id of the stream's events.
- `s.regs(n)` gives `n` registers. There are 16; register 15 holds the generator state
  and starts as the seed, so 15 are free. Asking for more raises an error that says how
  many were asked for and how many are free.
- `s.const(value)` gives a register that is set once before the first micro-op; equal
  values share one register.
- `s.label()` is a label at the next micro-op; `s.forward()` and `s.bind(label)` make
  a label for a later position. Jumps take labels.
- `s.issue(seg, team=(first, count), args=[...], exclusive=False, result=False)`: `first`
  and `count` are integers or registers; `args` are up to ten integers, of which up to
  four may be registers.
- `s.submit(seg, count, args=[...], exclusive=False, result=False)`: the same through the
  job queue; `count` is an integer or a register and the allocator chooses `first`.
- `s.encode()` gives the upload bytes; `Stream.decode(bytes)` is its inverse and raises
  `StreamFormatError` for an upload the check refuses.

## Micro-ops

Registers are unsigned 32-bit and arithmetic wraps. Unused operand fields are ignored.

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
| `HALT a` | wait until no record or job of the stream is pending, then end; the status is `a` |
| `ISSUE imm` | issue record template `imm` on its team |
| `SUBMIT imm, a` | queue template `imm` as a job of `a` cores |
| `WAIT imm` | go on when at most `imm` of the stream's records and jobs are pending; 0 = all done |
| `RESULT a` | `a` = result of the latest completed record that had `result=True` (0 before any) |
| `EVENT a` | post a progress event; its aux is the low 24 bits of `a` |
| `TIME a` | `a` = low 32 bits of the tick counter |

A record is pending from `ISSUE`, or from `SUBMIT`, until the grid acknowledges it.

`ISSUE` and `SUBMIT` fill a record from the template and the registers named in its
patch list, then wait if they cannot go on: `ISSUE` waits while all 16 completion slots
are busy, and with `exclusive=True` also while any busy team overlaps its team. A
program counter that reaches the end of the micro-ops is an error, as is a stream that
has run `max_steps` micro-ops. An `ISSUE` onto a team outside the grid and a job that
can never fit are errors too; see [Errors](#errors).

A record issued with `result=True` makes the team's leader return one 32-bit value with
its acknowledgement; `RESULT` reads it. When several results are swept together, the one
from the highest slot is the latest.

## The upload

All fields are little-endian.

| Part | Layout |
|---|---|
| Header | 64 bytes: `magic` 0x4B4D5354, `version` 1, `n_ops`, `n_records`, `max_steps`, `deadline_us`, `req`, `seed`, then 32 zero bytes |
| Micro-ops | `n_ops` times 16 bytes: `op`, `a`, `b`, `c` (one byte each), `imm`, `imm2`, 4 zero bytes; then zero padding to 64 bytes |
| Templates | `n_records` times 80 bytes: the 64-byte record, `arg[4]`, `reg[4]`, `first_reg`, `count_reg`, `n_patch`, 5 zero bytes; then zero padding to 64 bytes |

A template's record carries the segment, the flags (`exclusive` is bit 1 and `result`
bit 2 of the flag byte), the team if it is fixed and the integer arguments. The kernel
overwrites the sequence and landing words at issue. Patch `k` (for `k < n_patch`)
copies register `reg[k]` into argument `arg[k]` (0 to 9); a later patch of the same
argument wins. `first_reg` and `count_reg` are a register number, or 0xFF to take
`first` or `count` from the record. `SUBMIT` takes the count from its own register and
ignores `first_reg` and `count_reg`; the issued record never carries the queue flag.

Opcodes: SET 1, MOV 2, ADD 3, ADDI 4, SUB 5, MUL 6, AND 7, SHR 8, MOD 9, RAND 10,
JMP 16, JLT 17, JEQ 18, JNZ 19, HALT 20, ISSUE 32, SUBMIT 33, WAIT 40, RESULT 41,
EVENT 48, TIME 49. Opcode 0 is invalid. A stream has at most 65,535 micro-ops.

The host writes the upload as whole 64-byte lines into the region the kernel publishes,
then sends the `KM_SEG_RUN` command (`a0` = offset in the region, `a1` = byte count).
`KM_SEG_CANCEL` (`a0` = request id) ends a running stream with reason 10. At most four
streams run at once; a fifth `KM_SEG_RUN` is refused.

## The check

`check(upload, region_size)` is what the kernel does before it runs a stream. It returns
`None`, or `StreamError(reason, pc)`, the reason and program counter of the error event.
The checks run in this order and the first fault wins.

1. Header (reason 1, pc 0): at least 64 bytes and a multiple of 64; at most the region;
   magic and version; `seed` not 0; `n_ops` from 1 to 65,535; `n_records` at most 65,535;
   `max_steps` not 0; the total size equals the header's layout exactly.
2. Each micro-op in order: an unknown opcode (reason 2); a register number of 16 or more
   in a field the micro-op uses (reason 4); a jump target of `n_ops` or more (reason 3);
   a template index of `n_records` or more (reason 5), at the micro-op.
3. Each template in order (reason 5), at the pc of the first micro-op that uses it, or 0
   if none does: a segment outside 1 to 239; a flag outside exclusive and result; more
   than four patches; in the first `n_patch` entries an argument number above 9 or a
   register above 15; `first_reg` or `count_reg` neither 0xFF nor below 16.

A bad upload ends in one error event and the stream never starts. The event carries the
header's `req`, or 0 if the header is too short to read.

## Events and errors

Events are 16-byte `km_event_t` records with `req` set from the header. The status word
is the kind in the low 8 bits and the aux above it.

| Kind | Posted | Aux |
|---|---|---|
| `KM_EV_STREAM` 4 | by `EVENT a` | low 24 bits of `a` |
| `KM_EV_STREAM_END` 5 | when `HALT` ends the stream | low 24 bits of the halt register |
| `KM_EV_STREAM_ERROR` 6 | on any error; nothing follows it | reason in the low 8 bits, the program counter in the 16 bits above |

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
| 8 | a record's deadline passed (the grid is also stopped) | the `ISSUE` or `SUBMIT` that made it |
| 9 | a `SUBMIT` whose count is 0 or above the grid | the `SUBMIT` |
| 10 | cancelled | the micro-op the stream had reached |
| 11 | an `ISSUE` whose team is empty or leaves the grid | the `ISSUE` |

After an error the stream's queued jobs are dropped; records already issued stay on the
grid until they complete and nobody waits for them.

## The kernel pass and the job queue

The kernel loop runs in passes. In one pass it sweeps acknowledgements (freeing
completion slots, lowest slot first), checks deadlines, takes host commands, runs each
active stream for at most 64 micro-ops (a micro-op that has to wait does not count and
ends the stream's turn), then lets the queue issue.

A record takes the lowest free completion slot. The team allocator returns the lowest
`first` such that cores `[first, first + count)` overlap no busy slot's team.

The queue holds at most 64 waiting jobs; a `SUBMIT` into a full queue waits. Each pass
the queue looks at the first 16 waiting jobs in order and issues each that finds a free
slot and room. A job that does not fit does not block those behind it, but once 1,024
jobs have been issued ahead of it nothing behind it is issued until it fits, so a large
job is not starved.

## The simulator

```python
from komorebi.engines.stream import Stream, simulate

done = s.simulate(delay=lambda record: 1 + record.seq % 3)
# the same, from upload bytes
done = simulate(upload, n_cores=119, delay=0)
```

`simulate(upload, ...)` and `Stream.simulate(...)` run the check and then the stream as
the kernel would. They return `SimResult(records, events, status, passes, error, halt,
stopped)`:

- `records`: every record issued, in order, as `IssuedRecord(seq, slot, seg, first,
  count, flags, args, pass_index, pc, queued)`; `args` has ten words.
- `events`: `Event(kind, req, aux, pass_index)`, in order.
- `status`: `"ended"` (`halt` holds the status), `"error"` (`error` holds the reason and
  pc; `stopped` is true for reason 8) or `"stuck"` (no end within `max_passes`).
- `passes`: the number of passes up to the one that posted the last event.

How completions arrive is a parameter. `delay` is an integer or a function of the
record: a record issued in pass `p` is acknowledged by the sweep of pass
`p + 1 + delay`, and `None` means never. `result(record)` gives the value of a record
with `result=True`. `deadline_passes` makes a record late when it is still busy after
the sweep of pass `p + deadline_passes`. `cancel_pass` delivers a cancel command in that
pass. `n_cores` is the grid size (119 by default); `ticks_per_pass` sets `TIME`.

The simulator runs one stream. Pass counts, and what `TIME` returns, depend on the
model and are not what a card shows; the order of records and events is what a card
run is compared with where that order is determined (one record in flight), and the
sets of records where it is not.

## Reference streams

`tests/core/data/streams/` holds named streams as `.bin` uploads, each with a `.json`
beside it giving the simulator's answer: the check result, the issued records, the
events, the status and the halt value. They cover every micro-op and every error reason,
and exist so that another implementation, such as the kernel's interpreter, can replay
them. `python tests/core/stream_reference.py` rewrites them, and a test keeps the files
equal to what the simulator gives.
