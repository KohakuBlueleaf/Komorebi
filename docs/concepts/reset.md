---
title: How reset works
summary: How a run gets back from a core that does not answer without a reset of the card, one step at a time - the time-out, an inspect, the reset of one core, the reset of a link core, the recovery of an island - what each step does on the card, what it was run on, what it costs, and where it stops.
tags:
  - concepts
  - recovery
  - islands
---

# How reset works

Status: built up to the recovery of an island and the host's reset of a link core; a
core stuck inside a compute unit is not recovered and needs a reset of the card. This
page says how each step is done on this hardware and where it ends; not every case is
covered, and each step says what it was run on. For what to do at the keyboard when a command does not
return, read [Recover from a command that does not return](../guides/recovery.md), and
for finding the missing call in a kernel,
[Debug a command that does not finish](../guides/debug-a-command-that-does-not-finish.md).

## The steps

A step is tried only when the one before it was not enough. Every number is from one
Blackhole p150a (card 0) or two (cards 0 and 1 as islands), slow dispatch, on the date
given; the experiment that made it is in
[core_reset](../../experiments/recovery/core_reset/README.md).

| Step | What it does on the card | Run on | Cost |
|---|---|---|---|
| 1. A command gives up | At its deadline the dispatcher tells the cores to leave the command; the run goes on | the map/collect example, 2026-11-14 | the deadline (default 1 s); a core still in the command 20 ms later goes on to step 3 |
| 2. Inspect | The dispatcher reads each core's state and live words from the core's L1, and stops nothing | one card 2026-11-14, two cards 2026-11-14 | 2.9 ms for 120 cores |
| 3. Reset one core | RISC reset, firmware and kernels written again, the core takes commands again, the others keep running | one card, core 15, 2026-11-14 | 1.7 ms by the dispatcher, about 65 ms by the host |
| 4. Reset a link core | The host closes the core's connections at their routers, resets and loads it, the core opens them again | two cards, 2026-11-14 and 2026-11-15 | 46 to 105 ms, then 0.2 to 0.3 ms to open |
| 5. Recover an island | Each island's dispatcher resets its own cores at a give-up | two cards, 2026-11-14 | the same as step 3, on that island only |
| 6. A core stuck in a compute unit | Not recovered | one card, 2026-11-13 | a reset of the card |
| 7. The dispatcher resets a link core | Not built | | |

## 1. A command that gives up

Every command has a deadline: the seconds it may take from submit to its group's
completion, set with `session.load(program, deadline=s)` (default 1.0). At the
deadline the dispatcher reads the group's cores (step 2), writes the command's sequence
number into the abort word of each of them, and every runtime wait that polls the
word returns. The cores leave the command, clear its pages and take the next one; the
dispatcher then frees the slot and tells the host. `run.wait` raises
`km.CommandTimeout` with `given_up` True and the run goes on.

What the time-out names, so that a reader can tell the cause from the victim:

- the command (its ticket and handle), its slot and sequence number, and its group;
- each core of the group and the wait each of its kernels was in when the dispatcher
  read it (`reader waits in port`, `writer waits in data`; the names are the
  [where codes](../reference/runtime.md#gave-up-and-where-codes)), or `no give-up
  recorded` for a core that was not in the command;
- the cores the dispatcher reset (`recovered`, step 3);
- on islands, the island (`island 1:`) and the cores by flat rank; for a link core,
  the link and what it held (steps 4 and 5).

A time-out that was not given up (`timed out; not reported`) is another thing: a
command the dispatcher does not give up (an X280 thread's own, a dynamic run's) or a
core that did not leave. That run is stopped and read back, and cannot be used again.

Run on one card with the map/collect example, a handler that never returns on one core,
2026-11-14: with a 2 s deadline the command was given up and its time-out came 2.03 s
after the submit. The waits that poll the abort word are listed in the
[runtime reference](../reference/runtime.md#giving-up-one-command); a loop of a program's
own polls nothing, and for that core the run goes on to step 3.

## 2. Inspect a core

`run.inspect(group)` asks the dispatcher to read, for each core of the group, its
state words, its live words (the runtime wait each kernel is in and the command it
runs) and the trace entries of its latest four commands. The dispatcher reads them
from the core's control block in L1, one core a pass of its loop beside the commands in flight, and keeps them in its local memory; the host reads
them back. It reads nothing else on the core and stops nothing, so it can be asked of
a core that hangs, and the give-up at a deadline uses the same read. A core the
dispatcher cannot reach comes back as not read.

On one p150a with the map/collect example (2026-11-14, X280 at 1000 MHz): an inspect of
all 120 cores takes the dispatcher 2.9 ms, about 24 microseconds a core; NOP commands in
flight take 2.4 microseconds each alone and 8.6 with an inspect queued behind them. On
two cards each island's dispatcher reads its own cores and the answer is by flat rank;
an inspect of island 1's core while a command hung there named the reader waiting for a
port arrival (2026-11-14). An inspect of a core that is stuck inside a compute unit
was not run.

## 3. Reset one core while the others run

A core that did not leave a command (a loop of its own, 20 ms after the abort) can be
put back without touching the other 119. The reset does what the open of the card did
for that one core, because a RISC reset alone does not bring a core back:

1. The soft-reset register of the core is written to hold its five RISC CPUs in reset.
   This is the only register written.
2. The firmware's five images, its tables (bank to NoC coordinates and the coordinate
   maps), the boot jump at L1 address 0 and the first launch messages are written
   again. The first launch of the run overwrote the tables and the boot jump, since the
   kernels' area covers the scratch the firmware's start copies them from. Without
   them the firmware either never reports it started, or starts and its DRAM reads go
   to the wrong coordinates and never finish.
3. The first CPU (BRISC) is released and the dispatcher waits at most a second for the
   firmware to report.
4. The core's part of the launch is written (kernel binaries, buffer configuration,
   runtime arguments, launch message, go), and the kernels' start-up clears its ring and
   state words. The dispatcher's reset ends only when the reader's start-up wrote its
   state magic again (the list zeroes it first): a command placed before that would be
   cleared from the ring. A core that is not a column leader tells nothing else in the
   run that it restarted.

The writes are kept as a list for each group of cores with the same kernels, which
`session.load(program, recover=True)` (the default) hands to the dispatcher at load: 68,864
bytes and 3.8 ms of the load for the map/collect example on 120 cores. With the list
the dispatcher resets the core by itself at the end of a give-up (the core "recovered"),
or on the host's call `run.reset_core(core)`; without it the host makes the same writes
through the runtime's own calls.

Run on one p150a, core 15 (not a column leader, not core 0, whose start-up others
wait for), the map/collect example on 120 cores, 2026-11-14, every command checked
page by page before and after:

| What was reset | By | Time |
|---|---|---|
| a healthy core | the host | 65.4 ms: assert 0.4, firmware start 31.2, kernels 33.8 (32.6 ms in another run of the same writes) |
| a healthy core | the dispatcher | 1.69 to 1.70 ms on the X280, 1.8 to 1.9 ms from the host's command to its answer |
| a core whose reader never returns | the dispatcher, at the give-up | the deadline plus 20 ms plus 1.7 ms |
| a core whose math thread never returns | the dispatcher, at the give-up | the same |

Every command after each reset was exact, on the core's group and on all 120 cores;
the other cores saw nothing but the give-up itself. Limits of what was built: the
commands already in that core's ring are lost; the dispatcher's loop is blocked for the
reset (1.7 ms a core); a core with semaphores is not given a list, and neither is a run
whose groups switch program (`run.recovery_refused` says why, and then a core that does
not leave ends the run with its time-out).

## 4. Reset a link core from the host

A link core holds its island's connections to the other card: each of its two
data-movement CPUs holds one connection to the router that tt-metal's fabric runs on
an Ethernet core, opened at load and kept until the run ends. A router has one
channel for a core of the card, so a link core that is reset and loaded again
cannot simply open a new connection: the router still holds the old one. That is why
the connection has to be closed and reopened, and why a link core gets no reset list for
the dispatcher.

`run.reset_link(island, k)` does it from the host, in this order:

1. each of the core's connections is closed at its router (a close request, the
   router's teardown awaited, and the sender's cursor at the router set to the router's
   own read counter), so that the router no longer waits for the old core;
2. on link core 0, the ask and put lines that were not yet sent are dropped (their
   commands were already given up);
3. the core is reset and its kernels loaded again as in step 3;
4. the core opens its connections again and says hello to the far link core, which
   needs nothing done on it.

Before it, a link core that does not leave a command is found by the dispatcher
([recover](../guides/recovery.md)): its time-out says
`link core k: did not leave the command: reset it with run.reset_link`, and
`run.links_hung` lists it. Every other command whose traffic it held is given up at once
and named as lost (a crossing whose line was on it, a command whose cores wait for an
ask or a put through link core 0). Nothing that was lost is sent again; the packet
number of a lost crossing is taken again, so the far dispatcher does not wait for a gap.

Run on two p150a cards (a handler that never returns on island 0's link core 0 while
island 0 reads pages island 1 holds, and while it crosses a command to island 1;
example 01's islands script, example 25's count step and the `links.py` experiment),
2026-11-14 and 2026-11-15: in the eight runs with a hung link core the dispatcher named
it 20.05 to 20.26 ms after its command's deadline, and the command that held traffic on
it was given up 0.08 to 2.9 ms later. In all twelve runs the host's reset took 46 to
105 ms (the closing and assert 11 to 36 ms, the firmware 17 to 34, the load 17 to 34;
the same writes vary up to twofold from run to run) and the connections were open
again after 0.16 to 0.31 ms. The runs that proved the step, a read across the link and
a crossed command each checked page by page before and after the reset, were exact.

## 5. Recover an island

Each island of a session has its own dispatcher, so recovery is each dispatcher's
own, over its island's cores. `session.load` hands every island's dispatcher the reset
lists for its island's ranks (link cores excluded: step 4): 104,192 bytes and 3.4 to
4.0 ms of the load for each island, with 118 or 119 ranks on a p150a card. A command
given up on island i resets, at the end of its give-up, the cores of that island that are
still in it or whose kernels had ended at a wait's own limit; the time-out starts
`island i:` and names the cores by flat rank, `run.recoveries` keeps each one, and
`run.reset_core(core)` (`core` a flat rank) resets one core by hand. The run goes on
on both islands.

Run on two p150a cards, 2026-11-14: a handler that never returns on island 1's core 15,
in a group of island 1's ranks and in one group over both islands' ranks. Each was given up at
the 2 s deadline, the time-out named island 1 and the core as recovered, the next
command on the group was exact, and `run.stop()` was clean. A host hang given up on island 1
in example 25 (2026-11-14) also resets the cores of its group whose kernels had ended at
their own wait limit, so that the run's exit finds its leader, and the run stopped
clean.

Not run: the watch on the dispatcher's own loop on islands (it was run on one card);
a recovery on a card split into islands, which keeps no reset lists
(`run.recovery_refused` says `a card split into islands`) so that a core that does not
leave a command ends the run.

## 6. What is not recovered: a core stuck inside a compute unit

The reset of steps 3 and 4 holds the RISC CPUs in reset. A core whose compute unit
(the unpacker, the vector unit, the packer) is still working when its CPUs are stopped is
not brought back by it. What was tried and seen on one p150a, 2026-11-13:

- A core whose reader loops forever, whose math thread waits on a mailbox nobody feeds,
  or whose packer reads past its window: stopped by the reset the runtime issues at close
  and at the next open; the next open and the next programs were right.
- A core whose three compute CPUs looped over unpack, the vector unit and pack with
  no check of the stop word: the packer's output ran over the core's low L1, the
  runtime's own flags included, and the next open failed with the runtime's `failed to
  initialize FW` for that core.
- Writing the reset bits of the Tensix units to the soft-reset register of every core,
  beside the RISC bits, after that failed open: the next open passed and the other 119
  cores worked, but that core's kernels never finished a command again.

Each time the card needed a reset (`tt-smi -r`). The same state came back on
2026-11-14 from a compute kernel built against the wrong buffer formats: the first
command on that core timed out, the grid did not return, and the open failed twice.
The dispatcher's reset of such a core was not run; that it would not help is an
inference from the host's writes above, not a result. The only prevention found is
a rule for programs: every loop that drives the units checks the stop word. After a
`km.GridTimeout` the session does not close the device; at the process's exit the
runtime waits for the cores, leaves those that did not end as they are, then holds their
RISC CPUs in reset, and the next open resets them again. A thread that is only stuck is
recovered by that open; on Wormhole a core has, rarely, been left with its compute unit
not ready (its `INSTRN_BUF_STATUS` reads 0x11, not 0x77: a compute kernel there never
finishes, BRISC and NCRISC kernels do), which that open does not see. The Tensix
dispatcher reads that word of each core at its start and of a core after its reset of
it; the open raises `km.StartError`, a given-up command's error names it: "core N's
compute unit is not ready (status 0x11: the instruction FIFOs of threads 1, 2 are
full); on Wormhole no reset by software clears this; the card needs a reset" ([recovery](../guides/recovery.md#when-a-card-reset-is-still-the-only-way)).

## 7. Not built: the dispatcher resets a link core by itself

The dispatcher resetting a link core by itself is not built: the host does it, with
`run.reset_link`.

## See also

- [Recover from a command that does not return](../guides/recovery.md), the steps as
  calls to make, with the time-out's text.
- [Giving up one command](../reference/runtime.md#giving-up-one-command),
  [inspecting a run](../reference/host.md#inspecting-a-run) and
  [the watch on the dispatcher](dispatch.md#the-watch).
- [Islands](islands.md#when-a-core-stops) and the [islands reference](../reference/islands.md#the-run).
- [core_reset](../../experiments/recovery/core_reset/README.md): every run behind these numbers.
