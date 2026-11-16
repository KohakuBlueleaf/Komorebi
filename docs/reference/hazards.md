---
title: Hazards and the rules that avoid them
summary: What can go wrong in a program that Komorebi runs and builds without complaint, the rule that makes a program never meet it, and the guard the framework has for it (a refusal, a switch with its cost, or nothing); entries grouped by where a programmer meets them, each linked to the page that describes it at length.
tags:
  - reference
  - hazards
---

# Hazards and the rules that avoid them

A **hazard** is something that goes wrong on the card or in the framework, that a page of
this documentation has seen or states, and that a program never meets when it follows a
rule. It is a category of its own. The README's
[sizes a program works within](../../README.md#limits) are fixed
sizes of the runtime and limits of the card, which a program is written to fit; a hazard
is not a size but something a program does, or does in a certain order. What each page
lists under "Not in this version" is what is not built; a hazard is built, and works
until a program steps on it. Where a hazard costs time to guard against, the guard is a
switch, and the program that turns it off takes the rule on itself.

Each entry has the same four parts. **The hazard** is what goes wrong and what you see.
**The rule** is what a program does to never meet it. **The guard** is what the framework
does about it: it refuses (at the build, at the load or at the submit, with the error
named), or it is a switch (with its default and its cost, in the numbers the page
already has), or it does nothing and the rule is the programmer's. Where the guard says
"nothing", it means nothing: no error will tell you. The last line of an entry names the
page that describes the thing at length.

## Broadcasts, ports and messages

### A word cast followed at once by a cast on the same cores

**The hazard.** At `finish`, each receiver of a broadcast without a buffer clears its
VALID word, and the caster does not wait for that clear. The first cast of its next
command on the same cores can land before it: the clear then wipes the new VALID, that
receiver never sees the cast, and the command gives up at the wait limit. You see a
command that misses its deadline.

**The rule.** A command that casts a word on a broadcast is not followed at once by a
command that casts a word on the same broadcast's cores; something between them (the
host's wait for the first command's event, or a command of other cores) gives every
receiver the time to clear.

**The guard.** A switch, `broadcast_wait` of `km.Program` (and the same keyword at a
class's making), off by default. On (`True`), the kernels are built with
`KM_BROADCAST_WAIT` 1: each receiver marks the caster after its clear and the caster
waits at `finish` for every mark, under the core's wait limit and stop, so the rule is
not needed. Its cost on one p150a: the cast itself unchanged, a command of 2,000 casts
0.2 % longer, a command that is one cast of a word 4.94 against 5.38 µs (+0.44 µs), and
some hundred bytes of kernel text. Off, the rule is the program's.

Described in [a broadcast without a buffer](broadcast.md#a-broadcast-without-a-buffer)
and [Host API, `Program`](host.md#program).

### A broadcast's blocks not taken in order, or not popped

**The hazard.** The sender writes a slot again only when every receiver has popped the
block that was there. A compute kernel that does not pop every block, or leaves out
`broadcast_end`, or a buffer that is not exactly `window * pages` pages, leaves the
sender waiting for markers that do not come: it gives up at the wait limit and the
command misses its deadline.

**The rule.** The blocks of a command go in order on every core of the broadcast, compute
takes and pops every one of them and calls `broadcast_end`, and the buffer is a circular
buffer of exactly `window * pages` pages, used only through the broadcast's calls. A
command with broadcasts names exactly the group the broadcast words were built for.

**The guard.** Every call gives up at the run's wait limit and checks the stop flag; the
give-up names where (`broadcast_done`, `broadcast_valid`, `broadcast_free`) and the block.
Words that do not fit the command or the buffer give up at once as `bad_broadcast`
(`broadcast_begin`), and `broadcast_args` refuses a row whose cores are not consecutive
in NoC0. No card reset is needed. Beyond that, nothing: the rule is the programmer's.

Described in [Broadcast, rules for a program](broadcast.md#rules-for-a-program) and
[when a core stalls](broadcast.md#when-a-core-stalls).

### Commands that use ports or broadcasts in flight together

**The hazard.** A port's count and its arrival slots, and a broadcast's buffer, belong
to one command at a time. Two commands whose groups overlap, in flight together, land
what one sends in the other's slots or ports.

**The rule.** A command that uses ports or broadcasts carries `KM_FLAG_EXCLUSIVE`
(`exclusive=True` in `run.submit`), so overlapping groups are never in flight together.

**The guard.** For ports, `run.submit` refuses a command of a program's `port_handlers`
(the handlers of a class that reduce are in it by themselves) on cores that a port
command without its event still holds, naming both handles, unless it is submitted
exclusive. A thread's submits are not checked. For broadcasts the guard is nothing.

Described in [Ports, emit and reduce](ports.md#on-the-wire) and
[Broadcast](broadcast.md#rules-for-a-program).

### Two casters into one rectangle

**The hazard.** The chip's NoC documentation warns that unlinked multicasts from several
casters into one rectangle "can lose a flit and hang the NoC".

**The rule.** One caster for a rectangle: the sender alone multicasts each block, and the
other cores hand it their rows first.

**The guard.** Nothing. The probe of several casters was clean (800,000 casts, 1.44
million multicasts), which is a count and not the documentation's guarantee; in the dense
linear layer's prefill the step was slower at every shape measured, 1.31 to 2.07 times the
hand-over's.

Described in [Broadcast, rules for a program](broadcast.md#rules-for-a-program).

### The VALID of a block sent without waiting for the acknowledgements

**The hazard.** After `valid_when_sent()` the block's multicast and its VALID are one
linked NoC transaction. The chip's documentation orders the two packets in transit but
states no order in which a receiver's L1 takes two writes to different addresses, so a
receiver could see the VALID before the block.

**The rule.** Keep the default, the acknowledged form, which has a documented guarantee,
unless the program has measured that it needs the other.

**The guard.** A switch of the call, off by default. It is refused at build in dynamic
NoC mode. On the dense linear kernel it gained nothing measurable (4,480.4 us a call at
8192 cubed against 4,488.1), since that product is bound by compute.

Described in [Broadcast, valid when sent](broadcast.md#valid-when-sent).

### The value an atomic add returns

**The hazard.** On the card, the returned value of a non-posted atomic add is sometimes
lost when the tile that answers it submits atomics of its own on the same NoC: the add
lands and the answer never does. With every core keeping its own tail, 2 to 12 of 50 runs
of 10^6 forwards ended with reason 32 (`lost`).

**The rule.** A program counts acknowledgements and never reads a returned value, as the
runtime's ports and group completion do.

**The guard.** For a dynamic run, a switch: the tails live on counter tiles, the last
cores of the run's group (`counters`, 2 by default, 1 to 4), which only answer. A lost
answer still ends the run with reason 32, never with a wrong result. The cost is that the
counter tiles take no message, and 2,112 bytes of L1 on every core for every program: 10^6
forwards take 11.8 ms with 2 counter tiles and 11.6 ms with one. For a program's own
atomics, nothing.

Described in [Dynamic handlers, counter tiles](dynamic.md#counter-tiles) and
[ports](ports.md#on-the-wire).

### A send with no quota left

**The hazard.** Each core starts a run with `max_live / cores` sends. `send` takes one,
`consume` gives one back to the core where the message ends, and `forward` changes
nothing. A program that sends on some cores and consumes on others runs the senders dry:
`send` returns false and sends nothing, so a message the program does not look for is
lost.

**The rule.** Test the result of every `send`, and keep sends and consumes balanced on
each core: moving quota between cores is the program's business.

**The guard.** A `max_live` below the number of cores is refused before the card opens. A
`send` without quota returns false and sends nothing. Otherwise, nothing.

Described in [Dynamic handlers, how a message travels](dynamic.md#how-a-message-travels).

### Messages assumed to arrive in order

**The hazard.** Messages have no order between them, and the messages of one packet are
handled by handler, not in queue order. A program that depends on an order sees them
handled in another.

**The rule.** A program that needs an order sorts at the receiver.

**The guard.** Nothing.

Described in [Dynamic handlers, not in this version](dynamic.md#not-in-this-version).

### A message to a handler nobody has a case for

**The hazard.** A message carries any handler number below 256. When the handlers are
parts, a message whose handler is not one of the program's dynamic handlers stops the run
before compute sees it: the core gives up with reason 40 (`handler`). When the bodies are
in the program's compute kernel, Komorebi does not check the number at all.

**The rule.** Every handler number sent or forwarded is one of the program's dynamic
handlers, and a body has a case for each.

**The guard.** For parts, the give-up with reason 40. For bodies in the compute kernel,
nothing: a handler the body has no case for is the program's to report.

Described in [Dynamic handlers, a message names its handler](dynamic.md#a-message-names-its-handler)
and [handlers as overlays](dynamic.md#handlers-as-overlays).

### A start-up barrier of the program's own in a dynamic run

**The hazard.** Under slow dispatch the cores of a command start at different times. A
program that adds its own barrier between cores at start-up, with a timeout that does not
allow for that skew, sees cores give up: a timeout of 7 ms let 47 of 119 cores give up.

**The rule.** Allow more than the launch skew in such a barrier (the reservation probe
allows one second).

**The guard.** Nothing.

Described in [Dynamic handlers, how a message travels](dynamic.md#how-a-message-travels).

## Programs, kernels, overlays and switching

### Two commands on overlapping groups that must run in order

**The hazard.** Two commands in flight whose groups share a core may run on that core in
either order.

**The rule.** A command that must run after an earlier one on overlapping cores is
submitted with `exclusive=True`.

**The guard.** The flag is the guard, and it has a cost: the dispatcher submits an
exclusive command only when no command in flight overlaps it, and host commands behind a
held one wait with it, whatever their groups. Without the flag, nothing.

Described in [Host API, `Run`](host.md#run).

### An overlay loaded by one kernel alone

**The hazard.** The reader loads every part of an overlay, and an overlay with a writer
or a compute part needs `load()` called by the reader and by that kernel in the same
command, so that they meet. A kernel that loads alone waits until its wait limit and
gives up (`overlay_meet`); a `call()` without a successful `load()` gives up
(`overlay_find`).

**The rule.** The reader and each kernel that has a part of the overlay call `load()` in
the same command, and a part is called only after a `load()` that returned true.

**The guard.** Every wait is bounded and polls the stop flag, so the command ends in a
give-up, not a hang. Nothing refuses the program before it runs.

Described in [Overlays, the rules](overlays.md#the-rules).

### A part that keeps something from one call to the next

**The hazard.** The entry of a part zeroes the part's `.bss` at every call. A static
variable of a part is zero at the start of every call, and a part holds no state from one
command to the next.

**The rule.** What a part must keep from one command to the next, it reads from the
buffers' own pointers or from the command; it keeps none of it in a static.

**The guard.** Nothing for a static of the program's own. The builder refuses a part that
defines its own copy of the runtime's state (an object the persistent kernel also
defines), because the kernel's copy is the one that counts; the build's 22 constant
tables of the buffers pass when byte-equal to the kernel's. matmul.h's
`throttled_mop_status` (which matmul MOP is programmed) is set unknown in a math part's
copy at entry and in the resident kernel's after the call, so an inline body's
`matmul_block_math_dynamic_throttle` after an overlay call programs its MOP again;
compiled class code never emits that call.

Described in [Overlays, a part](overlays.md#a-part) and
[dynamic handlers as overlays](dynamic.md#handlers-as-overlays).

### What a group's buffers hold when it switches program

**The hazard.** When a group switches program (`run.load`, or a thread's `kmx_load`), the
cores' buffers are set up again for the new program: whatever the old program kept in
its buffers is gone. A command for the old program on those cores is refused afterwards,
and a data command for a program whose load did not run on a core gives up on compute
(`ProgramStart`, where 42).

**The rule.** A program keeps nothing in its buffers across a switch; its buffers are
empty when the load is submitted, and a command names the program its cores hold.

**The guard.** The load waits until every buffer of either program is empty. The host
refuses a command whose group has a core that does not hold its program (`ValueError`,
before anything is sent); in a run loaded with a dict, the dispatcher refuses the same
from any source (`CommandRejected`, reason 13), and a run whose thread loads has the
host's check off. The contents of the buffers are not saved or checked.

Described in [Host API, a group switches program](host.md#a-group-switches-program) and
[Overlays, a group switches program](overlays.md#a-group-switches-program).

### Programs on the same cores built differently

**The hazard.** The parts of a program run on any group's kernels, and the kernels of the
programs that share cores are one build. Programs that differ in compute options,
optimisation levels or the NoC of a kernel do not fit one build, and programs that
declare a buffer index two ways do not fit one layout.

**The rule.** Programs that share cores, and the programs of a dict, give the same
`compute`, `opt` and `noc`, and declare a shared buffer index identically.

**The guard.** Refused: `run.load` raises `ProgramError` and says which symbol differs,
and a list of programs with a buffer index declared two ways raises `L1LayoutError` with
the L1 budget table.

Described in [Host API, several programs in one run](host.md#several-programs-in-one-run)
and [movement, the NoC of each CPU](movement.md#the-noc-of-each-cpu).

### Formats another program left on the same cores

**The hazard.** In a run loaded with a list, every core holds every program and nothing
runs between one program's command and the next program's: the unpacker's and the
packer's data formats are what the last command left. A program that sets no format
itself (all its buffers of one format, as the runtime's start-up left them) reads or
packs its tiles in the other program's formats. You see wrong values, NaN among them,
and no error: on the card, example 22's Gate after the dense Linear's K split on the
same cores (Linear's reduce leaves the unpacker on float32).

**The rule.** A compute part in a run of several programs sets the formats it relies on
at the start of its command (`reconfig_data_format`, `pack_reconfig_data_format`), or
before each unpack and pack. A group that switches program has no such hazard: its
load runs the start-up again on the new program's buffers.

**The guard.** For a class in the runtime's form: each command starts by setting the
formats of the runtime's start-up on the program's buffers again (two calls on
compute, no cost above the noise on the card). For a hand-written part, nothing: a hand
program in a list must do the same.

Described in [Programs written as classes, the runtime's form](classes.md#the-runtimes-form)
and [Host API, several programs in one run](host.md#several-programs-in-one-run).

### An instruction the Tensix cores lack

**The hazard.** The compiler may emit an instruction a Tensix core does not implement
(float divide and square root among them, a `/` on floats being the usual cause). A core
that meets one stops without a report, and the program ends in a timeout that names no
cause.

**The rule.** Divide by an integer, multiply by a reciprocal computed on the host, or on
the vector unit use a reciprocal with Newton steps.

**The guard.** Refused at the load: `Session.load` disassembles each thread's linked ELF
and raises `km.KernelInstructionError`, naming the thread, the instruction, the function
and the source line, before anything is loaded. About 11 ms a thread the first time a
build is seen; `KMR_INSN_CHECK=0` turns it off.

Described in [Host API, instructions the cores lack](host.md#instructions-the-cores-lack).

### A wait that never completes

**The hazard.** A wait in a kernel that nothing ever satisfies (a page that is never
pushed, a marker that never comes, a NoC barrier that never clears) holds its core in
the command. Without the guard no one gets it out: the command misses its deadline, the
dispatcher frees it while the core stays in the wait, and the next command to that core
does not run.

**The rule.** The program's waits complete: every page, marker and acknowledgement a
wait names is sent by some core of the same command, on every path.

**The guard.** A switch, `guard` of `km.Program` (and the same keyword at a class's
making), on by default. On, the kernels keep live words, poll each command's abort word
in every runtime wait and time their NoC barriers: at a command's deadline the
dispatcher reads the group (`run.inspect` does the same on request), the waits return,
the cores leave, clear their pages and take the next command, and a core that cannot
leave is reset. Its cost on one p150a, example 16's empty command in the kernel form: 373
cycles on one core and 301 on 120 cores (1,012.5 against 639 and 1,149.5 against 848),
in the runtime's form 341 and 286, a sum of 16 tiles 250 to 890, and 9,520 B of kernel
text in the kernel form (34,848 against 25,328) and 6,400 B in the runtime's (30,432
against 24,032); the gate class's command of 120 cores is the same within 1 µs. Off
(`False`, kernels built with `KM_GUARD` 0): none of it. `run.inspect` raises naming the
program, the load's log says once that its commands are not given up at a deadline and
that a hang then costs a card reset, a `CommandTimeout` of such a run says that no core
was asked to leave, and the reset lists (`recover=True`) reset only a core whose own
wait limit ended its kernels. Programs that share the runtime's kernels (a list, a dict,
a merge) must agree, else `ProgramError` names two. A program known right takes the rule
on itself.

Described in [giving up one command](runtime.md#giving-up-one-command),
[Host API, `Program`](host.md#program) and
[recovery](../guides/recovery.md#the-ladder).

### A wait or a loop of the program's own that polls nothing

**The hazard.** A program's own `noc_async_*_barrier` calls have no timeout, and a loop
without a limit never ends. A core that is still in a command 20 ms after the dispatcher
gave it up cannot be reached by the give-up; without the dispatcher's reset list the
core ends the run with `KMD_ERROR_DEADLINE`, and a core stuck inside a compute unit needs
a card reset.

**The rule.** Every wait of the program's own goes through `poll` with the wait limit and
checks the stop flag, every loop has a limit, and a program never waits on a NoC barrier
for an address it has not checked.

**The guard.** A switch, `recover` of `session.load` (on by default): the dispatcher
resets a core still in the command and loads it again from the list it was handed at
the load, and the run goes on. The list costs the load about 4 ms and 68,864 bytes of
the upload region for the map/collect example. Without the list, or for a core stuck in
a compute unit, nothing.

Described in [Tensix runtime, timed waits](runtime.md#timed-waits),
[giving up one command](runtime.md#giving-up-one-command) and
[recovery](../guides/recovery.md).

### A wait whose code is dropped

**The hazard.** `core.poll` returns a code and records nothing. A helper that turns
a timeout into a bare `false` leaves no give-up word and does not set the stop flag:
the other cores of the group go on waiting for the core's part until the command's
deadline, and the deadline's inspect finds the core in no named wait. The decode
class's qkv hung that way twice on its q/k roots (2026-11-16): the movers' waits in
its header polled with no where and dropped the code, the rows read only "reader
waits in next, unpack waits in cinput", and both runs cost a card reset.

**The rule.** A wait of the program's own records its give-up: `core.wait(ready)`, or
`poll` followed by `gave_up(...)` on any code but `kGaveNone`. A copy inside L1 goes
through `km::copy_l1`.

**The guard.** None for a dropped code: the runtime cannot see it. With `core.wait`
and `km::copy_l1`, the give-up names the kernel, the where and the line.

Described in [Tensix runtime, your own waits and copies](runtime.md#your-own-waits-and-copies).

### A NoC write from memory the NoC cannot read

**The hazard.** The NoC cannot read a kernel's `static` variable (it lives in the CPU's
local memory). A write whose source is such memory never completes, and the card answers
no host read: it needs a reset.

**The rule.** The source of a NoC write is in L1 (or another memory the NoC reads), never
a `static` or a stack variable of the kernel.

**The guard.** A switch, off by default: building the program with `KM_CHECK_NOC_SOURCES`
defined stops the writing CPU before such a write, and also before a lookup of a core
past the core table (its live where word reads `coreindex`). Its cost is not given.

Described in [Recover from a command that does not return](../guides/recovery.md#when-a-card-reset-is-still-the-only-way).

### A small write whose two addresses differ in their low four bits

**The hazard.** A small write between two cores' L1 is silently wrong when the two
addresses do not share their low four bits.

**The rule.** Send a small write between two cores' L1 only with `km::small_write`, slot
to slot.

**The guard.** Nothing: the write is silently wrong.

Described in [Tensix runtime, what a program may and may not do](runtime.md#what-a-program-may-and-may-not-do).

### Stores to L1 followed at once by a send

**The hazard.** The core's stores pass through a small write-through cache. A
`cb_push_back` or a NoC send that starts before the stores land carries old bytes.

**The rule.** Fence with `invalidate_l1_cache()` between the stores that build data in L1
and the push or the send that hands it on.

**The guard.** Nothing.

Described in [Tensix runtime, what a program may and may not do](runtime.md#what-a-program-may-and-may-not-do).

### What the runtime's loop assumes of a body

**The hazard.** The runtime states four more things a body must not do. A writer body
that returns false after anything but a give-up means the group never completes. Compute
runs on three threads, and they must take the same path. The control block is the
runtime's. The compute kernel does not write to the NoC. The page states these as rules
and names no symptom for the last three.

**The rule.** A writer body returns true when it has nothing to report, and false only
after a give-up. Compute branches only on the command and on what `wait_front` returned.
A body writes the control block only through the runtime's calls, and never writes to the
NoC from the compute kernel.

**The guard.** Nothing.

Described in [Tensix runtime, what a program may and may not do](runtime.md#what-a-program-may-and-may-not-do).

### A buffer half with two producers

**The hazard.** A buffer's received counter takes one producer's count. Compute's packer
keeps its own count of the pages it pushed and writes it whole into the buffer's count,
so a half that compute fills in one command and a mover fills in another gets its count
rewound, and a second producer's pages never reach the wait.

**The rule.** A half has one filling CPU and one draining CPU for the whole load. A
stream that compute writes has its own buffer pair, and a mover's halves are its own: no
other code of that mover pushes or pops them while a stream through them is open.

**The guard.** `Program` refuses a half declared twice, and at `load` it reads the kernel
sources: one that names a half in a filling call and a draining call, or two sources that
both fill or both drain one half, are refused (`ProgramError`). Halves named through a
variable, or moved by `km::move`, are not seen. `km::move` gives up (`bad_item`, at
`data`) when a half's roles change within a load. The tile compiler's `emit` refuses a
buffer "pushed by X and Y".

Described in [Movement, rules for a program](movement.md#rules-for-a-program) and
[Host API, `Program`](movement.md#host).

### A buffer half drained by compute in one command and by a mover in another

**The hazard.** The draining side has the same counter problem as the filling side.
Compute's unpacker keeps its own count of the pages it popped and writes it whole into
the buffer's count of pages freed; a mover's pop adds to that count. A half that unpack
drains on a reduce's root and the mover sends from on the other ranks (a K split's field
of sums, `km::reduce_emit`) has its count rewound at unpack's next pop after the mover's:
the packer then sees pages free that are not, the next sum is sent early (wrong sums, a
different count of wrong words each run), and with the half full the packer's reserve
waits forever.

**The rule.** After a command in which the mover drained the half, unpack adds the
mover's pages to its own count: `UNPACK((get_local_cb_interface(cb).tiles_acked +=
pages))` under the same condition as the mover's send, as the dense project's K split
does (`projects/dense/kernels/split/compute.cpp`, on a rank other than the root).

**The guard.** A class's field reduce adds the line itself (a test checks the generated
text). The load's check of sources refuses a half drained by name in two sources, but
not one drained inside `km::reduce_emit`; a hand kernel that does so: nothing.

Described in [Movement, rules for a program](movement.md#rules-for-a-program).

### A chain's input read before it is written

**The hazard.** In a chain, a record's input waits for the marks of the cores that wrote
it. An input that the call before does not write is covered by no mark, and a core could
read a block before it is complete. A mark across islands is not built.

**The rule.** State each such input as an earlier input or an outside tensor, and give
each call the marks of the call before it.

**The guard.** Refused when the chain is made: `run.chain` proves every input and refuses,
naming the call, the tensor, the tile and the core, a core that reads before anything
tells it complete. It refuses an outside tensor that a call of the chain writes, and, over
islands, a record whose input waits for a mark from a root on another island. A chain of
a class's handler calls refuses a read of a tensor an earlier call wrote unless the call
marks it `km.past(t)`, which is then proven.

Described in [The chain, the earlier input](chain.md#the-earlier-input-and-the-direct-input),
[what is refused](chain.md#what-is-refused) and
[Programs written as classes, chains](classes.md#chains).

### A chain's read started before its input's marks

**The hazard.** In a chain of a class's handler calls, a read of a tensor that no call of
the chain writes goes before the record's input is marked (prefetch). A tensor that some
call does write, read so, could be read before that call's cores have written it: you
would see stale words, and no error.

**The rule.** Let the build choose: it prefetches only the tensors no call of the chain
writes. State `prefetch=True` on a read only of such a tensor; `prefetch=False` (on the
chain or on a stream) when a read must wait anyway.

**The guard.** Refused when the chain is made: a read stated `prefetch=True` of a tensor
a call of the chain writes, naming the call, the line and the tensor; a handler whose
inline body takes a tensor and states neither `reads=` nor `writes=` (what it reads is
its own C++), and a `kc.Call` among the class's calls (its tensors are names the build
cannot compare). The automatic prefetch is on by default and costs nothing; off, every
read waits for the input's marks.

Described in [Programs written as classes, chains](classes.md#chains).

### A broadcast's handler in a hand-stated chain on more cores than its rectangle

**The hazard.** A core the broadcast's rectangle leaves out returns from the handler
at its first broadcast line (`included()`), and in a chain's build that return skips the
record's end (`km_chain_end`), which counts the record's marks and, at a group's root,
marks the next record's cores. A chain that gives such a handler a group wider than the
rectangle's (the cores between a rectangle's rows) may leave a mark unmade: its readers
would wait for it. Read from the generated text only; no card has run it, so the hang
is suspected, not seen.

**The rule.** A chain does not call a handler that holds a broadcast on a group other
than the rectangle's whole group of whole rows; give such a handler to `run.submit` as a
command.

**The guard.** For a chain stated with a class's handler calls: refused when the chain
is made, by the read under `if b.sender():` ("a read or write under an if on the command's
words"), every cast being one. For a hand-stated `chain.Call` of the same handler:
nothing; the rule is the programmer's.

Described in [Programs written as classes, chains](classes.md#chains).

### A body's reads and writes stated wrong

**The hazard.** An inline body's `reads=[t]` and `writes=[t]` are the programmer's
statement of what its C++ reads and writes in DRAM; a chain describes the call from
them. A body that writes a tensor stated only in `reads=`, or reads or writes one it
does not state, leaves the chain with a wrong description: a call after it can read the
pages before they are written, or an outside input can change while the chain runs.
You see stale or mixed words, and no error.

**The rule.** State every tensor whose pages the body's C++ touches: `reads=` for what
it only reads, `writes=` for what it writes (both when it does both). A tensor in
`reads=` is never written by any call of the chain; one in `writes=` is read by no
other call of the chain (read it in a later chain or command).

**The guard.** At the build, `reads=` and `writes=` must name tensor parameters the
body takes (refused at the line). At `run.chain`: a taken tensor stated in neither, a
`reads=` tensor that a call of the chain writes, and a read by another call of a tensor
a body writes are refused, naming the call, the line and the tensor. That the C++ does
what is stated: nothing (the build does not read it).

Described in [Programs written as classes, a body's reads and writes
stated](classes.md#a-bodys-reads-and-writes-stated).

### A slice of a constant set's words

**The hazard.** `program.gate.args(..., 4)` gives words that carry the constant values
they state, so `submit_many`, `Graph.command`, a stream's submit and a chain's call run
the handler built for them. A slice or a concatenation of those words is a plain list:
it carries nothing and means the defaults, so the command runs the default set's
handler on words laid out for another. You see wrong results, or a command that reads
past its tensors, and no error.

**The rule.** Hand on the words `args` gave, whole; to change one, call `args` again.

**The guard.** Nothing: a plain list of words is how every caller stated words before
constant sets, and it means the defaults.

Described in [Programs written as classes, a constant parameter of a
handler](classes.md#a-constant-parameter-of-a-handler).

### Unsigned word arithmetic that wraps

**The hazard.** The words of a handler are unsigned 32-bit, as the C++ they become. A
subtraction below 0 wraps to a large word, so `a - b > 0` is true for every `a` and `b`
unless they are equal; and `a or b`, `a and b` of words are a word in Python and a truth
in C++.

**The rule.** Write `a > b` for the comparison, and `max(a, 1)` or an `if` for the
choice.

**The guard.** For a class: refused when the class is made, with the line and what to
write instead. For a hand-written kernel, nothing.

Described in [Programs written as classes, making and calling](classes.md#making-and-calling).

### A stream that is drained but never ended

**The hazard.** A mover that drains a stream compute fills stops when it has taken as
many pages as the end's count says. A stream no line ends has no count.

**The rule.** End the stream: `.end()` on compute's line, or a `with` block, which is
exactly an `.end()` at that place.

**The guard.** Refused at its line when the class is made (a drain of a stream that
neither `.end()` nor a `with` ends above it, and `.end()` twice).

Described in [Programs written as classes, a stream compute fills and a mover drains](classes.md#a-stream-compute-fills-and-a-mover-drains).

### Sets of a handler on one ring, with blocks that do not divide each other's

**The hazard.** A handler's sets compiled on their first use (`km.ANY`) run on the
template's buffers. Several sets with different blocks then use one ring, and its
position carries over from one command to the next. A block is one item, and an item
never wraps the ring's end (`km::item_fits`). If a mover's earlier commands left the
position at a place that is not a multiple of the next set's block, and the block
does not fit before the end, the mover gives up (`move_offer`) and the command misses
its deadline. Nothing is written wrong.

**The rule.** The pages each half takes in a command are a multiple of the largest
block among the sets in use. For example, a command's tiles split by its pattern into
blocks whose count per half keeps every half's position at a multiple of that block.
Or the sets that differ that way are listed at the make, each with buffers of its own.

**The guard.** At the first submit a set is refused when its pages exceed the load's or
do not divide them (so the ring holds whole blocks). The position across commands is
not checked: the rule is the programmer's. The gate's sets (blocks 2 to 16, 64 tiles
a core, `AAAB`) keep every half at a multiple of 16 pages.

Described in [Programs written as classes, a set compiled on its first use](classes.md#a-set-compiled-on-its-first-use).

## Tile formats and the matrix unit

### A compute loop that does not check the stop word

**The hazard.** A core whose compute unit (the unpacker, the vector unit, the packer) is
still working when its CPUs are stopped is not brought back by a core reset. A kernel
whose three compute CPUs looped over unpack, the vector unit and pack with no check of
the stop word ran the packer's output over the core's low L1, the runtime's flags
included, and the next open failed with `failed to initialize FW`. A compute kernel built
against the wrong buffer formats left the same state. Each time the card needed a reset.

**The rule.** Every loop that drives the units checks the stop word, and a kernel is built
against the formats of the buffers it reads and writes.

**The guard.** Nothing prevents it. After a `km.GridTimeout` the session does not close
the device, and the runtime resets the cores at the process's exit, which clears a stuck
CPU and not a unit left running.

Described in [how reset works, a core stuck inside a compute unit](../concepts/reset.md) and
[recovery](../guides/recovery.md#when-a-card-reset-is-still-the-only-way).

### A value one compute thread decides, steering a branch all three take

**The hazard.** `Compute::reserve_back` is decided by pack alone: on unpack and math it
returns true at once. A kernel that keeps its result in the flag its loops and branches
test (`ok = compute.reserve_back(...)`, then `if (ok)` around a section) lets pack leave
the body at a give-up while unpack and math go on issuing sections that only pack's
`tile_regs_release` frees. They stall inside the compute unit, where no wait reads the
abort word: the dispatcher cannot give the command up, its core reset does not clear
the unit, and the run ends with `km.GridTimeout` and a card reset (2026-11-16, a class's
products, before its compiler kept the reserve pack's own).

**The rule.** A value only one compute thread knows guards only that thread's calls.
Pack's reserve is held in a flag of its own (`room`) that guards `pack_reconfig_l1_acc`,
`pack_tile` and `cb_push_back` and nothing else; `tile_regs_acquire`, `commit`, `wait`
and `release`, the unpack and math calls and every loop head stay on all three threads,
steered only by what all three see alike (`wait_front`, which unpack tells the others).
The dense project's K split (`projects/dense/kernels/split/compute.cpp`) is the pattern:
`const bool room = compute.reserve_back(...)`, `room &&` on its packs, the push under
`ok && room`.

**The guard.** A class's generated compute follows the rule at every reserve (a test
checks the generated text: the products, the field reduce, a write stream's push of a
copy, of a tile function's result and of a body's tile, a tile or a sum written to a
page, a drained stream and its end). A body on compute that takes a written page gets
its word as pack's (the page's address, 0 once pack's reserve gave up; the address on
unpack and math): the class compiler refuses a body where that word steers a section,
a copy, a product or a way out (`return`, `break`), and lets it guard `pack_tile`. A
body that packs without testing the word packs into a buffer it holds no room in: the
three threads stay in step, and the pages are the given-up command's, cleared at its
end. A hand kernel: nothing.

Described in [Tensix runtime, giving up one command](runtime.md#giving-up-one-command).

### A merged state read before pack wrote it

**The hazard.** `km.reduce(self.state, ..., combine=MERGE)` runs the MERGE body once for
each arrival, and each run reads the state the run before wrote in place. The state is
written by pack and read by unpack, which share no variable and run apart: unpack can
read the state of a merge pack has not yet packed, and the root's body after the merge
can read it before the last merge. You see a state that misses an arrival, and no error.

**The rule.** The MERGE body hands a token from its pack to its unpack after it writes
the state (an own field `km.own(push="pack")`: pack pushes a page after its packs,
unpack waits for it and frees it before it reads the state), as the attention pass's
bodies do; the body after the merge waits for the last one.

**The guard.** Nothing: the build runs the body as written. A sender's push of the
state for mover B comes after the body's packs on pack itself.

Described in [Programs written as classes, a state merged, not
added](classes.md#a-state-merged-not-added).

### A vector word with a destination register of 12 or more

**The hazard.** While `LaneConfig`'s back door is open, any vector word but `SFPCONFIG`
and `SFPLOADMACRO` with a `VD` of 12 or more stores an instruction template and runs
nothing. A clamp against LReg 12 gave NaN at -127.5 and 1.76e38 at -128 on the card.

**The rule.** Only the init's template stores have a `VD` of 12 or more.

**The guard.** Refused at `emit`: the `back-door` check (and `macro-templates`, which
checks that each template a macro runs is the last word of the init to write it) refuse
the header. A hand-written body has nothing.

Described in [Tile functions, what is refused](tile.md#what-is-refused-and-why) and
[checks](tile.md#checks).

### A copy from buffers of two formats

**The hazard.** `copy_init` keeps the unpacker's data format it finds, so a copy from a
buffer of another format is read in the format the unpacker already holds.

**The rule.** A body that copies from buffers of two formats sets the unpacker's format
before each copy (`reconfig_data_format_srca`), and one that packs two formats sets the
packer's.

**The guard.** For a class: it reconfigures the unpack and the pack before each copy,
product and pack when it mixes tile formats. For a hand-written body, nothing.

Described in [Programs written as classes, a body's own buffer](classes.md#a-bodys-own-buffer) and
[products](classes.md#products).

### Buffers of the two exponent families in one kernel

**The hazard.** The unpacker and the packer hold one exponent width for a kernel's
buffers: the B family's 8 bits (bfloat16, bfp8_b, bfp4_b, bfp2_b) or the A family's 5
(float16, bfp8, bfp4, bfp2). A kernel that reads or writes both reads one of them with
the other's exponent: its values are wrong, with no error.

**The rule.** A kernel's buffers are of one family; float32 and the integer formats go
with either.

**The guard.** Refused when the program is made: `km.Program` names one buffer of each
family ("buffer 2 (bfp8) is of the A family ... and buffer 0 (bfloat16) of the B
family"); a class's product of the two families is refused at its line ("its operands
are of one exponent family").

Described in [Tile formats, the exponent families](formats.md#the-exponent-families).

### The A family with an fp32 Dst

**The hazard.** With `fp32_dest_acc_en`, every A-family datum (float16, bfp8, bfp4, bfp2)
arrives in `Dst` 2^-15 too large, through SrcA or unpacked straight to `Dst` (zero reads
2^-15, infinity 65536); an fp32 `Dst` packed to float16 is garbage, and a pack to bfp8 is
all zeros. Even with a 16-bit `Dst`, float16's subnormals become 0, -0 becomes +0, and a
host-encoded A row whose exponent saturated reads as NaN.

**The rule.** A kernel with an A-family buffer computes in a 16-bit `Dst`
(`fp32_dest_acc_en` off), within float16's normal range.

**The guard.** Refused when the program is made ("buffer 3 (bfp4) ... 16-bit Dst"). A
class of A buffers is built with a 16-bit `Dst`, refuses a float32 tile unpacked to
`Dst`, and its tile functions (which run on an fp32 `Dst`) refuse A tiles. The range
is the programmer's.

Described in [Tile formats, the exponent families](formats.md#the-exponent-families).

### A host-encoded block-float tensor compared with the card's packing

**The hazard.** The host's block-float encode (`session.tensor(..., "bfp8_b", "tile")`,
tt-metal's) rounds each datum once, half to even, to its row's shared exponent; the
card's packer rounds twice, ties away from zero. The same values encoded on the host
and packed on the card differ in the last bit of many datums (235, 377 and 244 of 1024
for bfp8_b, bfp4_b and bfp2_b in one test tile) and sometimes in a row's exponent. A
check that compares the two byte for byte fails, and one that compares values against
a host encode sees an error of the format's last bit.

**The rule.** Compare block-float values with a tolerance of the format's last bit, or
make both sides on one side (both packed on the card, or both encoded on the host).

**The guard.** Nothing.

Described in [Tile formats, the host encode and the card's](formats.md#the-host-encode-and-the-cards).

### A turned product's weight written without transposed tiles

**The hazard.** A product marked to put its right operand in SrcB (`x @ km.srcb(w)`)
reads W's tiles as the transposes of W's: a W tensor written plainly gives a wrong
product, and a W stored transposed under an unmarked product too.

**The rule.** The tensor a turned product's W is read from is written with
`session.tensor(..., transposed_tiles=True)`, and only such a product reads it.

**The guard.** Refused at the build: the class names the product's line and the tensor
("the product is turned and tensor w is not stored with its tiles transposed", or the
plain one's "is stored"), from the example tensors the class is made with; and refused
at each submit, from the command's tensor at that parameter ("tensor w is not stored
with its tiles transposed, and ... reads it as W of a turned product"). A W a
command reaches other than as a tensor argument is the programmer's.

Described in [Tile formats, placement](formats.md#placement).

### An A-family tile function's values at the 16-bit Dst's edges

**The hazard.** A tile function of the A family keeps its products' sums and its result
in a 16-bit `Dst`, truncated to 11 significant bits: sums above 65,504 read as
themselves up to 131,008 and saturate there (never infinity), values under 2^-14 read
0 (an operand under 2^-14 is 0 from its unpack), and a result above 65,504 packed to an
fp16 tensor reads as infinity or NaN on the host.

**The rule.** Scale an A-family function's operands so its products' sums and its
result lie between 2^-14 and 65,504, or give its result a float32 tensor, or compute
it in the B family.

**The guard.** Nothing.

Described in [Tile formats, the A family in a tile function](formats.md#the-a-family-in-a-tile-function).

### A tile function's sqrt, reciprocal and division outside their domain

**The hazard.** `km.sqrt(x)` is `x * rsqrt(x)` and `km.reciprocal(y)` the unit's
estimate refined by two Newton steps, neither guarded: `km.sqrt` of a negative value,
`km.rsqrt` of 0 or a negative value and `km.reciprocal(0)` (so `x / 0`) give what the
words compute, not the IEEE answer (tt-metal's `sqrt_tile` and `div_binary_tile` test
those cases; the unit program does not). `km.sqrt(0)` is 0.

**The rule.** Give `km.sqrt` values of 0 or more and `km.rsqrt`, `km.reciprocal` and a
divisor values away from 0, or write the guard in the expression (`km.maximum(x, 0.0)`),
or use the inline body of the library's call.

**The guard.** Nothing.

Described in [Tile functions, what a tile function may contain](tile.md#what-a-tile-function-may-contain).

### A float word given a float's bits

**The hazard.** A `scale: float` parameter takes the number and packs its float32 bits
itself. A caller that still passes the bits, as an `int` word took them
(`bits(2.0)`, 1073741824), gives the number 1073741824.0: the command runs and every
value it scales is wrong. A float word that is past float32's range is refused; one
that float32 rounds (0.1) is the rounded value, as a float32 tensor's would be.

**The rule.** Pass a float word the number (`Steps.scale(x, y, 2.0, 2)`); keep bits
for an `int` word whose body reads them.

**The guard.** A value that is not a number, or past float32's range, is refused at
submit, naming the parameter. An int is a number and is not refused.

Described in [Classes, the command words](classes.md#making-and-calling).

## X280 threads

### A thread that never polls the stop flag

**The hazard.** The runtime's stop is a word the cores must poll. A core that never polls
it never returns, its cluster stays stuck, and only a reset of the whole card brings it
back: the host's wait raises `ThreadFailed` (`stuck`) and the next session needs the
reset.

**The rule.** Every loop in a thread terminates: bound it by a count, or poll
`kmx_stop_requested()`. Say where the thread is with `kmx_progress(where)`.

**The guard.** The library's own waits (`kmx_parallel`, `kmx_join`, `kmx_parallel_all`,
`kmx_serve`) end at the thread's deadline, and the stop flag and the deadline are seen
inside library calls. A core stuck outside them: nothing.

Described in [X280 threads, writing one](../threads.md#writing-one) and
[limits](../threads.md#limits).

### A misaligned access in thread code

**The hazard.** A 64-bit load or store into a region at an offset that is not a multiple
of 8, or a 32-bit one not a multiple of 4, is not emulated: the runtime's trap handler
halts the core, and one such load was seen to stall the core with no trap at all.

**The rule.** Use offsets that are multiples of 8 for 64-bit accesses and of 4 for
32-bit ones. `kmx_region` returns 64-byte aligned memory.

**The guard.** Nothing.

Described in [X280 threads, limits](../threads.md#limits).

### A region read or written without the hand-over

**The hazard.** The cluster reads its DRAM bank through its cache, and a NoC write does
not pass through that cache. A thread that reads what the grid or the host wrote without
dropping its cached lines reads stale ones; a dirty line left in the cache and written
back later lands over bytes pushed or fetched since (one once failed the next load's
bank check).

**The rule.** `kmx_invalidate(id)` before the thread reads what the grid or the host
wrote, `kmx_flush(id)` after it wrote what they will read. A push lands on a range that
holds no dirty line, and its receiver invalidates the range before reading it.
`kmx_fetch_clean` is for a range none of whose lines is in the cluster's cache.

**The guard.** The library flushes every region when `kmx_main` returns. `kmx_fetch` drops
the range's lines before the copy and `kmx_store` writes them back first (a sweep, or
`kmx_evict` from 160 KiB on, about 20 µs whatever the range). Otherwise, nothing.

Described in [X280 threads, what a thread can touch](../threads.md#what-a-thread-can-touch) and
[reading and writing another bank](threads.md#reading-and-writing-another-bank).

### A call that belongs to core 0, made on another core

**The hazard.** `kmx_spawn`, `kmx_join`, `kmx_parallel`, `kmx_parallel_all`, `kmx_serve`,
`kmx_port_run` and the ring calls are core 0's. Called from cores 1 to 3, or inside
`kmx_parallel`, they command the library error `parallel` and return `KMX_STOPPED`.

**The rule.** Only core 0 calls them.

**The guard.** The library refuses them with that error.

Described in [X280 threads reference, C](threads.md#c-include-kmxh).

### Events a thread submitted and did not wait for

**The hazard.** The event of a command `kmx_program` submitted and did not wait for is
kept for a later `kmx_poll` or `kmx_wait`, but only the newest 32 are kept; older ones
are dropped.

**The rule.** A program that needs every event takes them inside the call that submitted
the commands (`kmx_wait` each, or `kmx_poll` until `KMX_NONE`).

**The guard.** A stash of 32 events (`KMX_STASH`); beyond it, nothing.

Described in [X280 threads reference, commands to a thread](threads.md#commands-to-a-thread).

### A tell taken for a message

**The hazard.** A tell overwrites the line it lands in: a side that looks less often than
the other tells sees only the newest, and the tells in between are lost. Tells to a core
that is not running a command that looks for them stay in the line, and the core's next
look sees one if it is newer.

**The rule.** Tell what a newer tell replaces, and use ask and answer, paired by sequence
number, for what must not be lost. One kernel of a Tensix core sends on its port, never
both; an X280 core sends only to the cores it serves; tell a core only while it runs a
command that looks for it.

**The guard.** An ask and its answer are paired by sequence number, one ask outstanding
at a time. For the rest, nothing.

Described in [Ports, X280 ports](ports.md#x280-ports) and
[X280 threads, X280 ports](../threads.md#x280-ports).

### Globals and libc in a thread's image

**The hazard.** The image has no `.bss` and links no libc. A global without a non-zero
initialiser, or a struct assignment or a large zero initialiser that the compiler turns
into a `memset` call, does not link.

**The rule.** Give each global a non-zero initialiser and
`__attribute__((section(".data")))`, and write no code that needs libc.

**The guard.** The build refuses: the image does not link.

Described in [X280 threads reference, the image](threads.md#the-image) and
[writing one](../threads.md#writing-one).

### A class's command from a thread on a tensor other than the make's

**The hazard.** A handler's function in a class's thread header writes the address of
the tensor the class was made with into each tensor parameter's word. A thread that
means the command for another tensor gets the make's tensor: it reads or writes the
wrong pages, and nothing fails.

**The rule.** For another tensor, set the word after the function
(`command->a<n>`, `n` the place `<NAME>_<HANDLER>_<PARAMETER>` gives), or make the
class with that tensor.

**The guard.** Nothing.

Described in [classes, X280 threads](classes.md#x280-threads).

### An X280 clock above the default under matrix load

**The hazard.** At 1,750 MHz the card lowers its core voltage when the Tensix grid is
under full matrix load, and the X280 side loses loads: the dispatcher trapped in 9 of 18
such runs.

**The rule.** Keep the session's clock at its default of 1,350 MHz, or one of 1,000,
1,250 and 1,500.

**The guard.** A switch, `pll_mhz` of `Session`: 1,750 is an opt-in and never the default,
and safe use is the caller's claim. What the default gives up: a host-fed command at
depth 16 costs 1.228 µs at 1,000 MHz and 0.726 at 1,750.

Described in [Host API, `Session`](host.md#session) and the README's
[sizes](../../README.md#limits).

## Islands and links

### A page read with tt-metal's own call

**The hazard.** A page read through `noc_async_read_page` reads its own island's DRAM
only. A kernel that calls it, or builds a tensor's address itself
(`InterleavedAddrGen`), works on one card and reads the wrong pages over islands.

**The rule.** A kernel reads and writes a tensor's pages through `km::read_pages` and
`km::write_pages` (and the byte calls), with the tensor as a `km::Tensor`.

**The guard.** Nothing: a program loaded on one island is built without the tensor table,
and each of Komorebi's calls compiles to exactly the tt-metal call it replaces there.

Described in [Islands, a kernel's page reads and writes](islands.md#a-kernels-page-reads-and-writes).

### A put that is never waited for

**The hazard.** `km::put_bytes` of a page another island holds appends a record to the
CPU's batch, which is sent only when the next record does not fit, when a record is for
another island, or at `km::put_wait`. A batch left by an earlier command is dropped. Two
puts to the same bytes from different cores have no order.

**The rule.** End each command that puts with `km::put_wait`, or use `km::write_bytes`,
which waits. Keep `source` until its write is done when the page is this island's. Two
cores never put to the same bytes.

**The guard.** Nothing for the batch or the order. A range that leaves its page, or an
offset or a byte count that is not a multiple of 4, or a `source` whose low four bits
differ from the offset's, is refused (`bad_transfer` at `ask`).

Described in [Islands, a kernel's page reads and writes](islands.md#a-kernels-page-reads-and-writes).

### A put given up after it reserved ring slots

**The hazard.** A putting core that gives up after it reserved slots in a page CPU's ring
leaves them unfilled. The ring sends in order, so it stays stopped there: later puts
through it do not go.

**The rule.** Treat a give-up of a put as the end of that load of the program: the
stopped ring is cleared by the cores' recovery or by the next load.

**The guard.** Nothing at the call.

Described in [Islands, pages across a link](islands.md#pages-across-a-link).

### A page call across given up with its lines still out

**The hazard.** A get or a put across a link names its pages in a line that a link core
serves later, from the asking core's L1 (a put) or into it (a get), and counts at the
CPU's page port. A command given up at its deadline while such a line is out leaves it
behind. The page's buffer is then cleared and reused while the link core may still read
or fill it, so another island's tensor or this core's next page may get wrong data.
The port's count and the put's confirmation words are left as the given-up wait had
them. A line number taken from the link core and never written stops every later line
through that link core.

**The rule.** Give a command that crosses a deadline its lines' service can meet. Treat a
given-up command that crossed as a fault of the load, not as an ordinary miss.

**The guard.** Under `guard` (the default), the given-up command's end on each reader and
writer first waits, bounded by the wait limit, until every line that CPU sent was
served. The pages stay in their buffers meanwhile. Then the page port's count and
want and the put's confirmation words are reset for the next command. A line not
served in time stops the core, so its buffer is never reused, and the row names the
command, the island, the first page and the pages left. A line number taken and never
written is named too ("took line N of the link core toward island I and never wrote
it"). The stop then ends: its EXIT, waiting for cores that ended by their own give-up,
is left done, and `run.stop` raises naming them, not a time-out. With `guard=False`, the
six ask waits watch only the stop flag, not a deadline: a command whose line is never
served stays in its wait until `run.stop`. Nothing is waited for or reset after it, and
the load's log says once that commands are not given up. Such a load is warned, not
refused: guard off is the programmer's choice on islands as on one card.

Described in [Islands, pages across a link](islands.md#pages-across-a-link).

### A card opened alone in a process that opened a mesh

**The hazard.** Closing a session of several cards leaves tt-metal's context set up for
the process. A later open with other parameters in the same process, such as one card
with its default kernel buffer, re-initialises that context. The re-initialisation
covers every card the process's driver opened, not only the one asked for. Once (2026-11-16, a
one-card open of card 0 after cards 0 and 1 as a mesh), that open timed out waiting for
the Ethernet core 27-25 to become active ("try resetting the board"). The process then
crashed at exit, and both cards refused every later open, in fresh processes too, until
a board reset. A process that closes a mesh and exits has never shown it.

**The rule.** One process opens either cards as a mesh or a card alone, never both. Run
the other kind in a new process.

**The guard.** A refusal at open (`KomorebiError`, "open it in a new process"), for a
card alone after a mesh and for a mesh after a card alone, before any device call.

Described in [Islands, the session](islands.md#the-session).

## The host

### A process that leaves the card half closed

**The hazard.** The X280 runtime must be closed while the device is open. A process that
ends with it open, or closes the device first, can leave the card needing a reset, and a
process killed with SIGKILL cannot close anything.

**The rule.** Close the session, never kill it with SIGKILL; with `device=`, close the
session before the device. One process holds a card from its first open until it exits.

**The guard.** The session closes itself at the process's exit and on SIGTERM (it exits
with status 143), and the exit paths try again if a clean-up raises. A second process that
opens a card another holds gets `km.CardBusy` naming the holder, in place of an unbounded
wait. SIGKILL: nothing, the card is reset.

Described in [Host API, `Session`](host.md#session) and
[recovery](../guides/recovery.md#when-a-card-reset-is-still-the-only-way).

### A session that follows its kernels

**The hazard.** A session not given `kernel_bytes` opens its card again at its first
load when the kernels need another kernel buffer, and writes into each tensor again
what it was given. It keeps the host arrays given to `session.tensor`, `session.write`
and `tensor.write` by reference until that load, so an array changed in between is
written as it is then, and bytes written through `tensor.buffer` itself are not
written again. Where it cannot open the card again (a run already loaded, a session
on a given device, an island or mesh session), the load is refused.

**The rule.** Until the first load, do not change an array once it is given to the
session, and write tensors through `session.tensor`, `session.write` or
`tensor.write`. A session that loads a second, larger program gives `kernel_bytes`
the size the refusal names, or runs the script again: the refusal is remembered and
the next run opens at that size.

**The guard.** The load decides from a bound of the kernel config taken from the
build before tt-metal lays it out (that check logs a fatal line): per group the most
runtime argument words over its cores and the common ones, the semaphores by the
highest id, the circular buffer configs by the highest index, the packed text, and one
line of alignment; it is 16 bytes over tt-metal's size in the mock builds measured. A
bound under the buffer whose layout is over still logs that line, then reopens. The
refusals: `km.KernelSizeError` (or `km.L1BudgetError` for a
smaller buffer) naming why and the size that holds the kernels; `km.KomorebiError`
when a tensor comes back at another address. A changed array or a write through the
buffer: nothing.

Described in [Host API, the kernel buffer](host.md#the-kernel-buffer).

### A tensor or a thread's region touched while the kernels run

**The hazard.** A persistent grid never returns until the stop, and the device API waits
for it. A tensor written or read through the device API while a run is loaded waits 60
seconds and then raises a `RuntimeError`. A read through the host channel does not wait
for the cores, so it also reads an output a kernel is still writing.

**The rule.** Make every tensor before `load`. While a run is loaded, read through
`session.read` only what no kernel writes any more (an output whose command's event has
arrived). Read a thread's region (`run.thread.read`) only after `run.stop()`.

**The guard.** An L1 tensor read without the host channel while a run is loaded raises
`km.ChannelError`; `run.thread.read` before the stop raises. For an output a kernel is
still writing, nothing.

Described in [Host API, tensors](host.md#tensors),
[the host channel](host.md#the-host-channel) and
[debug a command that does not finish](../guides/debug-a-command-that-does-not-finish.md#what-to-check-first).
