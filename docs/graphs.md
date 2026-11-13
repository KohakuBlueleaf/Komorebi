---
title: Graphs
summary: The graph executor's upload format, the Python builder, the check, the executor in plain C and in Python, and the simulator.
tags:
  - reference
  - dispatch
  - graphs
---

# Graphs

A graph says what depends on what. The executor submits every node whose predecessors
are done, so independent branches on disjoint groups are in flight together without the
author ordering them. A graph is uploaded once and run with no host step inside; a
graph that repeats ends on a value computed on the card; a kept graph is run again with
new bindings without a new upload.

Terms (*command*, *handler*, *group*, *completion slot*, *pass*, *template*, *micro-op*)
are those of [micro-op streams](streams.md), whose command templates and compute
micro-ops a graph reuses unchanged.

The code is `komorebi.engines.graph` (builder, encoding, check, executor, simulator)
and `src/komorebi_native/x280/engines/km_graph_engine.c` (the executor in plain C);
the shared definitions are in `src/komorebi_native/include/komorebi/abi/km_graph.h`.

## Building a graph

```python
from komorebi.engines.graph import Graph

HANDLER_STEP, HANDLER_LEFT, HANDLER_RIGHT, HANDLER_NORM = 1, 2, 3, 4

graph = Graph(deadline_microseconds=200_000, max_runs=100)
iteration, error = graph.bindings("iteration"), graph.register()
step = graph.command(HANDLER_STEP, group=(0, 60), args=[iteration])
left = graph.command(HANDLER_LEFT, group=(0, 30), after=[step])
right = graph.command(HANDLER_RIGHT, group=(30, 30), after=[step])  # with left
norm = graph.command(HANDLER_NORM, group=(0, 60), after=[left, right], result=error)
cpu = graph.cpu(after=[norm])
cpu.addi(iteration, iteration, 1)
cpu.lt(graph.again, error, graph.const(65))  # go round again while error < 65
upload = graph.encode()
bindings = graph.binding_values(iteration=0)
```

- `Graph(deadline_microseconds=0, max_runs=1, step_limit=4096, req=1, keep=False)`: the
  deadline applies to every command (0 is the kernel's default); `max_runs` bounds the
  rounds; `step_limit` bounds the micro-ops one CPU node may run (1 to 4,096); `req` is
  the request id of the graph's events; `keep` makes it a kept graph (below).
- Nodes are numbered in the order they are made, and `after` names earlier nodes only,
  so the upload's order is always one in which predecessors come first. A node has at
  most four predecessors.
- `graph.command(handler, group=(first, count) | count=n, args=[...], after=[...], result=register,
  exclusive=False, skip_if_zero=register)`: a `COMMAND` node. `first`, `count` and up to four
  of the ten arguments may be registers, read when the node is submitted. With `count=`
  instead of `group=` the command is queued (`KM_FLAG_QUEUE`) and the allocator places it.
  `result=register` asks for the record's result word and stores it in `register` at completion.
- `graph.command(handler, cluster=c, args=[...], after=[...], result=register)`: a
  `COMMAND` node for work cluster `c`'s X280 thread (1 to 3) instead of a group, never
  exclusive; `result` gets the word its `kmx_program` returns, so a thread's step can
  feed the next grid command ([commands to a
  thread](reference/threads.md#commands-to-a-thread)). A group from registers names a
  thread too: `first` 0xFF00 + `c`, `count` 1.
- `graph.cpu(after=..., skip_if_zero=...)`: a `CPU` node; its micro-ops are added through
  it: `set`, `mov`, `add`, `addi`, `sub`, `mul`, `and_`, `shr`, `mod`, `rand`, `jmp`,
  `jlt`, `jeq`, `jnz` with labels local to the node (`label`, `forward`, `bind`), and
  `lt(d, a, b)` / `ge(d, a, b)` that set `d` to 1 or 0 (four micro-ops each).
- `graph.join(after=..., skip_if_zero=...)`: a `JOIN` node, no work.
- `skip_if_zero=register`: when the node is taken from the ready ring and `register` is 0, it is
  not run and counts as done.
- Registers: 16 for the whole graph, unsigned 32-bit, wrapping. `graph.register()`, `graph.registers(count)`,
  `graph.bindings(*names)` (registers set by name at each run), `graph.const(value)` (a register
  set to `value` at each run), `graph.again` (the register that starts another round).
- `graph.binding_values(**values)` gives the binding block of one run: the values of
  registers 0 to n - 1, with every binding and constant in it. Every binding must be
  given. Registers not in the block start at 0 at every run.
- Identical command templates are shared; a graph has at most 64 distinct ones.
- `graph.encode()` gives the upload; `Graph.decode(data)` is its inverse (registers by
  number, binding names lost) and raises `GraphFormatError` for an upload the check
  rejects.

## Running: rounds and the executor

At a run, all 16 registers are set to 0, then registers 0 to n - 1 to the binding
block. Every node waits for its predecessors; the nodes without one are put on the
ready ring in node order. The executor, called once per kernel pass, takes at most 16
nodes from the ring per step:

| Node | What the step does |
|---|---|
| skipped (`skip_if_zero` register is 0) | done |
| `JOIN` | done |
| `CPU` | runs its micro-ops from the first of its slice until the program counter reaches the slice's end; done |
| `COMMAND` | fills the command from its template and the registers, checks the group, submits it; if it cannot be submitted now (no free slot, an overlapping exclusive group, the queue full) it goes back to the tail of the ring |

A command's completion stores its result word if asked and makes the node done. A node
that is done lowers the count of each successor; a successor whose count reaches 0 goes
on the ring. The step ends after 16 nodes, when the ring is empty, when every node in
it was found busy in a row, or after a `CPU` node once 64 micro-ops ran in the step.

When every node is done the round ends. If `again` is set (not `KM_GRAPH_NONE`) and its
register is not 0, the counts are reset, the roots go on the ring and another round
starts with the registers as they are (a `KM_EVENT_GRAPH` event says so); after
`max_runs` rounds that is an error instead. Otherwise the graph ends with
`KM_EVENT_GRAPH_END`.

The result is fixed by the dependencies: a graph in which no two nodes that can run
at the same time write a register the other reads submits the same set of commands with
the same arguments, and ends with the same registers, whatever the order in which
completions arrive. Only the order of the commands and the passes depend on the timing.

## The upload

All fields are little-endian; each part is padded with zeros to a 64-byte line.

| Part | Layout |
|---|---|
| Header | 64 bytes, twelve words: `magic` 0x4B4D4752 ("KMGR"), `version` 1, `node_count`, `edge_count`, `op_count`, `command_count`, `req`, `again_register` (0xFF: one round), `max_runs`, `deadline_microseconds`, `step_limit`, `flags` (`KM_GRAPH_KEEP` 1, other bits 0); then 16 bytes ignored |
| Nodes | `node_count` times 16 bytes: `kind`, `flags`, `predecessor_count`, `skip_register` (one byte each), `arg` (16 bits: the template of a `COMMAND`, the first micro-op of a `CPU`), `op_count` (16 bits: a `CPU`'s slice length), `result_register`, a pad byte, `successor_offset`, `successor_count` (16 bits each), 2 pad bytes |
| Edges | `edge_count` 16-bit node numbers: the successor lists, node after node |
| Micro-ops | `op_count` times 16 bytes, the stream's encoding; the `CPU` slices, node after node |
| Templates | `command_count` times 80 bytes, the stream's template encoding |

Kinds: `COMMAND` 1, `CPU` 2, `JOIN` 3. 0xFF in a register field means none. `req` is
header word 6, as in a stream. A template's flag byte may hold `KM_FLAG_EXCLUSIVE`,
`KM_FLAG_RESULT` and `KM_FLAG_QUEUE`.

Limits: 256 nodes, 1,024 edges, four predecessors per node, 4,096 micro-ops, 64
templates, `step_limit` 4,096, 16 bindings.

The host writes the upload as whole lines into the kernel's upload region and sends
`KM_HANDLER_RUN` with `a2 = KM_GRAPH_RUN_UPLOAD` (`a0` offset, `a1` bytes, `a3` offset of the
binding block, `a4` its count of values); a kept graph is run again with
`a2 = KM_GRAPH_RUN_KEPT` and `a0` its `req`. The binding block is `a4` 32-bit values at a
64-byte aligned offset of the region, read once when the run starts.

**Kept graphs.** A graph whose header has `KM_GRAPH_KEEP` stays kept after it ends,
whether it ended or failed. The kernel runs uploads in place and copies nothing, so a
kept graph stays where the host wrote it, and the host does not give that range of the
upload region to another upload while the graph is kept. At most eight graphs are kept;
a ninth upload with the bit is rejected with reason 15, and nothing is evicted by
itself. `KM_HANDLER_CANCEL` with the `req` of a kept graph that is running cancels the run
(reason 10) and keeps the graph; for one that is not running it drops the graph and
frees its range, and its `KM_EVENT_GRAPH_ERROR` with reason 10 (node 0xFFFF) says so. A
graph with the bit counts as kept from its `RUN`: two uploads with the bit running at
once both count, even if one of them then fails its check.

## The check

`check(data)` (Python) and `kmg_check(upload, bytes)` (C) return `None` / 0 or the
reason and node of the error event; the first fault in this order wins.

1. Header (reason 1, node 0xFFFF): at least 64 bytes and a multiple of 64; magic and
   version; `node_count` 1 to 256; `edge_count`, `op_count`, `command_count` within their limits;
   `again_register` 0xFF or below 16; `max_runs` not 0; `step_limit` 1 to 4,096; `flags`
   within `KM_GRAPH_KEEP`; the size equals the layout's exactly.
2. Each node in order, at that node: `kind` 1 to 3 and `flags` 0 (reason 12);
   `predecessor_count` at most 4 (13); `skip_register` and `result_register` 0xFF or below 16 (4). A
   `COMMAND`: template index below `command_count` and the template valid (5); `op_count` 0,
   and a result register only with `KM_FLAG_RESULT` in the template (12). A `CPU` or
   `JOIN`: no result register (12). A `JOIN`: `arg` and `op_count` 0 (12). A `CPU`: its
   slice starts where the previous `CPU` node's ended (the first at 0), is not empty and
   ends within `op_count` (12); each of its micro-ops is a compute or jump micro-op (2),
   uses registers below 16 (4) and jumps to a micro-op of the slice or to its end (3).
   Then its successor list: `successor_offset` is where the previous node's list ended,
   `successor_offset + successor_count` within `edge_count`, and the list rises strictly from above the
   node itself to below `node_count`, no node getting a fifth predecessor (13).
3. The lists end at `edge_count` (13, node 0xFFFF); the slices end at `op_count` (12, node
   0xFFFF).
4. Each node's `predecessor_count` equals the times it appears in the lists (13, at the node).
5. Every template is valid (5, node 0xFFFF): handler 1 to 239; flags within
   `KM_FLAG_EXCLUSIVE | KM_FLAG_RESULT | KM_FLAG_QUEUE`; at most four patches, each of an argument
   below 10 from a register below 16; `first_register` and `count_register` 0xFF or below 16.

The compute and jump micro-ops are `SET`, `MOV`, `ADD`, `ADDI`, `SUB`, `MUL`, `AND`,
`SHR`, `MOD`, `RAND`, `JMP`, `JLT`, `JEQ`, `JNZ`, with the stream's meanings; a jump to
the slice's end finishes the node.

## Events and errors

| Kind | Posted | Auxiliary | `ticks` word |
|---|---|---|---|
| `KM_EVENT_GRAPH` 7 | a round ended and another starts | rounds done | - |
| `KM_EVENT_GRAPH_END` 8 | the graph ended | register 0, low 24 bits | register 1 |
| `KM_EVENT_GRAPH_ERROR` 9 | any error; nothing follows it | reason, then the node in the 16 bits above | - |

| Reason | Meaning | Node |
|---|---|---|
| 1 | bad header or size; more than 16 bindings | 0xFFFF |
| 2 | a micro-op a `CPU` slice may not hold | the node |
| 3 | a jump outside the slice | the node |
| 4 | a register number of 16 or more | the node |
| 5 | bad template or template index | the node using it; 0xFFFF for one no node uses |
| 6 | a `CPU` node ran `step_limit` micro-ops without reaching its end | the node |
| 7 | `MOD` by zero | the node |
| 8 | a command's deadline passed | the node |
| 9 | a queued command whose count is 0 or above the grid | the node |
| 10 | cancelled | 0xFFFF |
| 11 | a command whose group is empty or leaves the grid (tested without a sum) and names no work cluster's thread, or names one with `exclusive` | the node |
| 12 | bad node: kind, flags, a field its kind does not use, a slice | the node, or 0xFFFF |
| 13 | bad successor list or predecessor count | the node, or 0xFFFF |
| 14 | `again` still set after `max_runs` rounds | 0xFFFF |
| 15 | `KM_GRAPH_KEEP` while eight graphs are kept | 0xFFFF |

## The C engine

`km_graph_engine.c` is freestanding: no library calls, no floating point, no static
data, every loop capped by a limit above; all state is in a `kmg_state_t` the caller
gives. The same file is compiled into the dispatcher and, on the host, into the
library `komorebi.engines.graph_native` loads with ctypes.

| Function | Contract |
|---|---|
| `kmg_check(upload, bytes)` | 0 or `reason \| node << 8` |
| `kmg_start(state, upload, bindings, n, core_count)` | starts a checked upload, which stays in place and unchanged while it runs; 0, or 1 for more than 16 bindings |
| `kmg_step(state, submit, ctx)` | one step; returns `KMG_RUNNING`, `KMG_ROUND`, `KMG_END` or `KMG_ERROR` |
| `kmg_complete(state, node, result)` | a command completed; 1 (and nothing changes) if the graph is not running or the node has no command in flight |
| `kmg_retry(state, node)` | the node's command was submitted and then rejected for now: the node goes back to the ready ring's tail; 1 as for `kmg_complete` |
| `kmg_stop(state, reason, node)` | ends a running graph from outside (deadline, cancel) |

`submit(ctx, node, &command)` gets the command with `sequence`, the landing words and the slot
bits 0, its group resolved (a queued command keeps `KM_FLAG_QUEUE` and `first` 0), and
returns `KMG_SUBMITTED` or `KMG_BUSY`. The first 20 words of the state are the registers,
the round, the phase, the error reason and the error node.

## In the dispatcher

Core 1 of cluster 0 runs up to two graphs beside the streams, one `kmg_step`
each per pass of its loop, and sends their commands into the engine's ring like a
stream's; core 0 submits them in the engine's turn and answers each with its `KM_EVENT_DONE`
(the result word in its ticks) or `KM_EVENT_TIMEOUT`. A queued command goes into the
kernel's job queue with the graph's tag; a graph that ends with an error drops its jobs
still waiting there. Before it hands a `RUN` to core 1,
core 0 answers at once:

| Case | Event |
|---|---|
| `a2` above 2 | `KM_EVENT_REJECTED`, reason 1 |
| any command after EXIT | `KM_EVENT_REJECTED`, reason 4 |
| two graphs running | `KM_EVENT_REJECTED`, reason 6 |
| `a2 = 2` and no kept graph that is not running has `req` | `KM_EVENT_REJECTED`, reason 7 |
| `a2 = 1`: a range that is not whole lines inside the region, or that overlaps a running stream or a running or kept graph; a `req` one of those has | `KM_EVENT_GRAPH_ERROR` reason 1, node 0xFFFF, with the header's `req` (0 if the header is outside the region) |

Core 1 then runs the check, the kept table, the bindings (a block outside the region is
reason 1), in that order; the first fault ends the graph with its error event. A
stream's `RUN` whose range or `req` a running or kept graph has ends with
`KM_EVENT_STREAM_ERROR` reason 1. EXIT waits until no graph runs. A command of a graph that
passes its deadline ends the graph with reason 8 and the kernel with
`KMD_ERROR_DEADLINE`, as for a stream.

## The simulator

```python
from komorebi.engines.graph import simulate
from komorebi.engines.graph_native import NativeExecutor

done = graph.simulate(
    {"iteration": 0}, delay=2, result=lambda command: command.args[0] + command.count
)
same = simulate(upload, bindings=bindings, delay=2, engine=NativeExecutor())
```

`simulate(data, bindings=(), core_count=119, delay=0, result=None,
deadline_passes=None, cancel_pass=None, max_passes=100_000, kept=0, engine=None)` runs
the check, rejects a kept graph when `kept` graphs are kept already and that is eight,
then runs the graph against a model of the kernel. Each pass: completions due are
swept in slot order; a command still busy `deadline_passes` passes after its submit ends
the graph (the lowest slot's); a cancel in `cancel_pass`; one executor step. Commands
take the lowest free of 16 slots; an exclusive command waits while a busy group overlaps
it; a queued command is submitted when a slot is free and the allocator finds room, else
it is busy (the kernel's queue accepts it earlier; only the timing differs). `delay`
and `result` are as for streams. `engine` is the executor to drive: `Executor()` in
Python by default, or `NativeExecutor()` for the C engine.

It returns `GraphSimResult(commands, events, status, error, registers, runs, passes,
stopped, kept)` (`kept`: the graph stays kept after this run): every command as `GraphCommand(node, run, sequence, slot, handler, first, count, flags,
args, pass_index, queued)`; events as `Event(kind, req, auxiliary, extra, pass_index)`;
`status` `"ended"`, `"error"` or `"stuck"`.

## Host-fed

```python
done = run.run_hostfed(graph, {"iteration": 0}, timeout=10.0)  # run: a komorebi.run.Run
```

`Run.run_hostfed(graph, bindings=None, timeout=10.0)` runs the same C engine on the
host and submits its commands through the run's host ring, so a graph gives the same set
of commands and the same registers host-fed and in the dispatcher. After each step
the host waits for the first command in flight to finish (`Run.wait_any`), then hands
every completion seen to the engine, with the result word of a command that asked for
one (`Run.submit(..., result=True)`). A queued command goes to the kernel by its count
(`queued=True`), whose allocator picks its group; its `first` is None here. A command's
timeout ends the graph with reason 8; after any other error the commands still in
flight are drained. It returns `HostfedResult(status, error, registers, runs, commands,
events)`: `commands` are
`HostfedCommand(node, run, handle, handler, first, count, flags, args, queued)`, `events`
the graph events the kernel would have posted. A graph that does not end within
`timeout` seconds raises `WaitTimeout`.

## Reference graphs

`experiments/graph_checks/graphs/` holds 40 named graphs as `.bin` uploads with a
`.json` beside each: the bindings, the simulation parameters and the answer (the check,
status, error, final registers, rounds, the set of commands submitted and the events). They
are for another implementation, such as the dispatcher's, to replay; the set and
the registers are binding, the order is not. `graph_reference.py` in that folder
rewrites them and with `--check` compares both executors with them. On a card,
`tests/hardware/dispatcher/graph_card.py graphs` replays every one that ends by
itself in the dispatcher against the simulator on the card's grid.
