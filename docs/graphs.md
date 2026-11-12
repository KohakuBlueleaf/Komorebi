---
title: Graphs
summary: The graph executor's upload format, the Python builder, the check, the executor in plain C and in Python, and the simulator.
tags:
  - reference
  - dispatch
  - graphs
---

# Graphs

A graph says what depends on what. The executor issues every node whose predecessors
are done, so independent branches on disjoint teams are in flight together without the
author ordering them. A graph is uploaded once and run with no host step inside; a
graph that repeats ends on a value computed on the card; a kept graph is run again with
new bindings without a new upload.

Terms (*record*, *segment*, *team*, *completion slot*, *pass*, *template*, *micro-op*)
are those of [micro-op streams](streams.md), whose record templates and compute
micro-ops a graph reuses unchanged.

The code is `komorebi.engines.graph` (builder, encoding, check, executor, simulator)
and `src/komorebi_native/x280/engines/km_graph_engine.c` (the executor in plain C);
the shared definitions are in `src/komorebi_native/include/komorebi/abi/km_graph.h`.

## Building a graph

```python
from komorebi.engines.graph import Graph

SEG_STEP, SEG_LEFT, SEG_RIGHT, SEG_NORM = 1, 2, 3, 4

g = Graph(deadline_us=200_000, max_runs=100)
it, err = g.bindings("it"), g.reg()
a = g.record(SEG_STEP, team=(0, 60), args=[it])
b = g.record(SEG_LEFT, team=(0, 30), after=[a])
c = g.record(SEG_RIGHT, team=(30, 30), after=[a])  # in flight together with b
d = g.record(SEG_NORM, team=(0, 60), after=[b, c], result=err)
e = g.cpu(after=[d])
e.addi(it, it, 1)
e.lt(g.again, err, g.const(65))  # go round again while err < 65
upload = g.encode()
bindings = g.binding_values(it=0)
```

- `Graph(deadline_us=0, max_runs=1, step_limit=4096, req=1, keep=False)`: the
  deadline applies to every record (0 is the kernel's default); `max_runs` bounds the
  rounds; `step_limit` bounds the micro-ops one CPU node may run (1 to 4,096); `req` is
  the request id of the graph's events; `keep` makes it a kept graph (below).
- Nodes are numbered in the order they are made, and `after` names earlier nodes only,
  so the upload's order is always one in which predecessors come first. A node has at
  most four predecessors.
- `g.record(seg, team=(first, count) | count=n, args=[...], after=[...], result=reg,
  exclusive=False, skip_if_zero=reg)`: a `RECORD` node. `first`, `count` and up to four
  of the ten arguments may be registers, read when the node is issued. With `count=`
  instead of `team=` the record is queued (`KM_F_QUEUE`) and the allocator places it.
  `result=reg` asks for the record's result word and stores it in `reg` at completion.
- `g.cpu(after=..., skip_if_zero=...)`: a `CPU` node; its micro-ops are added through
  it: `set`, `mov`, `add`, `addi`, `sub`, `mul`, `and_`, `shr`, `mod`, `rand`, `jmp`,
  `jlt`, `jeq`, `jnz` with labels local to the node (`label`, `forward`, `bind`), and
  `lt(d, a, b)` / `ge(d, a, b)` that set `d` to 1 or 0 (four micro-ops each).
- `g.join(after=..., skip_if_zero=...)`: a `JOIN` node, no work.
- `skip_if_zero=reg`: when the node is taken from the ready ring and `reg` is 0, it is
  not run and counts as done.
- Registers: 16 for the whole graph, unsigned 32-bit, wrapping. `g.reg()`, `g.regs(n)`,
  `g.bindings(*names)` (registers set by name at each run), `g.const(v)` (a register
  set to `v` at each run), `g.again` (the register that starts another round).
- `g.binding_values(**values)` gives the binding block of one run: the values of
  registers 0 to n - 1, with every binding and constant in it. Every binding must be
  given. Registers not in the block start at 0 at every run.
- Identical record templates are shared; a graph has at most 64 distinct ones.
- `g.encode()` gives the upload; `Graph.decode(bytes)` is its inverse (registers by
  number, binding names lost) and raises `GraphFormatError` for an upload the check
  refuses.

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
| `RECORD` | fills the record from its template and the registers, checks the team, issues it; if it cannot be issued now (no free slot, an overlapping exclusive team, the queue full) it goes back to the tail of the ring |

A record's completion stores its result word if asked and makes the node done. A node
that is done lowers the count of each successor; a successor whose count reaches 0 goes
on the ring. The step ends after 16 nodes, when the ring is empty, when every node in
it was found busy in a row, or after a `CPU` node once 64 micro-ops ran in the step.

When every node is done the round ends. If `again` is set (not `KM_GRAPH_NONE`) and its
register is not 0, the counts are reset, the roots go on the ring and another round
starts with the registers as they are (a `KM_EV_GRAPH` event says so); after
`max_runs` rounds that is an error instead. Otherwise the graph ends with
`KM_EV_GRAPH_END`.

The result is fixed by the dependencies: a graph in which no two nodes that can run
at the same time write a register the other reads issues the same set of records with
the same arguments, and ends with the same registers, whatever the order in which
completions arrive. Only the order of the records and the passes depend on the timing.

## The upload

All fields are little-endian; each part is padded with zeros to a 64-byte line.

| Part | Layout |
|---|---|
| Header | 64 bytes, twelve words: `magic` 0x4B4D4752 ("KMGR"), `version` 1, `n_nodes`, `n_edges`, `n_ops`, `n_records`, `req`, `again_reg` (0xFF: one round), `max_runs`, `deadline_us`, `step_limit`, `flags` (`KM_GRAPH_KEEP` 1, other bits 0); then 16 bytes ignored |
| Nodes | `n_nodes` times 16 bytes: `kind`, `flags`, `n_pred`, `skip_reg` (one byte each), `arg` (16 bits: the template of a `RECORD`, the first micro-op of a `CPU`), `n_ops` (16 bits: a `CPU`'s slice length), `result_reg`, a pad byte, `succ_off`, `succ_n` (16 bits each), 2 pad bytes |
| Edges | `n_edges` 16-bit node numbers: the successor lists, node after node |
| Micro-ops | `n_ops` times 16 bytes, the stream's encoding; the `CPU` slices, node after node |
| Templates | `n_records` times 80 bytes, the stream's template encoding |

Kinds: `RECORD` 1, `CPU` 2, `JOIN` 3. 0xFF in a register field means none. `req` is
header word 6, as in a stream. A template's flag byte may hold `KM_F_EXCL`,
`KM_F_RESULT` and `KM_F_QUEUE`.

Limits: 256 nodes, 1,024 edges, four predecessors per node, 4,096 micro-ops, 64
templates, `step_limit` 4,096, 16 bindings.

The host writes the upload as whole lines into the kernel's upload region and sends
`KM_SEG_RUN` with `a2 = KM_GRUN_UPLOAD` (`a0` offset, `a1` bytes, `a3` offset of the
binding block, `a4` its count of values); a kept graph is run again with
`a2 = KM_GRUN_KEPT` and `a0` its `req`.

**Kept graphs.** A graph whose header has `KM_GRAPH_KEEP` stays kept after it ends,
whether it ended or failed. The kernel runs uploads in place and copies nothing, so a
kept graph stays where the host wrote it, and the host does not give that range of the
upload region to another upload while the graph is kept. At most eight graphs are kept;
a ninth upload with the bit is refused with reason 15, and nothing is evicted by
itself. `KM_SEG_CANCEL` with the `req` of a kept graph that is running cancels the run
(reason 10) and keeps the graph; for one that is not running it drops the graph and
frees its range.

## The check

`check(upload)` (Python) and `kmg_check(upload, bytes)` (C) return `None` / 0 or the
reason and node of the error event; the first fault in this order wins.

1. Header (reason 1, node 0xFFFF): at least 64 bytes and a multiple of 64; magic and
   version; `n_nodes` 1 to 256; `n_edges`, `n_ops`, `n_records` within their limits;
   `again_reg` 0xFF or below 16; `max_runs` not 0; `step_limit` 1 to 4,096; `flags`
   within `KM_GRAPH_KEEP`; the size equals the layout's exactly.
2. Each node in order, at that node: `kind` 1 to 3 and `flags` 0 (reason 12);
   `n_pred` at most 4 (13); `skip_reg` and `result_reg` 0xFF or below 16 (4). A
   `RECORD`: template index below `n_records` and the template valid (5); `n_ops` 0,
   and a result register only with `KM_F_RESULT` in the template (12). A `CPU` or
   `JOIN`: no result register (12). A `JOIN`: `arg` and `n_ops` 0 (12). A `CPU`: its
   slice starts where the previous `CPU` node's ended (the first at 0), is not empty and
   ends within `n_ops` (12); each of its micro-ops is a compute or jump micro-op (2),
   uses registers below 16 (4) and jumps to a micro-op of the slice or to its end (3).
   Then its successor list: `succ_off` is where the previous node's list ended,
   `succ_off + succ_n` within `n_edges`, and the list rises strictly from above the
   node itself to below `n_nodes`, no node getting a fifth predecessor (13).
3. The lists end at `n_edges` (13, node 0xFFFF); the slices end at `n_ops` (12, node
   0xFFFF).
4. Each node's `n_pred` equals the times it appears in the lists (13, at the node).
5. Every template is valid (5, node 0xFFFF): segment 1 to 239; flags within
   `KM_F_EXCL | KM_F_RESULT | KM_F_QUEUE`; at most four patches, each of an argument
   below 10 from a register below 16; `first_reg` and `count_reg` 0xFF or below 16.

The compute and jump micro-ops are `SET`, `MOV`, `ADD`, `ADDI`, `SUB`, `MUL`, `AND`,
`SHR`, `MOD`, `RAND`, `JMP`, `JLT`, `JEQ`, `JNZ`, with the stream's meanings; a jump to
the slice's end finishes the node.

## Events and errors

| Kind | Posted | Aux | `ticks` word |
|---|---|---|---|
| `KM_EV_GRAPH` 7 | a round ended and another starts | rounds done | - |
| `KM_EV_GRAPH_END` 8 | the graph ended | register 0, low 24 bits | register 1 |
| `KM_EV_GRAPH_ERROR` 9 | any error; nothing follows it | reason, then the node in the 16 bits above | - |

| Reason | Meaning | Node |
|---|---|---|
| 1 | bad header or size; more than 16 bindings | 0xFFFF |
| 2 | a micro-op a `CPU` slice may not hold | the node |
| 3 | a jump outside the slice | the node |
| 4 | a register number of 16 or more | the node |
| 5 | bad template or template index | the node using it; 0xFFFF for one no node uses |
| 6 | a `CPU` node ran `step_limit` micro-ops without reaching its end | the node |
| 7 | `MOD` by zero | the node |
| 8 | a record's deadline passed | the node |
| 9 | a queued record whose count is 0 or above the grid | the node |
| 10 | cancelled | 0xFFFF |
| 11 | a record whose team is empty or leaves the grid (tested without a sum) | the node |
| 12 | bad node: kind, flags, a field its kind does not use, a slice | the node, or 0xFFFF |
| 13 | bad successor list or predecessor count | the node, or 0xFFFF |
| 14 | `again` still set after `max_runs` rounds | 0xFFFF |
| 15 | `KM_GRAPH_KEEP` while eight graphs are kept | 0xFFFF |

## The C engine

`km_graph_engine.c` is freestanding: no library calls, no floating point, no static
data, every loop bounded by a limit above; all state is in a `kmg_state_t` the caller
gives. The same file is compiled into the dispatch kernel and, on the host, into the
library `komorebi.engines.graph_native` loads with ctypes.

| Function | Contract |
|---|---|
| `kmg_check(upload, bytes)` | 0 or `reason \| node << 8` |
| `kmg_start(state, upload, bindings, n, n_cores)` | starts a checked upload, which stays in place and unchanged while it runs; 0, or 1 for more than 16 bindings |
| `kmg_step(state, issue, ctx)` | one step; returns `KMG_RUNNING`, `KMG_ROUND`, `KMG_END` or `KMG_ERROR` |
| `kmg_complete(state, node, result)` | a record completed; 1 (and nothing changes) if the graph is not running or the node has no record in flight |
| `kmg_stop(state, reason, node)` | ends a running graph from outside (deadline, cancel) |

`issue(ctx, node, &record)` gets the record with `seq`, the landing words and the slot
bits 0, its team resolved (a queued record keeps `KM_F_QUEUE` and `first` 0), and
returns `KMG_ISSUED` or `KMG_BUSY`. The first 20 words of the state are the registers,
the round, the phase, the error reason and the error node.

## The simulator

```python
from komorebi.engines.graph import simulate
from komorebi.engines.graph_native import NativeExecutor

done = g.simulate({"it": 0}, delay=2, result=lambda r: r.args[0] + r.count)
same = simulate(upload, bindings=bindings, delay=2, engine=NativeExecutor())
```

`simulate(upload, bindings=(), n_cores=119, delay=0, result=None,
deadline_passes=None, cancel_pass=None, max_passes=100_000, kept=0, engine=None)` runs
the check, refuses a kept graph when `kept` graphs are kept already and that is eight,
then runs the graph against a model of the kernel. Each pass: completions due are
swept in slot order; a record still busy `deadline_passes` passes after its issue ends
the graph (the lowest slot's); a cancel in `cancel_pass`; one executor step. Records
take the lowest free of 16 slots; an exclusive record waits while a busy team overlaps
it; a queued record is issued when a slot is free and the allocator finds room, else
it is busy (the kernel's queue accepts it earlier; only the timing differs). `delay`
and `result` are as for streams. `engine` is the executor to drive: `Executor()` in
Python by default, or `NativeExecutor()` for the C engine.

It returns `GraphSimResult(records, events, status, error, regs, runs, passes,
stopped, kept)` (`kept`: the graph stays kept after this run): every record as `GraphRecord(node, run, seq, slot, seg, first, count, flags,
args, pass_index, queued)`; events as `Event(kind, req, aux, extra, pass_index)`;
`status` `"ended"`, `"error"` or `"stuck"`.

## Reference graphs

`experiments/graph_checks/graphs/` holds 39 named graphs as `.bin` uploads with a
`.json` beside each: the bindings, the simulation parameters and the answer (the check,
status, error, final registers, rounds, the set of records issued and the events). They
are for another implementation, such as the dispatch kernel's, to replay; the set and
the registers are binding, the order is not. `graph_reference.py` in that folder
rewrites them and with `--check` compares both executors with them.
