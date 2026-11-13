# 06 graph

A [graph](../../docs/graphs.md) run on the card twice over: host-fed (the graph executor
runs on the host and issues its records through the dispatch kernel's host ring) and on
the X280 (the dispatch kernel's engine runs the uploaded graph and decides every next
step itself). The loop it contains ends on a value computed on the card.

```python
g = Graph(max_runs=50, req=6)
it = g.bindings("it")  # register 0
err = g.reg()  # register 1: the two an X280 run's end event carries
addr, limit = g.bindings("state", "limit")
a = g.record(STEP, team=(0, 60), args=[addr, it])
b = g.record(LEFT, team=(0, 30), args=[addr], after=[a])
c = g.record(RIGHT, team=(30, 30), args=[addr], after=[a])  # in flight with b
d = g.record(NORM, team=(0, 60), args=[addr, 60], after=[b, c], result=err)
e = g.cpu(after=[d])
e.addi(it, it, 1)
e.lt(g.again, err, limit)  # go round again while err < limit

got = run.run_hostfed(g, {"state": x_addr, "it": 0, "limit": 1_000_000})
got = run.run_hostfed(g, {"state": y_addr, "it": 3, "limit": 10_000_000})  # kept

up = run.upload(g)  # on the X280: uploaded once, kept, restarted by its req
t = run.start(up, {"state": x2, "it": 0, "limit": 1_000_000}, kept=True)
end = run.wait_engine(t)
end = run.wait_engine(run.restart(6, {"state": y2, "it": 3, "limit": 10_000_000}))
run.release(up)  # end.regs: (it, err)
```

The NORM leader's writer kernel gives the result word with `wr.result(sum)` before it
reports; the record carries `KM_F_RESULT`, so its acknowledgement brings the value.

The state is one 32-bit word per core (word 0 of a 64-byte page of a DRAM tensor).
Each round:

| Node | Team | What it does |
|---|---|---|
| `STEP` | cores 0 to 59 | `x[p] += it + p` |
| `LEFT` | cores 0 to 29, after `STEP` | `x[p] >>= 1` |
| `RIGHT` | cores 30 to 59, after `STEP`, in flight together with `LEFT` | `x[p] *= 3` |
| `NORM` | cores 0 to 59, after both | the leader reads every word; their sum is the record's result word, stored in register `err` |
| CPU | after `NORM` | `it += 1`; go round again while `err < limit` |

The state tensor's address, `it` and `limit` are bindings. The graph is built once and
run twice, the second time on a fresh tensor with other bindings, without being built
again. The host checks against a numpy replay of the same rounds: the number of rounds,
the final `it` and `err`, and every state word read back after the run. The trace shows
whether `LEFT` and `RIGHT` were in flight together in each round of the host-fed run.
On the X280 the host takes only the end event: `it` and `err` from registers 0 and 1,
the rounds from the round events, then the state read back. Each path has a run of its
own, so the kernel's result block gives each one's most slots busy at once (`max_busy`);
2 on the X280 means `LEFT` and `RIGHT` were in flight together there. Both paths must
also agree on what the card saw: the kernel's record count and every core's count of
records taken, against each other and against the records `Graph.simulate` issues (plus
the EXIT each run ends with).

The X280 run first gets four bad uploads (with `--decide host`, a run of their own:
a host-fed trace refuses a run with an engine in it), built in `graph.py` from a good
two-node graph with one fault each: a size other than the header's, a successor that is
not after its node, a template index past the last, and an ISSUE in a CPU node's slice.
Each must end in its error event (`header`, `edge`, `template`, `opcode`, at the node
of the fault), the host's `check` naming the same; a class that ends otherwise counts as
a failure (`bad_uploads_refused` below 4/4).

    python graph.py [--first IT LIMIT] [--again IT LIMIT] [--max-runs N]
                    [--decide host|x280|both] [--deadline S] [--stall]

`--stall` first runs, in a run of its own, a graph whose record never completes (on the
X280 unless `--decide host`): the record's deadline ends the graph with error reason 8,
and the dispatch kernel with its deadline error (4); the runs after it are exact. It
prints one JSON line; `ok` is true when every check above holds, and the script exits 1
when it is not. `tests/hardware/test_graph_host.py` runs it with `--stall`.

On a p150a (card 1, 60 cores, default arguments), host-fed and on the X280 alike: the
first run ends after 5 rounds and the kept graph's run after 7, as in the replay, with
every state word right; 49 records (48 and the EXIT) on both paths and in the simulator,
every core's count right; `max_busy` is 2 on both paths, and host-fed `LEFT` and `RIGHT`
were in flight together in 5 of 5 rounds. With `--stall --deadline 0.2` the stalled
graph ends with reason 8 at node 0 and the kernel with error 4, without a reset.

| Segment | Arguments |
|---|---|
| `STEP` | a0 state, a1 `it` |
| `LEFT`, `RIGHT` | a0 state |
| `NORM` | a0 state, a1 pages to sum |
| `STALL` (a test) | a0 the member that never reports |
