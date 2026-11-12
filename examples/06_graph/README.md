# 06 graph

A [graph](../../docs/graphs.md) run host-fed on the card: the graph executor runs on the
host and issues its records through the dispatch kernel's host ring; the loop it
contains ends on a value computed on the card.

```python
g = Graph(max_runs=50)
addr, it, limit = g.bindings("state", "it", "limit")
err = g.reg()
a = g.record(STEP, team=(0, 60), args=[addr, it])
b = g.record(LEFT, team=(0, 30), args=[addr], after=[a])
c = g.record(RIGHT, team=(30, 30), args=[addr], after=[a])  # in flight with b
d = g.record(NORM, team=(0, 60), args=[addr, 60], after=[b, c], result=err)
e = g.cpu(after=[d])
e.addi(it, it, 1)
e.lt(g.again, err, limit)  # go round again while err < limit

got = run.run_hostfed(g, {"state": x_addr, "it": 0, "limit": 1_000_000})
got = run.run_hostfed(g, {"state": y_addr, "it": 3, "limit": 10_000_000})  # kept
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
whether `LEFT` and `RIGHT` were in flight together in each round.

    python graph.py [--first IT LIMIT] [--again IT LIMIT] [--max-runs N] [--stall]

`--stall` ends with a graph whose record never completes: the record's deadline ends
the graph with error reason 8. It prints one JSON line.

On a p150a (card 1, 60 cores, default arguments): the first run ends after 5 rounds and
the kept graph's run after 7, as in the replay, with every state word right; `LEFT` and
`RIGHT` were in flight together in 5 of 5 rounds. With `--stall --deadline 0.2` the
stalled graph ends with reason 8 at node 0 and the run stops without a reset.

| Segment | Arguments |
|---|---|
| `STEP` | a0 state, a1 `it` |
| `LEFT`, `RIGHT` | a0 state |
| `NORM` | a0 state, a1 pages to sum |
| `STALL` (a test) | a0 the member that never reports |
