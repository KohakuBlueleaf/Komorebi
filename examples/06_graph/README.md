# 06 graph

A [graph](../../docs/graphs.md) run on the card twice over: host-fed (the graph executor
runs on the host and submits its commands through the dispatcher's host ring) and on
the X280 (the dispatcher's engine runs the uploaded graph and decides every next
step itself). The loop it contains ends on a value computed on the card.

```python
graph = Graph(max_runs=50, req=6)
iteration = graph.bindings("iteration")  # register 0
error = graph.register()  # register 1: the two an X280 run's end event carries
state_address, limit = graph.bindings("state", "limit")
step = graph.command(STEP, group=(0, 60), args=[state_address, iteration])
left = graph.command(LEFT, group=(0, 30), args=[state_address], after=[step])
right = graph.command(
    RIGHT, group=(30, 30), args=[state_address], after=[step]
)  # with left
norm = graph.command(
    NORM, group=(0, 60), args=[state_address, 60], after=[left, right], result=error
)
cpu = graph.cpu(after=[norm])
cpu.addi(iteration, iteration, 1)
cpu.lt(graph.again, error, limit)  # go round again while error < limit

bindings = {"state": first_state, "iteration": 0, "limit": 1_000_000}
got = run.run_hostfed(graph, bindings)
bindings = {"state": second_state, "iteration": 3, "limit": 10_000_000}
got = run.run_hostfed(graph, bindings)  # kept: the same graph, other bindings

upload = run.upload(graph)  # on the X280: uploaded once, kept, restarted by its req
bindings = {"state": first_state, "iteration": 0, "limit": 1_000_000}
handle = run.start(upload, bindings, kept=True)
end = run.wait_engine(handle)
bindings = {"state": second_state, "iteration": 3, "limit": 10_000_000}
end = run.wait_engine(run.restart(6, bindings))
run.release(upload)  # end.registers: (iteration, error)
```

The NORM leader's writer kernel gives the result word with `writer.result(total)` before it
reports; the command carries `KM_FLAG_RESULT`, so its acknowledgement brings the value.

The state is one 32-bit word per core (word 0 of a 64-byte page of a DRAM tensor).
Each round:

| Node | Group | What it does |
|---|---|---|
| `STEP` | cores 0 to 59 | `x[p] += iteration + p` |
| `LEFT` | cores 0 to 29, after `STEP` | `x[p] >>= 1` |
| `RIGHT` | cores 30 to 59, after `STEP`, in flight together with `LEFT` | `x[p] *= 3` |
| `NORM` | cores 0 to 59, after both | the leader reads every word; their sum is the command's result word, stored in register `error` |
| CPU | after `NORM` | `iteration += 1`; go round again while `error < limit` |

The state tensor's address, `iteration` and `limit` are bindings. The graph is built once and
run twice, the second time on a fresh tensor with other bindings, without being built
again. The host checks against a numpy replay of the same rounds: the number of rounds,
the final `iteration` and `error`, and every state word read back after the run. The trace shows
whether `LEFT` and `RIGHT` were in flight together in each round of the host-fed run.
On the X280 the host takes only the end event: `iteration` and `error` from registers 0 and 1,
the rounds from the round events, then the state read back. Each path has a run of its
own, so the kernel's result block gives each one's most slots busy at once (`max_busy`);
2 on the X280 means `LEFT` and `RIGHT` were in flight together there. Both paths must
also agree on what the card saw: the kernel's command count and every core's count of
commands taken, against each other and against the commands `Graph.simulate` submits (plus
the EXIT each run ends with).

The first path's run (the host-fed one by default, so its trace is read in a run that
has started engines) first gets four bad uploads, built in `graph.py` from a good
two-node graph with one fault each: a size other than the header's, a successor that is
not after its node, a template index past the last, and an SUBMIT in a CPU node's slice.
Each must end in its error event (`header`, `edge`, `template`, `opcode`, at the node
of the fault), the host's `check` naming the same; a class that ends otherwise counts as
a failure (`bad_uploads_rejected` below 4/4).

    python graph.py [--first ITERATION LIMIT] [--again ITERATION LIMIT] [--max-runs N]
                    [--decide host|x280|both] [--deadline S] [--stall]

`--stall` first runs, in a run of its own, a graph whose command never completes (on the
X280 unless `--decide host`): the command's deadline ends the graph with error reason 8;
the dispatcher gives the late command up and the run goes on, so the stop is clean
(`"stop": "clean"`); the runs after it are exact. It
prints one JSON line; `ok` is true when every check above holds, and the script exits 1
when it is not. The `--stall` form waits out a deadline, so card_check's list
(`src/komorebi/tools/card_checks.txt`) leaves it out: run it by hand.

On a p150a (card 1, 60 cores, default arguments), host-fed and on the X280 alike: the
first run ends after 5 rounds and the kept graph's run after 7, as in the replay, with
every state word right; 49 commands (48 and the EXIT) on both paths and in the simulator,
every core's count right; `max_busy` is 2 on both paths, and host-fed `LEFT` and `RIGHT`
were in flight together in 5 of 5 rounds. With `--stall --deadline 0.2` the stalled
graph ends with reason 8 at node 0 and the stop is clean.

| Handler | Arguments |
|---|---|
| `STEP` | a0 `state_address`, a1 `round` |
| `LEFT`, `RIGHT` | a0 `state_address` |
| `NORM` | a0 `state_address`, a1 `pages` to sum |
| `STALL` (a test) | a0 `stalled_rank`, the rank that never reports |

## Over islands

`islands.py` runs the same graph over every rank of a run of several islands, from
island 0 alone: `STEP` and `NORM` on all of them, `LEFT` on the first half and `RIGHT`
on the second, which with two islands of equal ranks is island 1's. Host-fed through
island 0's run (`run.island(0).run_hostfed`), then uploaded to island 0's dispatcher and
run by its engine; island 0's dispatcher sends each node's island-1 part across
([commands across islands](../../docs/reference/threads.md#commands-across-islands)).
Each case's rounds, registers and every rank's state word are checked against the numpy
replay.

    python islands.py [--cards 0,1] [--first ITERATION LIMIT] [--again ITERATION LIMIT]
                      [--max-runs N] [--deadline S]

On two p150a cards (2026-11-16, 236 ranks, 118 an island, X280 at 1350 MHz): host-fed and
on island 0's X280 alike, the first case ends after 4 rounds and the kept graph's run
after 6, as in the replay, every one of the 236 state words right on both paths, no
kernel error on either island.
