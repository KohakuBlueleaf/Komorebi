# 07 work cluster

A program whose control loop is decided by a value computed on the card. Each core
holds one state word (page `p` of a DRAM tensor belongs to core `p`, starting at
`7 p`). Every round issues two records:

    round k:
        STEP  on cores [first, first + count):  x[p] += k + 1 + p
        SUM   on every core, exclusive, with a result word:
              the leader reads x[0 .. cores) and returns their sum
        sum >= limit          -> end
        otherwise             first = sum mod (cores - count + 1), next round

So the team of each round is chosen by the last result, and the number of rounds by
the card's arithmetic. The loop is written once, in plain C (`decide.c`), and run
three ways:

| Path | Who runs `decide.c` | How records reach the grid |
|---|---|---|
| `thread` | an X280 thread on work cluster 1 (`thread/driver.c`) | `kmx_issue` / `kmx_wait` on the thread's command ring |
| `hostfed` | the thread's host fallback (`session.load(prog, thread="host")`), through ctypes | `Run.issue(..., result=True)` and `Run.wait` on the run the fallback is given |
| `cpu` | the thread's own files on the CPU (`cpu_ring.c`) | the ring calls of `kmx_ring.h` against a fake kernel that runs STEP and SUM on a model of the grid |

Each must give the same records, rounds, final sum and team sequence as a numpy
replay written apart from `decide.c`; the card paths must also leave the state tensor
equal to the replay's. `thread` and `hostfed` load the same program, with the thread on
the card or on the host; the fallback (`host_fallback(regions, words, run)`) fills
region 0 and returns the result words as the thread does.

    python work_cluster.py [--paths cpu hostfed thread] [--cores 119] [--count 30]
                           [--limit 1000000] [--max-rounds 1000]

Prints one JSON line: the replay's code, rounds and sum, and for each path whether it
matched (`same_as_numpy`, `same_records`, `state_wrong`), and `ok`.

On a Blackhole p150a with the defaults: 204 rounds (408 records), final sum 1,004,917;
the thread on the card took 10.3 ms (about 50 microseconds per round, two records
issued and waited for in turn over its ring); host-fed 50 ms (about 0.24 ms per round);
records, teams, sums and the state tensor exact on every path.

## The thread

`thread/driver.c` is the whole thread: it reads the program's words, runs
`wc_round` / `wc_result` from `decide.c`, issues the round's two records one after
the other, waiting for each (an exclusive SUM issued while STEP is in flight on an
overlapping team would be refused, reason 11), and passes SUM's result word (the event's `ticks`) to `wc_result`. It
writes each round's STEP team into region 0 (`teams`) and gives it back at the end.

| Words | `state address, cores, count, limit, max rounds` |
|---|---|
| Codes | 0 the limit was reached; 1 bad words or region; 2 no end within the rounds allowed; 3 the ring failed (none, late, stopped); 4 a record did not finish |
| Result words | rounds, last sum, a hash of the STEP teams (`trail`) |
| Progress | `4 x round + step`: 1 records built, 2 issued, 3 both events in |

`cpu_ring.c` prints `record seg flags first count a0 a1` per record the fake kernel
ran, then `result code rounds sum trail`, then `teams` and region 0's words.

## The grid

`kernels/writer.cpp` runs both segments (the reader and compute kernels have nothing
to do): STEP reads and writes its core's page; SUM's leader reads every page and calls
`wr.result(sum)`, which the dispatch kernel returns as the result word of the SUM
record's done event.

## Files

| File | What it is |
|---|---|
| `decide.h`, `decide.c` | the decision: the records of a round and what a sum means |
| `thread/driver.c` | the X280 thread |
| `cpu_ring.c` | the thread on the CPU against a fake ring |
| `kernels/` | the grid's kernels |
| `work_cluster.py` | the host script: the replay, the three paths, the comparison |
