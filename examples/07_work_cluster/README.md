# 07 work cluster

A program whose control loop is decided by a value computed on the card. Each core
holds one state word (page `p` of a DRAM tensor belongs to core `p`, starting at
`7 p`). Every round submits two commands:

    round k:
        STEP  on cores [first, first + count):  x[p] += k + 1 + p
        SUM   on the loop's cores, exclusive, with a result word:
              the leader reads x[base .. base + cores) and returns their sum
        sum >= limit          -> end
        otherwise             first = base + sum mod (cores - count + 1), next round

So the group of each round is chosen by the last result, and the number of rounds by
the card's arithmetic. The loop is written once, in plain C (`decide.c`), and run
three ways:

| Path | Who runs `decide.c` | How commands reach the grid |
|---|---|---|
| `thread` | an X280 thread on a work cluster (`thread/driver.c`) | `kmx_submit` / `kmx_wait` on the thread's command ring |
| `hostfed` | the thread's host fallback (`session.load(prog, thread="host")`), through ctypes | `Run.submit(..., result=True)` and `Run.wait` on the run the fallback is given |
| `cpu` | the thread's own files on the CPU (`cpu_ring.c`) | the ring calls of `kmx_ring.h` against a fake kernel that runs STEP and SUM on a model of the grid |

Each must give the same commands, rounds, final sum and group sequence as a numpy
replay written apart from `decide.c`; the card paths must also leave the state tensor
equal to the replay's. `thread` and `hostfed` load the same program, with the thread on
the card or on the host; the fallback (`host_fallback(regions, words, run)`) fills
region 0 and returns the result words as the thread does.

    python work_cluster.py [--paths cpu hostfed thread] [--cores 119] [--count 30]
                           [--limit 1000000] [--max-rounds 1000] [--clusters 1 2 3]

`--clusters` runs one loop per cluster, all at once: the cores are split into equal
shares (`base`, the sixth word, is the share's first core), each cluster's thread runs
the loop on its share, and the host fallbacks run likewise for `hostfed`. Prints one
JSON line: each loop's replay (`replay_<cluster>`), and for each path and cluster
whether it matched (`same_as_numpy`, `same_commands`, `state_wrong`), and `ok`: every
path asked for ran and matched on every cluster. The script exits non-zero without it.

On a Blackhole p150a with the defaults (X280 at the session's 1000 MHz): 204 rounds
(408 commands), final sum 1,004,917; the thread on the card took 10.8 ms (about 53
microseconds per round, two commands submitted and waited for in turn over its ring);
host-fed 48 ms (about 0.23 ms per round); at the 1750 MHz opt-in, 10.3 and 50 ms;
commands, groups, sums and the state tensor exact on every path. On cluster 2 or 3
alone the thread takes 10.71 ms, with the same results. With `--clusters 1 2 3` (39
cores a loop), the three threads ran at once in 5.61, 4.73 and 4.07 ms for 239, 203 and
176 rounds, about 23 microseconds a round each; every loop exact.

## The thread

`thread/driver.c` is the whole thread: it reads the program's words, runs
`wc_round` / `wc_result` from `decide.c`, submits the round's two commands one after
the other, waiting for each (an exclusive SUM submitted while STEP is in flight on an
overlapping group would be rejected, reason 11), and passes SUM's result word (the event's `ticks`) to `wc_result`. It
writes each round's STEP group into region 0 (`groups`) and gives it back at the end.

| Words | `state address, cores, count, limit, max rounds, base` |
|---|---|
| Codes | 0 the limit was reached; 1 bad words or region; 2 no end within the rounds allowed; 3 the ring failed (none, late, stopped); 4 a command did not finish |
| Result words | rounds, last sum, a hash of the STEP groups (`trail`) |
| Progress | `4 x round + step`: 1 commands built, 2 submitted, 3 both events in |

`cpu_ring.c` prints `command handler flags first count a0 a1` per command the fake kernel
ran, then `result code rounds sum trail`, then `groups` and region 0's words.

## The grid

`kernels/writer.cpp` runs both handlers (the reader and compute kernels have nothing
to do): STEP reads and writes its core's page; SUM's leader reads every page and calls
`wr.result(sum)`, which the dispatcher returns as the result word of the SUM
command's done event.

## Files

| File | What it is |
|---|---|
| `decide.h`, `decide.c` | the decision: the commands of a round and what a sum means |
| `thread/driver.c` | the X280 thread |
| `cpu_ring.c` | the thread on the CPU against a fake ring |
| `kernels/` | the grid's kernels |
| `work_cluster.py` | the host script: the replay, the three paths, the comparison |
