# 13 Thread commands

A command can go to a work cluster's X280 thread instead of a core group
([threads reference](../../docs/reference/threads.md#commands-to-a-thread)). The
thread's `kmx_program` answers it with a word, and that word comes back as the
command's result, as a group leader's word does. Here the host and a graph chain the
two kinds, round after round:

1. AFFINE on cores 0 to 7: the leader gives `a0 * 0x9E3779B1 + a1 + 8`
   (`kernels/writer.cpp`);
2. FOLD on the thread of cluster 1, taking that word: `x` xor the sum of 16 words of a
   256-word table the thread was given at load (`thread/fold.c`);
3. FOLD's word is the next AFFINE's `a1`.

```python
made = km.Program("thread_commands", kernels=KERNELS, handlers=KERNELS / "handlers.h")
made.thread(
    cluster=1,
    sources=[HERE / "thread" / "fold.c"],
    regions={"table": TABLE},
    handlers={"fold": 1, "served": 2},  # kHandlerFold, kHandlerServed in fold.c
)
run = session.load(made)
word = run.wait(run.submit("affine", (0, 8), args=(iteration, carry), result=True))
carry = run.wait(run.submit("fold", cluster=1, args=(word.result,), result=True))

grid = graph.command(affine, group=(0, 8), args=[iteration, carry], result=word)
then = graph.command(fold, cluster=1, args=[word], after=[grid], result=carry)
```

The thread has no loop of its own. `kmx_main` reads the table's lines once
(`kmx_invalidate`) and returns; the library then calls `kmx_program` for each command
until the run stops. SERVED returns how many commands the thread has answered, so the
host can check that none was lost or answered twice.

    python thread_commands.py [--cluster 1] [--rounds 200] [--graph-rounds 1000]
                              [--waves 200]

What it runs, in one run:

| Part | What | Checked |
|---|---|---|
| `host` | `--rounds` rounds from the host, one command at a time | every word, and the end against a numpy replay |
| `mixed` | `--waves` waves of 8 commands in flight, thread and grid alternating | every word |
| `graph_host` | the graph's `--graph-rounds` rounds host-fed | its registers against the replay |
| `graph_x280` | the same graph on the X280 (the dispatcher's engine sends FOLD to the thread itself) | the end event's registers |
| `graph_x280_grid_only` | the same graph with FOLD replaced by an AFFINE on core 8 | the end event's registers |
| `served` | SERVED, last | every FOLD counted once |

It prints one JSON line; `ok` is false, and the script exits non-zero, unless every
check holds and the dispatcher ended without an error.

## Numbers

A Blackhole p150a, the X280 at the session's 1000 MHz, every word right:

| | Grid command (AFFINE on 8 cores) | Thread command (FOLD) |
|---|---|---|
| From the host, one at a time, median | 132.6 µs | 133.0 µs |
| From the host, 8 in flight, per command (the two mixed) | 29.5 µs | |

| One round of the graph | Host-fed | On the X280 |
|---|---|---|
| AFFINE, then FOLD on the thread, then the CPU node | 329.9 µs | 19.3 µs |
| AFFINE, then AFFINE on core 8, then the CPU node | | 11.2 µs |

From the host the two kinds cost the same: the time is the host's round trip. On the
X280 a round with the thread costs 8 µs more than one with a second grid command: the
dispatcher stores the line into the thread's memory and takes its answer from the
thread's ring, and FOLD reads 16 table words from DRAM. The experiment
`experiments/thread_commands` measures the dispatcher's part alone (2.56 µs against
1.64 µs for a NOP to one core).
