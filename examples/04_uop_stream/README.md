# 04 uop stream

The deciding is done by a micro-op stream: 1,000 `MAP` jobs, each on a team whose size
the stream draws at random, run three ways that must give the same tiles.

```python
s = ks.Stream(seed=seed)
j, x, count, first, room, base, page = s.regs(7)
top = s.label()
s.rand(x)
s.mod(count, x, n_cores)
s.addi(count, count, 1)  # 1 to every core
if form == "issue":  # the stream picks the first core too
    s.sub(room, n_cores, count)
    s.addi(room, room, 1)
    s.shr(first, x, 8)
    s.mod(first, first, room)
    s.issue(MAP, (first, count), args=[inp, T, out, base, page])
    s.wait(0)  # one record in flight
else:  # the dispatch kernel's queue picks it
    s.submit(MAP, count, args=[inp, T, out, base, page])
s.add(base, base, count)  # the next job's output tiles
s.addi(page, page, 61)  # the input window moves (modulo W)
...
s.addi(j, j, 1)
s.jlt(j, n_jobs, top)
s.wait(0)
s.halt(j)
```

| Way | Who decides | How the records go |
|---|---|---|
| `hostfed` | the host | the package's stream simulator runs the ISSUE stream; its records go to the dispatch kernel one command per job (`Run.issue_many`) |
| `issue` | the X280 (core 1 of cluster 0) | the ISSUE stream is uploaded and started; it issues each record and waits for it |
| `submit` | the X280 and the dispatch kernel's job queue | the SUBMIT stream is uploaded and started; the queue gives each job the lowest free team of its size, so several jobs are in flight |

`MAP` (`kernels/`): member m of a job adds its share of the job's T input tiles (from
tile a4) into one tile and stores it at output page a3 + m. Teams differ between the ways
(only `issue` and `hostfed` pick the same ones), but every job writes the same tiles. The
inputs are integers in [-2, 2], so every sum is exact in bf16, and the host compares every
output tile of every way bit for bit with numpy. It also checks each core's record count
against the simulated teams (`hostfed`, `issue`) or the total of member records
(`submit`), and gives microseconds per job, from the first command to the last event.

## Running it

    python uop_stream.py --card 0                   # all three ways
    python uop_stream.py --card 0 --ways hostfed    # one way

Options: `--jobs` (1,000), `--tiles` T (64), `--seed` (the stream's xorshift seed),
`--depth` (ring slots, 16), `--deadline` (per record, seconds). It prints one JSON line.

On a p150a (card 0, 120 cores, default arguments), host-fed: 1,000 jobs, 59,456 tiles,
0 wrong, every core's count right, 3.4 microseconds per job.
