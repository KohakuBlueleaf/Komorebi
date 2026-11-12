# 07 work cluster

A thread program with an X280 thread on work cluster 1. In this first form the thread
does the whole job and the grid only idles: the host fills a region with a cores x
bins table of counts, and the thread computes, on the cluster's four cores,

    offsets[c][b] = (counts of every core in bins before b) + (bin b on the cores before c)

the start of each core's share of each bin in a table ordered by bin, which is the
decision a renderer makes between its binning and its drawing. The thread writes the
offsets into a second region and returns the sum of all counts. The same program is
then loaded with `thread="host"`, which runs the Python function given as
`host_fallback` instead; the two must agree bit for bit.

    python work_cluster.py [--cores 119] [--bins 1024] [--seed 1]

Prints one JSON line: the card thread's result block, the fallback's result, and
whether the offsets and result words are the same (`ok`).

On a Blackhole p150a, 119 x 1,024 counts: the thread took 3.05 ms from its start to
its return (3.51 ms before the library set the L2 prefetchers), and the offsets and
the sum agreed with the fallback.

## The program

```python
prog = km.Program("work_cluster", kernels="kernels/")  # NOP records only
prog.thread(
    cluster=1,
    sources=["thread/prefix.c"],
    regions={"counts": counts, "offsets": counts.nbytes},  # ids 0 and 1
    words=[cores, bins],
    host_fallback=prefix_offsets,
    deadline=5.0,
)
with km.Session() as session:
    run = session.load(prog)  # or thread="host"
    result = run.thread_result(timeout=10.0)
    run.stop()
    offsets = run.thread.read("offsets").view(np.uint32)
```

The thread (`thread/prefix.c`) takes region 0 (`kmx_take`: the host wrote it, so the
cluster's cached lines go), runs two passes with `kmx_par` (each core totals its
quarter of the bins over all cores; then each core writes its offsets after the
totals of the cores before it), gives region 1 back (`kmx_give`) and stamps its
progress before and after each step. See [X280 threads](../../docs/threads.md).

## Files

| File | What it is |
|---|---|
| `work_cluster.py` | the host script: the program, both runs, the comparison |
| `thread/prefix.c` | the X280 thread |
| `kernels/` | the grid's kernels: every record is a NOP in this form |
