---
title: "Twelve cores"
part: 5
summary: "the same thread on three clusters: kmx_parallel_all runs one function on all twelve X280 cores"
---

# Twelve cores

When the same thread runs on all three clusters, `kmx_parallel_all` runs one function on all
twelve X280 cores, and one cluster, the lead, tells the others what to do. This lesson relies
on [[four-cores]] and [[sharing-memory]].

## Why three clusters

Work that is arithmetic, such as the sum of [[four-cores]], gets faster with every core you
add. A cluster has four. The card has three clusters that you can use, and so twelve cores in
all. The price is that the clusters are three separate computers: each has its own copy of
the thread's regions, in its own memory bank, and a core reads only the bank of its own
cluster.

## The general form

In Python, where `thread` was:

```python
program.thread(
    clusters=(1, 2, 3),
    sources=[FILES],
    regions={NAME: SIZE},
    words=[NUMBERS],
)
results = run.thread_results(timeout=SECONDS)
part = run.threads[CLUSTER].read(NAME)
```

In the C file, one image for all three clusters:

```c
static void piece(int cluster, int core, uint64_t argument) {
    BODY
}

int kmx_main(int core) {
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : 2;
    }
    BODY
    int status = kmx_parallel_all(piece, ARGUMENT);
    ...
}
```

- `clusters=(1, 2, 3)` declares the same thread on each of these clusters. Each gets
  its own copy of every region, with the same names and the same ids.
- `kmx_cluster()` is the number of the cluster this code runs on, and `kmx_lead()` is the
  lowest of the clusters that run this thread: the **lead**. With all three, the lead is
  cluster 1.
- The lead runs the program. The other clusters call `kmx_serve()`, which waits for the lead's
  calls and carries them out, until the lead's `kmx_main` returns. `kmx_serve` returns
  `KMX_OK` when it ends well.
- `kmx_parallel_all(piece, ARGUMENT)` is called on core 0 of the lead only. It runs
  `piece(cluster, core, ARGUMENT)` on every core of every cluster, twelve in all, and
  returns when all twelve have finished. `ARGUMENT` is one 64-bit number, the same on all of
  them. The first call waits until the other clusters are serving.
- `piece` is yours. It gets the cluster, the core in that cluster (0 to 3), and the
  argument. The number of the core in the whole, 0 to 11, is `(cluster - 1) * 4 + core`.
- `run.thread_results(timeout=SECONDS)` is `thread_result` for all the threads: a
  dictionary from cluster to result.
- `run.threads[CLUSTER].read(NAME)` reads a region of one cluster's thread, after
  `run.stop()`, as in [[sharing-memory]]. Data from one cluster to another goes by a call
  `kmx_push`; this lesson does not need it, because Python reads each cluster's piece.

## One line at a time

```{ .python }
--8<-- "tutorial/35_twelve_cores/twelve_cores.py:thread"
```

The same declaration as before, with `clusters=(1, 2, 3)` in the place of `cluster=1`. The
region `pieces` is 256 bytes, four lines of 64: one line for each core of a cluster.

```{ .c }
--8<-- "tutorial/35_twelve_cores/thread/twelve.c:serve"
```

The first lines of `kmx_main`. Clusters 2 and 3 never leave this `if`: they serve until the
lead is done. Only cluster 1 goes on.

```{ .c }
--8<-- "tutorial/35_twelve_cores/thread/twelve.c:piece"
```

Each of the twelve cores takes a twelfth of the numbers, adds them as in [[four-cores]], and
stores the sum in its line of its own cluster's region. `kmx_flush_at(0, core * 64, 64)`
flushes that one line, from this core alone.

```{ .c }
--8<-- "tutorial/35_twelve_cores/thread/twelve.c:all"
```

The lead calls `kmx_parallel_all` twice, and times each call. The sum is the same both
times; the point of the second call is to see what the first one costs.

```{ .python }
--8<-- "tutorial/35_twelve_cores/twelve_cores.py:read"
```

After the run, Python reads the `pieces` region of each cluster's thread and keeps the first
number of each of the four lines. Twelve pieces in all. Python adds them and compares with
numpy.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "clusters 1, 2 and 3 start the\nsame thread", "gem": "sapphire"},
  {"t": "clusters 2 and 3: kmx_serve\nwait for the lead's calls", "gem": "sapphire"},
  {"t": "cluster 1, the lead:\nkmx_parallel_all(piece, n)", "gem": "sapphire"},
  {"t": "all twelve cores run piece:\neach adds a twelfth of n"},
  {"t": "Python reads the pieces of the\nthree clusters and adds them"}]}
</script>
<figcaption>One lead, two clusters that serve, twelve cores that work.</figcaption>
</figure>

## The whole program

=== "twelve_cores.py"

    ``` { .python }
    --8<-- "tutorial/35_twelve_cores/twelve_cores.py"
    ```

=== "thread/twelve.c"

    ``` { .c }
    --8<-- "tutorial/35_twelve_cores/thread/twelve.c"
    ```

The kernels are the empty ones. Run it from the folder `tutorial/35_twelve_cores`:

```bash
python twelve_cores.py
```

```{ .text .km-output }
--8<-- "tutorial/35_twelve_cores/output.txt"
```

The twelve pieces add up to the sum numpy computes. The second call took about six
milliseconds for 12 million numbers. By the rate of [[four-cores]], one core adds four million
of them in 24 milliseconds, so one core would take 72 for these twelve million: the twelve
cores were 12 times as fast. The first call took far longer, tens of milliseconds, and its
time changed from run to run, as the other clusters were still starting. A thread that calls
`kmx_parallel_all` once pays that wait once.

Work that mostly moves memory does not gain as much as this sum does. Each cluster reads its
bank at a fixed speed, so more cores do not make memory move faster. Arithmetic scales; copying
does not.

## Common mistakes

- Writing the thread for one cluster and declaring it on three. Every cluster runs the same
  `kmx_main`. The `if` that sends the others to `kmx_serve` is what makes them
  serve. Without it, the lead's first `kmx_parallel_all` waits for clusters that are not
  serving.
- A `piece` that reads a region of another cluster. A core reaches only the bank of its own
  cluster. Put the data in each cluster's own region, or push it with `kmx_push`.
- A number of items that twelve does not divide. As in [[four-cores]], the last items belong
  to nobody's piece, and the sum is wrong without a message.
- Counting the first call. Time the second, or time the whole program.

Next: [[core-asks-x280]] turns round: a Tensix core asks the X280 for something.
