---
title: The programming model
summary: Threads with a program counter and a place, the breakpoints that move, emit and wait, homes, where a role is still used, and how several chunking axes in space and time are written.
tags:
  - concepts
  - model
---

# The programming model

Status: the breakpoints are implemented in the Tensix runtime and run on one card:
`emit` and `wait` as ports and reduce ([ports](../reference/ports.md)), a block to a
row or a column as broadcast broadcast ([broadcasts](../reference/broadcast.md)), and `at` as items
that move between cores in a dynamic handler ([dynamic](../reference/dynamic.md)).
Homes declared and the thread language compiled into handlers are planned; today a
program is written as the handler bodies of its kernels.

A Tensix grid has many cores and a network between them. A program for it has to say
what runs where, and what travels between cores. Komorebi's model keeps the part of
that which every program needs and drops the rest: you write one straight-line program
per item of work, as in SIMT, and mark the points at which that program has to deal
with another core.

## A thread has a program counter and a place

SIMT gives every thread a program counter. Here every thread has a program counter and
a **place**. A thread stays on its core by default. Between two breakpoints it runs on
that core without interruption, and after a move it stays where it landed.

A **breakpoint** is a point where the thread must deal with another core. There are
three kinds, and one variant of the first:

| The author writes | Breakpoint | What crosses the NoC | What the thread does |
|---|---|---|---|
| `at X[i]: block` | move | the thread's live variables | continues where `X[i]` lives; no breakpoint if it is already here |
| `emit X[i] += v`, `emit v -> thread(j)` | emit | the value | keeps running |
| `v = wait(n)` | wait | nothing | pauses until n values have landed |
| `at pool(p): block` | move to a pool | live variables | continues on a core of the pool; the dispatcher sizes the pool |

`reduce(op, part) over group via = owner | tree | ring` and
`scan(op, s) over group in order` are library forms built from `emit` and `wait`.

Nothing in the program says how a breakpoint is carried out. Where the sender, the
receiver and the number of arrivals are known before the run, a breakpoint becomes a
write into a fixed slot and a counted wait: no header and no queue. Where they depend
on data, it becomes a command with a header, an inbox ring and flow control. The author
writes the same line in both cases.

## Homes

The author never names a core. The author names state that has a **home**, and says
which state a value must meet. Where the home is placed is decided separately.

| Kind | Who holds it | Example |
|---|---|---|
| `home` (mutable) | exactly one core | a film tile, a screen tile, the running state of a recurrence, a block of a key-value cache |
| `copies` (read-only) | one core, or several when the data is hot; the sender picks one | a BVH treelet, a block of weights |
| `everywhere` (small, read-only) | every core | the top of a BVH, placement tables |

Every core can run every piece of code. A home pins state to a core, not code to a core.
`at X[i]` is the way a thread meets state: it continues on the core that holds `X[i]`.

## An example: ray tracing as one thread

```text
thread path(pixel):
    r = camera_ray(pixel);  c = 0;  thr = 1
    for depth in 0 .. MAX:
        hit = none
        for k in treelets_along(r):               # top of the BVH, on every core
            at treelet[k]:                        # breakpoint: move, unless treelet[k] is here
                hit = closer(hit, walk(r, treelet[k]))
        if not hit:  c += thr * sky(r);  break
        m = material(hit)                         # no breakpoint: shading happens where the ray is
        c += thr * emitted(m);  thr *= bsdf(m);  r = bounce(r, m)
    emit film[pixel] += c                         # breakpoint: a value goes, the thread ends
```

The code between breakpoints is a **handler**. This program has three:

| Handler | Starts when | What travelled |
|---|---|---|
| S0 | a thread is started on a core | `pixel` |
| S1 | a thread arrives at a treelet | the 64-byte ray command: `r, hit, c, thr, depth, k, pixel` |
| S2 | a value arrives at a film tile | `pixel, c` |

- On the card this is one persistent kernel with a switch over handler numbers.
- Threads waiting at the same handler on the same core run together, 32 per packet, one
  per vector broadcast. Nothing is masked.
- If the whole scene is on every core, no `at` fires: SIMT is the degenerate case.
- Cutting the scene differently changes how often threads move, not the program.

## An example: matrix product with K split, merged by a tree

Every thread runs the same program; what differs is its index. There is no separation
into roles.

```text
thread mm(j, n):                                 # one per core, pinned
    part = A[:, K(j)] @ B[K(j), :]               # one handler; no breakpoint
    for r in 0 .. levels(n) - 1:
        if j % 2^(r+1) == 0:                     # I merge in this round
            if j + 2^r < n:
                part += wait(1)                  # breakpoint: my partner has handed over
        else:
            emit part -> thread(j - 2^r)         # breakpoint: hand my partial to my merger
            return
    write C = part                               # only thread 0 arrives here
```

```text
 round 0:   t1 -> t0     t3 -> t2     t5 -> t4     t7 -> t6
 round 1:   t2 -> t0                  t6 -> t4
 round 2:   t4 -> t0
```

When C is larger than one core's local memory, threads are indexed (i, j): i picks the
output block, j the K slice, and merging happens only among threads with the same i.
`via = owner` is the simple alternative: one core adds every partial, one after the
other.

## Where a role is still used

A role is a unit of dispatch: which cores do which kind of work, sized by load. It is
used only where there is a real balance decision, and it is not the centre of the model.

- **Which program a group hosts**, for example prefill jobs against decode jobs, sized
  per job by the dispatcher.
- **A pool for heavy work with no data affinity**, such as a neural material network in
  shading: hits are gathered and run in batches on as many cores as the load needs.

Plain ray tracing, rasterisation, matrix product and attention need no role. Two things
that look like roles are not: a resume point, where a core starts executing when
something arrives, is always present and is produced by the compiler; a named piece of
code is just a function.

## Several axes at once: space and time

Real kernels cut more than one axis: some across cores, some in time on one core. The
model states both.

- **Space is the thread's index.** `thread f(broadcast, i)` has one thread per index point,
  pinned to a core. An axis split across cores is an index dimension.
- **Time is a loop in the thread body.** An axis split in time on one core is a `for`
  over chunks.
- **An axis may be both.** The index picks my share of the axis; a loop walks my share
  in chunks.
- **A group is a slice of the index space.** `threads(broadcast, *)` is every thread of one
  broadcast. A broadcast is `emit x -> threads(broadcast, *)`; a merge along one index is
  `reduce(op, x) over threads(i, *)`.
- **Chunk sizes are named parameters**, with a binding time: fixed at compile time, per
  size class, or at run time.

A fused attention kernel as a thread program:

```text
thread attend(broadcast, i):                        # thread index: two spatial cuts
    for job in jobs(broadcast):                     # time: (batch, head, row range)
        rows = partition(job.rows, i)              # space: query rows over the broadcast's cores
        for kb in key_blocks(job.head, KB):    # time: keys, KB tiles per block
            if i == 0:  kv = read_dram(kb);  emit kv -> threads(broadcast, *)    # breakpoint: broadcast
            else:       kv = wait(1)                                        # breakpoint
            for sub in chunks(kv, 3):          # time: 3 key tiles per step
                for r in rows:                 # time: my rows, paired
                    acc[r] = merge(acc[r], score(Q[r], sub))                # D summed inside score
        write out[rows]
```

The cuts form a nest that alternates between space and time: the grid is cut in space
into broadcasts; a broadcast works in time through its jobs; each job is cut in space over the
broadcast's cores; each core works in time through key blocks, sub-chunks and its own rows.

What an axis does to the program:

| Treatment of an axis | In the program | Breakpoints it creates |
|---|---|---|
| Split across threads, and the output is indexed by it (query rows; m and n in matmul) | an index dimension | none |
| Split across threads, and the output reduces over it (k in split-K matmul; keys in decode against a large cache) | an index dimension and a `reduce` along it | emit and wait: a merge |
| Looped in time on each thread, with data that many threads need (keys in a flash-attention kernel) | a loop, and a broadcast to a group | a broadcast, or none if every thread reads DRAM itself |
| Looped in time on each thread, with data only this thread needs (head dimension, value dimension) | a loop | none |

The same holds for a matrix product `C[m, n] = sum over k of A[m, k] B[k, n]`, where k
is cut both across cores and in time:

```text
thread mm(i, j):                               # space: i = a block of output rows m; j = a slice of k
    for nb in col_blocks(N, NB):               # time: output columns
        part = 0
        for kc in chunks(K(j), KC):            # time: my slice of k, in chunks that fit local memory
            part += A[M(i), kc] @ B[kc, nb]
        part = reduce(+, part) over threads(i, *)   via = tree     # breakpoints: merge along j only
        if j == 0:  write C[M(i), nb] = part
```

The cut decides the communication; the program text differs only in the index, the
loops and the `reduce` line.

## What this costs

1. Live state must be small and visible: whatever is live at a move travels. The tool
   must report the command size at each breakpoint.
2. No unbounded recursion across a breakpoint: a thread has a fixed-size command, not a
   stack.
3. Compute follows data, so hot data makes hot cores. Shedding is legal only at points
   where the thread needs nothing local; the compiler can find those points.
4. The on-core scheduler has to choose which handler to run when several have threads
   waiting, and whether to run a half-full packet or wait for 32.
5. A wait for an unknown number of values is the end-of-run problem: the run ends when
   every core has consumed what was sent to it.

## Open

- Who chooses the cut and the sizes: the author, or a planner working from the memory
  budget and the load.
- The tool should print, for any thread program, the table of axes: where each is cut
  (space or time), the chunk size and the breakpoints it creates.
- Binding times for size parameters: some sizes force a new kernel when they change,
  others are run-time values.

Related: [dispatch](dispatch.md) decides what runs; [runtime](runtime.md) is what runs
on each core.
