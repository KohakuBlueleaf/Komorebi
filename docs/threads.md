---
title: X280 threads
summary: How to write the irregular part of a program as C on an X280 work cluster, with the kmx library, and how the host declares, starts, checks and replaces it.
tags:
  - reference
  - x280
  - threads
---

# X280 threads

A thread program runs in three places: the resident worker on the Tensix grid, the
dispatch kernel on X280 cluster 0, and, when the program needs them, **X280 threads**
on work clusters 1, 2 and 3, one per cluster. A thread is the irregular part of the program: a
decision over every core's data (prefix sums over all cores' counts, a cost split),
work whose size is not known ahead, the program's own control loop. It is ordinary C
on the cluster's four coherent 64-bit cores. It does not compute on tiles.

Status: a thread can be declared, built, loaded beside the dispatch kernel, read and
write its regions, run on all four cores, return a result, and be replaced by its host
fallback, and issue records to the grid through its command ring to the dispatch
kernel (`kmx_issue`, `kmx_wait`, `kmx_poll`), which the session sets up at load. A
program may run threads on all three work clusters at once, each with its own ring, and
two threads on one bank may share regions. Inside a cluster, hart 0 starts a function
on another hart and joins it later (`kmx_spawn`, `kmx_join`; `kmx_par` is both); across
clusters, one image runs a function on all twelve harts (`kmx_par_all`) and pushes data
into another cluster's bank (`kmx_push`). X280 cores and Tensix cores exchange messages both
ways through X280 ports, without the dispatch kernel ([X280 ports](#x280-ports)).

## What a thread can touch

| Thing | Access |
|---|---|
| its argument block | read: the region table and the program's own words |
| its regions | read and write, as memory: pages of its cluster's DRAM bank, which Tensix cores reach through the NoC |
| shared regions | the same, for regions the program shares between threads on one bank |
| other clusters | write only: `kmx_push` stores into another cluster's region (see [Clusters and banks](#clusters-and-banks)) |
| its result block | write: a code, up to 48 words, a progress stamp |
| the Tensix grid | none: everything goes through DRAM regions |
| the host | none while it runs: the host starts it, may ask it to stop, reads its result |

**Regions are the meeting place.** A Tensix core writes into a region through the
NoC, the thread reads it as memory, and the reverse. The cluster reads its DRAM bank
through its cache, and a NoC write does not pass through that cache, so a region
changes hands explicitly:

- `kmx_take(id)` before the thread reads what the grid (or the host) wrote: it drops
  the cluster's cached lines of the region;
- `kmx_give(id)` after the thread wrote what the grid (or the host) will read: it
  writes the cached lines back to DRAM.

Both are a sweep of the cache controller's flush register over the region's lines
(`experiments/x280_threads/` measures what happens without them and what they cost).

**Decisions travel as result words and events; bulk data after the stop.** A value
that steers the program while it runs (the one that ends a loop, a count that sizes
the next segment) comes back as a record's result word: a record issued with
`KM_F_RESULT` is acknowledged by its team's leader with a 32-bit result, which arrives
in the `ticks` field of its `KM_EV_DONE` event, to the thread through its ring and to
a host-fed program through the host ring, by the same path. The host reads a region of
DRAM only after the run has stopped; while the grid is resident it does not read DRAM
at all.

## Clusters and banks

| Cluster | NoC tile | Its DRAM bank |
|---|---|---|
| 1 | (8, 9) | 6 |
| 2 | (8, 5) | 7 |
| 3 | (8, 7) | 7 |

A cluster reads and writes memory through its cache only in its own bank (the start-up
check fails with the error `bank` otherwise), so a program's regions follow the banks:

- **A region has a home bank:** the bank of the threads that use it. A thread reads
  only regions of its own bank; `kmx_cluster()` says which cluster it runs on.
- **Clusters 2 and 3 share bank 7.** A region declared with `Program.region` and named
  in both threads' `shared=` is one region for both; it changes hands between them with
  `kmx_give` by the writer and `kmx_take` by the reader, as with the grid. The host
  refuses a shared region named by threads on two banks.
- **Data crossing banks is pushed by its producer**, never pulled: `kmx_push` stores
  into another cluster's region through NoC windows (a push of generated data runs at
  2.3 GB/s, a pull at 0.024 GB/s). See [Twelve harts](#twelve-harts).
- **A region a Tensix team and a thread share lives in the thread's bank.** Tensix
  cores reach every bank over the NoC; the thread reaches only its own.

Measured at the session's 1,000 MHz, four harts per cluster: each cluster reads and
writes its bank at about 1.2 GB/s, alone or with the others (the limit is per cluster,
not per bank: clusters 2 and 3 together move 2.3 GB/s of bank 7, all three 3.5 GB/s);
one cluster wakes another and gets an answer in 0.66 µs through a flag in the other's
local memory, in 1.2 µs through a shared bank-7 region. See `experiments/work_clusters/`.

## The command ring

```c
km_record_t r = {0};
r.hdr = SEG_MAP | KM_F_RESULT << 8;      /* a segment of the program, with a result */
r.team = first | count << 16;
uint32_t ticket;
if (kmx_issue(&r, &ticket) != KMX_OK) { /* KMX_FULL: no free slot yet */ }
kmx_event ev;
if (kmx_wait(ticket, kmx_time() + KMX_TICKS_PER_SECOND, &ev) == KMX_OK &&
    ev.kind == KM_EV_DONE) {
    uint32_t value = ev.ticks;           /* the team leader's result word */
}
```

The ring follows the host ring's protocol (the dispatch kernel's README, "Protocol with
the host"): command `i`, numbered from 1, is one 64-byte line in slot `i % depth`,
written words first and then its number in `seq` and the three landing words; a slot
is reused once the kernel's consumed count allows it; each finished command comes back
as one 16-byte event whose index is read before and after its fields; the thread
writes the number of events it took into a line the kernel reads. The ticket is the
command's number. `kmx_wait` keeps the events of other tickets for `kmx_poll`, up to
32. Where the ring lives is data: the host writes its addresses, depths and kind into
the thread's ring line before the start (`kmx_ring_desc` in `kmx_layout.h`). The
session puts each half where its reader is: the command lines and the read line in the
dispatch kernel's local memory, which the thread writes through a NoC window its
start-up opens (`KMX_RING_REMOTE`), and the counts and events in the thread's own local
memory, which the kernel writes. Nothing passes through a cache, so no hand-over needs
a sweep (a ring in cached memory, `KMX_RING_CACHED`, would). Only hart 0 issues and
waits. A missed deadline or a stop while waiting ends the thread's run as `late` or
`stopped`.

The kernel never makes a thread's record wait: one that cannot be issued now (an
exclusive record whose team overlaps one in flight, a full job queue, no free slot) is
refused at once with reason 11, and the kernel then sends the thread one wake event
(`KMX_EV_WAKE`, ticket 0) when a slot frees or the queue issues a job: the thread issues
the record again after it (`kmx_poll` returns the wake; `kmx_wait` passes it over). A round of two records issued and waited for in turn takes
about 53 microseconds at the session's 1000 MHz (50 at the 1750 MHz opt-in); the
kernel's loop looks at one work cluster's ring per pass, each every fourth pass.

## Writing one

```c
#include "kmx.h"

static void work(int hart, void* arg) {
    /* hart 0 to 3: this core's share of the job */
}

int kmx_main(int hart) {                  /* hart 0 */
    uint64_t bytes;
    uint32_t* in = kmx_region(0, &bytes); /* region 0, as declared on the host */
    kmx_take(0);
    if (kmx_par(work, in) != KMX_OK) {    /* all four cores, bounded */
        return 2;
    }
    kmx_give(1);
    uint32_t sum = 0;
    kmx_result(0, &sum, 1);
    return 0;
}
```

The library's start-up is the image's `main`. It turns floating point and vector on,
sets each core's L2 prefetcher to the values the chip's documentation recommends
(scalar code that walks memory runs about 20 % faster; vector reads do not gain and
vector writes lose 20 to 40 %; `prefetch="keep"` on the host leaves them as they are),
checks that the argument block is this run's and every region lies in range, checks
that the cluster sees the bank the host wrote (a line with a value only this run
knows), then calls `kmx_main` on hart 0. Harts 1 to 3 wait for what `kmx_spawn` gives
them until `kmx_main` returns; then the library gives back every region (a dirty line
left in the cluster's cache would be written over a later load's data when it is
evicted: it once failed the next load's bank check). Every call is listed in the
[reference](reference/threads.md).

[08_parallel_thread](../examples/08_parallel_thread/) is a whole thread built on
`kmx_par`: a map, a reduction and a two-pass prefix sum, each on one hart and on four,
checked against numpy and timed, with the rules its numbers give: which calls harts 1
to 3 may make, shares in 4 KiB chunks, arrays placed off each other's 128 KiB cache
way, false sharing, the vector unit.

Rules. The runtime's stop is a word the cores must poll: a core that never polls it
never returns, its cluster stays stuck, and only a reset of the whole card brings it
back. That is why every loop in a thread is bounded:

- **Every loop ends by itself.** Bound it by a count, or poll `kmx_stop_requested()`,
  which is true once the host or the runtime asked for a stop or the thread's deadline
  passed. The library's own waits (`kmx_par`, `kmx_join`, `kmx_par_all`, `kmx_serve`)
  are bounded by that deadline.
- **Pointers come from `kmx_region`.** Write nowhere else. Globals need a non-zero
  initialiser and `__attribute__((section(".data")))` (the image has no `.bss`).
- **No libc.** The image links without one; a struct assignment can become a `memset`
  call that does not link.
- **`kmx_progress(where)`** leaves a stamp the host reads with the result, also after a
  timeout: say where the thread is before and after each step.
- **The image is at most 126,976 bytes** (the part of the cluster's memory that
  instructions can be fetched from); globals count.

`kmx_time()` counts the 50 MHz timer (`KMX_TICKS_PER_SECOND`).

What it costs, measured at 1750 MHz (the experiment sets it; the session's default,
1000 MHz, is slower; one PLL clocks all four clusters): the cluster reads its bank at
about 1.3 GB/s and writes it at about 1.3 to 1.8 GB/s with four cores, and a hand-over
sweep runs at 5 to 8 GB/s of region. Prefix sums over 119 x 1,024 counters take about
1.7 ms on four cores (a decision per frame, not per segment); see
`experiments/x280_threads/`.

Where data lives matters as much as the rate. A thread's stack is in the cluster's local
memory and slower than a region: the rasterizer's heap of core loads took 355 us on the
stack and 163 us in a region. Pages that only the Tensix cores read and write belong in
interleaved tensors, not in the thread's bank: its counters took 247 us in bank 6 and
59 us interleaved (`projects/raster/`).

## Twelve harts

**Spawn and join, on one cluster.** `kmx_spawn(hart, fn, arg)` starts `fn(hart, arg)`
on hart 1, 2 or 3 and returns at once; `kmx_join(hart)` waits (bounded) until it has
returned. Hart 0 goes on meanwhile, with the ring calls, which stay hart 0's. A hart
still running its last spawn refuses another (`KMX_FULL`), and so does `kmx_par`,
which is spawn on harts 1 to 3, `fn(0, arg)` on hart 0, and join: one mechanism, 0.71
µs per empty call. Three clusters whose mains each spawn three are twelve independent
threads.

| Call | Hart 0 | A spawned hart, or harts 1 to 3 in a `kmx_par` or `kmx_par_all` |
|---|---|---|
| `kmx_args_get`, `kmx_region`, `kmx_cluster`, `kmx_clusters`, `kmx_lead`, `kmx_time`, `kmx_stop_requested` | yes | yes |
| `kmx_take`, `kmx_give`, `kmx_take_at`, `kmx_give_at`, `kmx_push` | yes: outside `kmx_par` spread over the four harts | yes, this hart alone |
| `kmx_spawn`, `kmx_join`, `kmx_par`, `kmx_par_all`, `kmx_serve`, `kmx_issue`, `kmx_wait`, `kmx_poll` | yes (not inside `kmx_par`'s `fn`) | no: the library error `par`, `KMX_STOPPED` |
| `kmx_progress`, `kmx_result` | yes | best not: one result block, no lock |
| `kmx_port_cores`, `kmx_port_owner`, `kmx_port_open`; `kmx_port_tell`, `kmx_port_asked`, `kmx_port_answer`, `kmx_port_told` ([X280 ports](#x280-ports)) | yes; the last four for the cores it serves | yes; the same |
| `kmx_port_run` | the lead's, as `kmx_par_all` | no: as `kmx_par_all` |

**All three clusters.** Put the same image on the clusters
(`prog.thread(clusters=(1, 2, 3), ...)`). The lowest, `kmx_lead()`, runs the program;
the others serve it:

```c
static void part(int cluster, int hart, uint64_t arg) {
    /* this cluster's share, from its own copy of the regions (kmx_region) */
}

int kmx_main(int hart) {
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : 2; /* until the lead's kmx_main returns */
    }
    /* push each cluster's input into its bank, then: */
    if (kmx_par_all(part, 0) != KMX_OK) {     /* all twelve harts, bounded */
        return 2;
    }
    return 0;                                 /* the others' kmx_serve returns */
}
```

`kmx_par_all(fn, arg)` writes the call into each other cluster's local memory through
a NoC window, runs `fn` on its own four harts, and waits for each cluster's done line
in its own local memory; the first call waits until every cluster serves. `arg` is a
64-bit word, the same on every cluster; data lives in each cluster's own copy of the
regions, which have the same ids.

**Pushing.** `kmx_push(cluster, id, offset, src, bytes)` stores into region `id` of
`cluster`'s thread (64-byte multiples), in posted 512-byte vector stores per 2 MiB
window, then the last line once more through a non-posted window, and returns when that
store is acknowledged (one hart's stores to one tile arrive in order). The receiver:

- holds no dirty line of the range when the push lands (it gave it, or never wrote
  it): a dirty line written back later would overwrite the pushed bytes;
- takes the range (`kmx_take_at`) before reading it.

The cluster that leads pushes once the others serve; the others push inside a
`kmx_par_all` call (`KMX_BAD` before the first). The source is read through the
cluster's cache, so a push from data in the bank moves at the bank's read rate, about
1.2 GB/s.

What it gives ([08_parallel_thread](../examples/08_parallel_thread/), `--twelve`,
1,000 MHz, best of 3, every output exact): eight and twelve harts include pushing the
input out and the outputs back.

| 262,144 words (1 MiB) | 1 hart | 4 | 8 | 12 |
|---|---|---|---|---|
| map, ms | 2.89 | 1.66 | 1.94 | 1.61 |
| map, `f` 16 times, ms | 26.44 | 6.68 | 4.46 | 3.38 |
| prefix sum, two passes, ms | 4.90 | 2.57 | 2.46 | 2.07 |

Twelve harts lose to four on small inputs: at 16,384 words the map takes 0.13 ms on
twelve and 0.11 on four, the prefix sum 0.18 and 0.13; work that only moves memory
gains little at any size (each cluster reads its bank at 1.2 GB/s and the input must
be pushed first), and work that computes gains most (the heavy map runs 2.0 times faster than on four).

## X280 ports

An **X280 port** joins one Tensix core and the X280 core that serves it, both ways,
with no dispatch kernel in the path: the X280 core stores into the Tensix core's
control block through a NoC window it holds for the run, and the Tensix core stores
into a mailbox line in the X280 core's local memory. A message is a sequence number and
three 32-bit words, landed as one 16-byte NoC write. Two kinds share that format, in
both directions:

- **tell**: the newest wins. The receiver looks without waiting (or waits, bounded) and
  sees only the newest tell since its last look; tells in between are lost.
- **ask and answer**: the Tensix core asks, the X280 core answers that ask by its
  sequence number, the Tensix core waits (bounded) for that answer.

The host declares the ports (`prog.ports()`); every served Tensix core has ONE X280
core, of one of the program's threads, for the whole run. By default the grid's Tensix
cores are dealt over the threads' X280 cores in grid order, cluster by cluster (ten or
so to an X280 core with three clusters); `prog.ports(serve={core: (cluster, cpu),
...})` names them instead (`cpu` the X280 core, 0 to 3). The threads must run on the
card.

On an X280 core (`kmx.h`), a call names a Tensix core the calling X280 core serves,
else `KMX_BAD`:

```c
static void serve(int cluster, int cpu, uint64_t arg) {
    while (kmx_port_open()) {                 /* until the record's event */
        for (int c = 0; c < kmx_port_cores(); c++) {
            kmx_msg m;
            if (kmx_port_owner(c) == cpu && kmx_port_asked(c, &m) == KMX_OK) {
                kmx_port_answer(c, m.seq, m.w[0] * 2, 0, 0);
            }
        }
    }
}

/* X280 core 0 of the lead: issue r, serve on all twelve X280 cores until its event */
kmx_event ev;
int e = kmx_port_run(&r, serve, 0, &ev);
```

`kmx_port_run` issues the record and runs `fn` on every X280 core of the group as
`kmx_par_all` does. X280 core 0 of the lead looks for the record's event in its ring
(every fourth `kmx_port_open`) and, when it comes, writes "ended" into every cluster's
ports line: each X280 core's next `kmx_port_open` returns 0, its `fn` returns, and once
all have, `kmx_port_run` returns the event. Nothing is stored into a Tensix core after
that; a record that times out ends the serve the same way (its event is
`KM_EV_TIMEOUT`), and a stop or the deadline also ends it (`KMX_STOPPED`, `KMX_LATE`).
On the Tensix core (`komorebi/device/km_x280.hpp`):

```cpp
km::Reader rd;
km::X280Port port(rd);  // the reader owns the port
rd.run([&](const km_record_t& r) {
    km::X280Msg m;
    if (port.call(r.a0, 0, 0, m)) {  // ask, wait for the answer
        port.tell(m.w[0], 0, 0);     // tell the X280 core, never waits
    }
});
```

**Costs** (one Blackhole p150a, AICLK 1350 MHz, X280 1000 MHz;
`experiments/x280_ports/`): an ask is answered in 0.94 µs with one Tensix core on its
X280 core; with ten each, 9.1 µs at full load and 13.1 M asks a second over twelve X280
cores. The X280 core's loop bounds it, about 0.9 µs a served message, not the NoC (NoC 1
measured the same). Its image, data and stack live in its local memory, 30 cycles a
word, uncached: a look at a mailbox line costs 259 cycles, a tell 144. Example
[10_x280_ports](../examples/10_x280_ports/) runs it on 1, 4 and 12 X280 cores.

**Rules.**
- One kernel of a Tensix core sends on its port (the reader by default; the writer by
  `km::X280Port(wr, km::kStWriter)`); never both. Any kernel may look.
- An X280 core sends only to the Tensix cores it serves; the windows are opened at the
  start and cleared at the end of the run, one per served core (a cluster holds up to
  168).
- Tell a Tensix core only while it runs a record that looks for it: a tell stays in the
  line, and the core's next look sees it if it is newer than the last one it saw.
- Every wait on the Tensix core is bounded by the wait limit and watches the stop word
  (`kWhereX280Answer`, `kWhereX280Tell`, `kWhereX280Send`; `kWhereX280None` when the
  program gave the core no X280 core).
- The control block grew by one 64-byte area (`KM_CTL_X280`) for every program; the
  ports area takes 8.6 KiB of each serving cluster's local memory after its block.

### The pull

When the work of a command is uneven, equal ranges leave the cores with the light part
waiting for the slowest. The **pull** hands the work out at run time instead: units 0
to `units - 1` in ranges of `per` units, each range to the first Tensix core that asks,
each range once. On X280 core 0 of the lead:

```c
kmx_pull p = {.units = 31000, .per = 16, .mode = KMX_PULL_COUNT};
kmx_event ev;
int e = kmx_pull_run(&c, &p, &ev); /* submits command c, serves its pulls until its event */
```

and on the Tensix core, in the command's handler:

```cpp
uint32_t first, count;
while ((count = port.next(first)) != 0) {  // a range, or 0: none left (or gave up)
    for (uint32_t u = first; u < first + count; u++) {
        work(u);
    }
}
```

`next()` asks for the following range as soon as it returns one, so the answer comes
while the range runs: one ask is in flight, and a core waits only when its range takes
less than a round trip. Each cluster of the group keeps a counter in its local memory
(an atomic add): cluster `i` of `n` hands out ranges `i`, `i + n`, `i + 2n`.., so no
counter is shared across clusters. The modes:

- `KMX_PULL_COUNT`: the counter, or `pick(core, k, arg)` when set: the `k`-th range of
  `core` (or `KMX_PULL_NONE`), to hand out by the asker, for example by its cost.
- `KMX_PULL_LOG`: the same, and the ranges each core got are written in order to region
  `log` (`KMX_PULL_LOG_BYTES(per_core)`: a count per core, then `per_core` range
  numbers per core; each cluster its own copy). `KMX_BAD` when the command ran but a
  core got more than the log holds.
- `KMX_PULL_REPLAY`: each core gets the ranges the log holds for it, in the same order:
  a second pass over the same units on the same cores (a count, then the work it
  counted). Use the same ports and clusters as the logged run.

The range is the unit of balance: size it so that one range is worth well over the
round trip (a pull every 200 µs or more costs under 1 %). **Costs**: example
[10_x280_ports](../examples/10_x280_ports/) `--pull`, 31,000 units of 15 to 27 µs on 120
Tensix cores, slowest core over mean 1.54 with equal ranges: pulled in ranges of 16
units (278 µs) the pass takes 4.73 ms against 6.96 ms, slowest over mean 1.045, on 1, 4
or 12 serving X280 cores alike (a core waits 0.7 µs a pull on twelve, 2.4 µs on one).
With units of 1 µs and one unit a range, the pull loses: 1.51 ms against 0.48 ms on
twelve X280 cores, 16.5 ms on one.

## From the host

```python
import komorebi as km

prog = km.Program("frame", kernels="kernels/")
prog.thread(
    cluster=1,  # 1, 2 or 3; clusters=(1, 2, 3) declares the same thread on each
    sources=["thread/frame_driver.c"],
    regions={"counts": counts, "tables": 1 << 20},  # an array (initial bytes) or a size
    words=[cores, bins],
    host_fallback=frame_driver,  # the same function in Python
    deadline=5.0,  # seconds, for the whole thread
)
with km.Session() as session:
    run = session.load(prog)  # grid, dispatch kernel, then the thread
    result = run.thread_result(timeout=10.0)  # waits, stops it, reads its result block
    run.stop()
    tables = run.thread.read("tables")  # after the stop
```

- `Program.thread(...)` declares the thread (one per cluster). Its sources are built
  with the kmx library by the dispatch kernel's toolchain, once, into the build cache.
  With `clusters=(1, 2, 3)` it declares the same thread on each and returns the list;
  each thread has its own copy of its regions in its own bank, and its own words.
- `session.load(prog)` writes the regions before it launches the grid (with slow
  dispatch a tensor write waits for every core to be idle, which a resident grid never
  is), then starts the thread once the dispatch kernel is ready. The regions are pages
  of one DRAM tensor per bank in use; arrays given as initial contents are written
  there.
- `run.thread_result(timeout)` returns a `ThreadResult`: `ran` (false when the thread
  did not return in time and was stopped), `state`, the library `error`, the thread's
  `code` and `words`, the last progress stamp `where` and its time `where_us`, and the
  thread's time `us`. A thread that does not return within 1 s of the stop word either
  is `stuck`, and the call raises `ThreadFailed` that says so, with the runtime's
  account of the stop in `detail` when the cluster could not be stopped (see Limits).
- `run.thread.read(name)` returns a region's bytes once `run.stop()` has returned (the
  thread writes its regions until it stops; an earlier read raises).
  `run.thread.regions.address(name)` gives `(bank, bank address)` for a Tensix kernel,
  which reaches the region with `get_noc_addr_from_bank_id<true>(bank, address)`.
- `run.stop(timeout)` stops every thread still running before it stops the kernel,
  inside the same deadline.

**Several threads.** `run.threads` holds each thread's `ThreadRun` by cluster
(`run.thread` is the lowest cluster's). `run.thread_results(timeout)` waits for all and
returns `{cluster: ThreadResult}`; when one does not return in time or ends with a
library error, the others are asked to stop at once and `ThreadFailed` is raised with
the failing `cluster` and every thread's `results`. A shared region:

```python
prog.region("data", 1 << 20)  # a size or an array of initial contents
prog.thread(
    clusters=(2, 3), sources=["thread/pass.c"], regions={"own": 4096}, shared=["data"]
)  # region ids: own regions first, then shared ones
```

`run.threads[2].read("data")` and `run.threads[3].read("data")` read the same pages.

**The host fallback.** `session.load(prog, thread="host")` loads the same program
without the image: `run.thread_result()` then calls
`host_fallback(regions, words, run) -> (code, words)` with each region as a `uint32`
array holding its initial contents (zeros for a region given as a size) and the run's
`Run`, through which it issues the records the thread would issue through its ring
(`run.issue(..., result=True)`, `run.wait`); `run.thread.read` then returns those
arrays as the fallback left them. A thread is developed this way first, and its card
result is checked against it bit for bit. A thread whose decision is a plain-C file
can give the fallback the same file through ctypes, so both run one program
([07_work_cluster](../examples/07_work_cluster/)).

## Limits

- The prefetcher registers keep their values until a card reset: once a thread set
  them, every later image on that cluster runs with them, whatever it asks. The
  cluster's L3 stays as the runtime sets it (one 128 KiB cache way); the library
  never changes it.
- One thread per work cluster, three at most. A thread's regions and the shared regions
  it names are at most 16 in all. Threads on banks 6 and 7 share data only by pushing
  it.
- `kmx_par_all` runs on the clusters that run the same image, led by the lowest; every
  other one must call `kmx_serve`, or the lead's first `kmx_par_all` waits until its
  deadline.
- The stop word and the deadline are seen only inside library calls (`kmx_wait`,
  `kmx_par`, `kmx_stop_requested`, ...). A hart stuck outside them, in a loop that
  calls none or in a load that never completes, cannot be stopped: the host's wait
  raises `ThreadFailed` (`stuck`, the last progress stamp, the runtime's account), the
  runtime marks the cluster unhealthy, the other clusters go on until the process
  closes the card, and the next session needs a card reset. A hart that traps is
  halted by the runtime's handler; its account names `mcause`, `mepc` and `mtval`.
- A 64-bit load or store into a region uses an offset that is a multiple of 8, a
  32-bit one a multiple of 4: an access across that boundary is not emulated (the
  runtime's trap handler halts the hart), and one such load into a region was seen
  to stall the hart with no trap at all. `kmx_region` returns 64-byte aligned memory.
