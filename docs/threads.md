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
dispatch kernel on X280 cluster 0, and, when the program needs one, an **X280 thread**
on a work cluster (cluster 1). The thread is the irregular part of the program: a
decision over every core's data (prefix sums over all cores' counts, a cost split),
work whose size is not known ahead, the program's own control loop. It is ordinary C
on the cluster's four coherent 64-bit cores. It does not compute on tiles.

Status: a thread can be declared, built, loaded beside the dispatch kernel, read and
write its regions, run on all four cores, return a result, and be replaced by its host
fallback, and issue records to the grid through its command ring to the dispatch
kernel (`kmx_issue`, `kmx_wait`, `kmx_poll`), which the session sets up at load.

## What a thread can touch

| Thing | Access |
|---|---|
| its argument block | read: the region table and the program's own words |
| its regions | read and write, as memory: pages of the work cluster's DRAM bank, which Tensix cores reach through the NoC |
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
about 50 microseconds; the kernel's loop looks at the ring every fourth pass.

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
knows), then calls `kmx_main` on hart 0. Harts 1 to 3 wait for `kmx_par` calls until
`kmx_main` returns. Every call is listed in the
[reference](reference/threads.md).

Rules. The runtime's stop is a word the cores must poll: a core that never polls it
never returns, its cluster stays stuck, and only a reset of the whole card brings it
back. That is why every loop in a thread is bounded:

- **Every loop ends by itself.** Bound it by a count, or poll `kmx_stop_requested()`,
  which is true once the host or the runtime asked for a stop or the thread's deadline
  passed. The library's own waits (`kmx_par`) are bounded by that deadline.
- **Pointers come from `kmx_region`.** Write nowhere else. Globals need a non-zero
  initialiser and `__attribute__((section(".data")))` (the image has no `.bss`).
- **No libc.** The image links without one; a struct assignment can become a `memset`
  call that does not link.
- **`kmx_progress(where)`** leaves a stamp the host reads with the result, also after a
  timeout: say where the thread is before and after each step.
- **The image is at most 126,976 bytes** (the part of the cluster's memory that
  instructions can be fetched from); globals count.

`kmx_time()` counts the 50 MHz timer (`KMX_TICKS_PER_SECOND`).

What it costs: the cluster reads its bank at about 1.3 GB/s and writes it at about
1.3 to 1.8 GB/s with four cores, and a hand-over sweep runs at 5 to 8 GB/s of region.
Prefix sums over 119 x 1,024 counters take about 1.7 ms on four cores (a decision per
frame, not per segment); see `experiments/x280_threads/`.

## From the host

```python
import komorebi as km

prog = km.Program("frame", kernels="kernels/")
prog.thread(
    cluster=1,
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

- `Program.thread(...)` declares the thread (one per program). Its sources are built
  with the kmx library by the dispatch kernel's toolchain, once, into the build cache.
- `session.load(prog)` writes the regions before it launches the grid (with slow
  dispatch a tensor write waits for every core to be idle, which a resident grid never
  is), then starts the thread once the dispatch kernel is ready. The regions are pages
  of one DRAM tensor in the work cluster's bank; arrays given as initial contents are
  written there.
- `run.thread_result(timeout)` returns a `ThreadResult`: `ran` (false when the thread
  did not return in time and was stopped), `state`, the library `error`, the thread's
  `code` and `words`, the last progress stamp `where` and its time `where_us`, and the
  thread's time `us`.
- `run.thread.read(name)` returns a region's bytes once `run.stop()` has returned (the
  thread writes its regions until it stops; an earlier read raises).
  `run.thread.regions.address(name)` gives `(bank, bank address)` for a Tensix kernel,
  which reaches the region with `get_noc_addr_from_bank_id<true>(bank, address)`.
- `run.stop(timeout)` stops a thread still running before it stops the kernel, inside
  the same deadline.

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
- One work cluster per program, cluster 1. Its DRAM bank (bank 6 on the p150a) was
  measured; a wrong bank fails the start-up check with the error `bank`.
- A thread that never returns and polls nothing cannot be stopped: the runtime marks
  the cluster unhealthy, the other clusters go on until the process closes the card,
  and the next process needs a card reset.
