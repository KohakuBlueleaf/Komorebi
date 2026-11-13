# Komorebi

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/teaser-dark.svg">
  <img src="docs/images/teaser-light.svg" width="1280" alt="Komorebi in one figure. Left, what you write: a compute kernel that runs on every Tensix core and picks a handler per command, a C thread on the X280 that submits the passes and calls kmx_par, and the Python lines that declare the program and its thread. Centre, the card runs it by itself: two panels of 6 by 10 Tensix cores with persistent kernels, X280 cluster 0 submitting commands to both panels, X280 clusters 1 to 3 running the program's threads, values crossing core to core; below it one rasterizer frame of a million triangles over time: the host only before and after, the X280 thread's 0.80 ms step between passes, seven commands, the Tensix passes COUNT 3.8, SETUP 5.1 and FILL 2.1 ms, 12.1 ms in all against 4.4 ms on a V100. Right, what comes out: Sponza and Bistro rasterized (7.2 and 15.4 ms) and ray traced (360 and 597 ms) on the card, and four neural programs against TT-NN on the same card: matrix products 6.3 us (7.7 times) and 7.99 ms (1.5 times), SwiGLU 42.6 us (6.2 times), attention 163 us (1.3 times).">
</picture>

A programming model and framework for the Tensix grid and the X280 cores of a
Tenstorrent Blackhole card (developed on a P150A). A program is a set of persistent kernels on every Tensix core; each command they receive names a handler of code and a
core group, and every member works on its share. Values cross between cores, a
dispatcher on one X280 cluster submits the commands and runs streams and graphs of
them on the card, and C threads on the other three X280 clusters run the irregular part
of a program.

**Status: early development; nothing is released.** Everything below runs on one
P150A, from a clone, against a tt-metal source tree.

## At a glance

One P150A, 2026-11-13; every number is a row of [Measured](#measured), from the
program's page and its committed results.

| Program | Shape | Komorebi | Reference | Reference / Komorebi |
|---|---|---|---|---|
| Matrix product | 128 x 896 x 128 | 6.3 µs | Kohaku-TT, same card: 28.7 µs | 4.58 |
| Matrix product | 8192 cubed | 7985.9 µs | Kohaku-TT, same card: 11797.6 µs | 1.48 |
| Matrix product | 16,384 cubed | 84018.0 µs | Kohaku-TT, same card: 75522.5 µs | 0.90 |
| SwiGLU | 256 x 256 x 512 | 42.6 µs | Kohaku-TT, same card: 181.6 µs | 4.26 |
| SwiGLU | 16,384 x 384 x 384 | 161.7 µs | Kohaku-TT, same card: 458.1 µs | 2.83 |
| Attention | 2 x 8 x 1024 x 128, 2 K / V heads | 163.3 µs | Kohaku-TT, same card: 261.8 µs | 1.60 |
| Attention | 1 x 8 x 1024 x 64, causal | 73.6 µs | Kohaku-TT, same card: 53.7 µs | 0.73 |
| Rasterizer, whole frame | 10^6 triangles, 1024 x 1024 | 12.13 ms | CUDA on a V100: 4.42 ms | 0.36 |
| Ray tracer, whole frame | Sponza close, 1024 x 1024 | 359.86 ms | CUDA on a V100: 102.00 ms | 0.28 |

Above 1, Komorebi takes less time. The neural workloads are on par with or ahead of
Kohaku-TT on most shapes; the rasterizer and the ray tracer run whole frames on the
card, bit-exact against numpy, today at a third of the V100's speed (the rasterizer)
and a third or less (the ray tracer).

## A kernel

The merge-tree example adds up a job's input tiles over a core group. Each member
runs three kernels: the reader brings the member's share of the tiles from DRAM into
L1 and waits for the parts other members send it; the compute kernel below adds them;
the writer sends the member's part to its receiver, or, on member 0, writes the total.
The parts meet either at one owner or along a tree
([02_merge_tree/kernels/compute.cpp](examples/02_merge_tree/kernels/compute.cpp),
lines 36 to 81; above them are the includes and two helpers: `add_page` adds one tile
of a buffer into a sum in DST, `clear_sums` sets the sums to zero):

```cpp
void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    // The runtime's loop: the body runs once per command whose group holds this core.
    cp.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerMerge) {  // this program answers MERGE commands only
            return;
        }
        const uint32_t tiles = r.a1;  // input tiles T of the job
        const uint32_t form = r.a4;   // 0: merge at one owner, 1: along a tree
        const uint32_t part = r.a7;   // tiles per part P (1 to 3), one sum each in DST
        // The group is the command's range of cores; my rank in it gives my share of T.
        const km::Group g = km::group_of(r, cp.me());
        const km::Partition s = km::partition(tiles, g.rank, g.size);
        const km::Arrival arrival = {kCbArrive, kArrivePages, part};
        tile_regs_acquire();  // DST, the vector unit's registers, until the release
        clear_sums(part);     // the P sums start at zero
        // My own tiles, as the reader brings them from DRAM: tile t into sum t mod P.
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t i = 0; ok && i < s.len; i++) {
            ok = cp.wait_front(kCbIn, 1);
            if (ok) {
                add_page(kCbIn, 0, (s.start + i) % part);
                cb_pop_front(kCbIn, 1);
            }
        }
        // The crossing: wait for the parts other members send me, add them in rank
        // order (by owner: member 0 takes all; by tree: one partner a round).
        ok = ok && km::reduce(cp, arrival, g, form, [&](uint32_t k) {
                 for (uint32_t d = 0; d < part; d++) {
                     add_page(kCbArrive, km::slot_page(arrival, k) + d, d);
                 }
             });
        tile_regs_commit();
        tile_regs_wait();
        // The P sums to the writer, which sends them on or (member 0) writes the total.
        // After a give-up (a wait ran out, or the run was stopped) nothing is packed.
        if (ok && cp.reserve_back(kCbOut, part)) {
            for (uint32_t d = 0; d < part; d++) {
                pack_tile(d, kCbOut);
            }
            cb_push_back(kCbOut, part);
        }
        tile_regs_release();
    });
}
```

The host declares the program (`program()` in
[merge.py](examples/02_merge_tree/merge.py): the three kernels, three circular buffers,
the handler header) and submits one `MERGE` command per job to its group (`launch()`,
lines 103 to 107 and 123 to 127):

```python
run = session.load(
    program(args, inputs[0], out),
    deadline=args.deadline if stall < 0 else 4 * args.wait,
    wait=args.wait,
)
# ... the command words of job j (lines 108 to 122)
group = (first, count)
handles.append(run.submit("merge", group, words, exclusive=True, result=True))
log(f"{run.drain(timeout=30.0)} events")
host_s = time.perf_counter() - t0
run.stop()
```

It checks every total against numpy and prints one JSON line with `ok`: 1,000 jobs on
random groups, 0 wrong; 36.7 µs a job on 8 cores and 51.8 µs on 119, each command
returning a checksum of its total
([02_merge_tree](examples/02_merge_tree/README.md)).

The irregular part of a program is C on an X280 work cluster, where `kmx_par` runs a
function on the cluster's four cores (cores). A two-pass prefix sum over chunks of the
input, 2.48 ms for 262,144 words on four cores, exact against numpy
([08_parallel_thread/thread/parallel.c](examples/08_parallel_thread/thread/parallel.c),
lines 148 to 198 without `nothing`, an empty function for timing `kmx_par`; the job
struct and the chunk helpers are above them):

```c
/* (c) prefix, pass 1: each chunk's sum. */
static void chunk_sums(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    /* my chunks of CHUNK words: core, core + parts, ... (parts: the cores in use) */
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        uint32_t s = 0;
        for (uint64_t i = c * CHUNK; i < end(j, c); i++) {
            s += j->in[i];
        }
        g_chunk[c] = s;
    }
}

/* Pass 2: out[i] = the chunk's base + the chunk's words before i. */
static void chunk_prefix(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        uint32_t run = g_chunk[c];
        for (uint64_t i = c * CHUNK; i < end(j, c); i++) {
            j->out[i] = run;
            run += j->in[i];
        }
    }
}

/* Runs fn on core 0 alone or on all four; KMX_OK, or kmx_par's error. */
static int on(void (*fn)(int, void*), job_t* j) {
    if (j->parts == 1) {
        fn(0, j);
        return KMX_OK;
    }
    return kmx_par(fn, j);
}

/* The two-pass exclusive prefix sum (mod 2^32); between the passes core 0 turns the
 * chunk sums into bases. */
static int prefix(job_t* j) {
    const int e = on(chunk_sums, j);
    uint32_t base = 0;
    for (uint64_t c = 0; c < chunks(j); c++) {
        const uint32_t s = g_chunk[c];
        g_chunk[c] = base;
        base += s;
    }
    return e != KMX_OK ? e : on(chunk_prefix, j);
}
```

## The programming model

### The card as Komorebi runs it

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/card-dark.svg">
  <img src="docs/images/card-light.svg" width="830" alt="The P150A tile grid in NoC0 coordinates, 17 columns by 12 rows: DRAM tiles in columns 0 and 9, Ethernet in row 1, PCIe and ARC in row 0, the security tile and four X280 tiles in column 8, Tensix cores in columns 1 to 6 and 11 to 16 of rows 2 to 11 (columns 7 and 10 fused off on this card), each numbered with its Komorebi core index 0 to 119. The host declares the program and its thread and submits commands; commands and events pass through PCIe to X280 0 at (8, 3), which runs the dispatcher. Its command goes out to two panels, columns 1 to 6 and 11 to 16, one multicast window each. A core group 26 to 55 is filled; its leader, core 26 at (3, 4), acknowledges on NoC1. X280 1 at (8, 9), X280 2 at (8, 5) and X280 3 at (8, 7), drawn in their own colour, each run a C thread of the program on four cores (kmx_main, kmx_par); a ring joins each to X280 0, commands going to the dispatcher and events with result words coming back, so commands have up to four sources. Each cluster's regions lie in the DRAM bank beside it, D6 for X280 1 and D7 for X280 2 and 3, which Tensix cores reach over the NoC; kmx_invalidate and kmx_flush hand them over. Boxes below list what the dispatcher, each Tensix core and the work clusters do.">
</picture>

The grid is the card's real one in NoC0 coordinates; the numbers in the Tensix cores
are Komorebi's core index. A **command** is one 64-byte line: the handler, the group
(a contiguous range of cores, whose first member is its **leader**) and ten 32-bit
argument words. Every core sees every command and skips those whose group does not hold
it; members report to the leader, which acknowledges to the dispatcher, with a
result word when the command asks for one.

### A thread has a program counter and a place

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/breakpoints-dark.svg">
  <img src="docs/images/breakpoints-light.svg" width="830" alt="Two listings of real code with line numbers. First, a dynamic handler from examples/05_forward/kernels/compute.cpp: while (km::Packet pk = dy.next()) takes the items that landed on this core; for each item the generator advances and is added to the sum; when no hops are left dy.retire() ends the item there; otherwise dy.hop(...) sends its 64 bytes to the member its state names; dy.done(pk) frees the packet. The next, retire and hop lines are marked. Second, the merge tree from examples/02_merge_tree/kernels: the writer's km::reduce_emit sends a member's part into a fixed slot (marked emit), the reader's km::reduce_arrive waits for the senders' parts (marked wait), and the compute kernel's km::reduce adds them in rank order. Below, on a schematic grid of cores, two programs run one at a time: one item starts on a core, hops twice to the core its state names and retires there; and in a group of four merging by owner, ranks 1 to 3 emit their parts to rank 0, which waits for them and adds them.">
</picture>

A **breakpoint** is a point where a thread must deal with another core; between two
breakpoints it runs on its core. Today a program is written as the handler bodies of
its kernels, as above; the thread language in the first column is planned:

| Planned thread language | What crosses | On the card today |
|---|---|---|
| `at X[i]: block` | the thread's state | a dynamic handler: the state as a 64-byte item, `dy.hop` ([dynamic handlers](docs/reference/dynamic.md), [05_forward](examples/05_forward/)) |
| `emit v -> thread(j)` | the value | `km::emit` into a fixed slot, reduce by owner or by tree ([ports](docs/reference/ports.md), [02_merge_tree](examples/02_merge_tree/)) |
| `v = wait(n)` | nothing | `km::arrive`: a bounded wait until `n` values have landed |
| a block to a row or a column | the block, once per broadcast | broadcast broadcast ([broadcasts](docs/reference/broadcast.md), [03_broadcast](examples/03_broadcast/)) |

### Groups and commands in time

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/groups-dark.svg">
  <img src="docs/images/groups-light.svg" width="830" alt="Time runs left to right over cores 0 to 11, the first row of the grid (NoC0 row 2, columns 1 to 6 in panel 0 and 11 to 16 in panel 1). The dispatcher submits R1 to cores 0 to 4 and R2 to cores 6 to 11; each command lands on every core, the members work at the same time, report to their leader, and each leader acknowledges its own command, so R1 and R2 are in flight together. R3, for cores 3 to 8, overlaps both groups. In the drawn case it lands after cores 3, 4 and 6 to 8 have begun R1 or R2, so they take it next, while core 5, in neither group, takes it at once; each member's report ends on the broadcast of the leader, core 3, which acknowledges after the last report. A note says this order is the drawn case, not a guarantee: a shared core may run two commands in flight in either order unless the later one is submitted exclusive.">
</picture>

Up to 16 commands are in flight at once. A command that must run after an earlier one
on overlapping cores is submitted with `run.submit(..., exclusive=True)`.

### One program on the grid

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mapcollect-dark.svg">
  <img src="docs/images/mapcollect-light.svg" width="830" alt="The map/collect example on the real tile grid: the core group 0 to 7 sits at NoC0 (1, 2) to (6, 2), (11, 2) and (12, 2); core 3 at (4, 2) is highlighted. A table lists each member's share: core m reads tiles 8m to 8m+7 and writes output page a3 + m. Members report to core 0, which acknowledges to X280 0. Inside core 3: the reader reads tiles 24 to 31 from DRAM on NoC0 into circular buffer 0 in L1, compute copies each tile to DST and the vector unit adds them in fp32, pack writes the sum to circular buffer 16, the writer writes it to DRAM on NoC1, and the core reports to core 0.">
</picture>

## What exists

| Part | On main | Page | Example |
|---|---|---|---|
| Persistent kernel: commands, 16 mailbox slots per core, the reader, compute and writer loops, every wait bounded, the trace | implemented | [Tensix runtime](docs/reference/runtime.md) | [01_mapcollect](examples/01_mapcollect/) |
| Groups per command, several in flight, completion per group, a result word per command | implemented | [Tensix runtime](docs/concepts/runtime.md) | [01_mapcollect](examples/01_mapcollect/) |
| Static crossings: `emit`, `arrive`, `take`, groups, reduce by owner and by tree | implemented | [ports](docs/reference/ports.md) | [02_merge_tree](examples/02_merge_tree/) |
| Static crossings: broadcast broadcast along a row or a column, two broadcasts at once | implemented | [broadcasts](docs/reference/broadcast.md) | [03_broadcast](examples/03_broadcast/) |
| Dynamic crossings: items through per-core queues in DRAM, a spawn quota, the quiet end found by the dispatcher | implemented; one dynamic run at a time | [dynamic handlers](docs/reference/dynamic.md) | [05_forward](examples/05_forward/) |
| The dispatcher on X280 cluster 0: host ring, slots, deadlines, events | implemented | [dispatcher](src/komorebi_native/x280/kernel/README.md) | all |
| Micro-op streams and the job queue in the dispatcher | implemented; builder, check and simulator on the CPU | [streams](docs/streams.md) | [04_uop_stream](examples/04_uop_stream/) |
| Graphs in the dispatcher: command nodes, CPU-task nodes, joins, a kept graph run again | implemented; also host-fed | [graphs](docs/graphs.md) | [06_graph](examples/06_graph/) |
| X280 threads: `kmx_par`, `kmx_spawn` / `kmx_join`, regions, the command ring, a host fallback in Python; a thread on each of work clusters 1 to 3, regions shared on one bank; `kmx_par_all` over twelve cores and `kmx_push` across banks | implemented | [X280 threads](docs/threads.md) | [07_work_cluster](examples/07_work_cluster/), [08_parallel_thread](examples/08_parallel_thread/) |
| The host package: `Program`, `Session`, `Run`, uploads and engine events, reads by page range, errors that name the core | implemented | [host API](docs/reference/host.md) | all |
| Hardening: the kernel instruction check, an optimisation level per kernel, `Session(kernel_bytes=...)`, a bounded exit | implemented | [host API](docs/reference/host.md#instructions-the-cores-lack) | |
| The thread language; homes declared | planned | [manifest](src/komorebi/manifest/README.md), [lang](src/komorebi/lang/README.md) | |

## Measured

Whole programs on one P150A, every row from the program's page and its committed
results, all dated 2026-11-13: AICLK 1,350 MHz nominal (the card lowers it under its
power limit in the long matrix loops; each page gives the clock it sampled) and the
X280 at the session's 1,000 MHz. Power is the whole card's mean over the loop.

### Rasterizer and ray tracer: whole frames

Both programs run the whole frame on the card, 1024 x 1024, and both are far below
half of a V100's speed today. The V100 and C++ columns are the reference's whole
frame (the same algorithm in CUDA on one idle Tesla V100 32 GB, and in C++ on a shared
96-thread AMD EPYC 7K62); none of the three counts the scene's upload or the image's
read-back.

The **rasterizer**: the passes and the counting run on the 120 cores, the serial rest of
the step between them on a thread on the X280 work cluster; the image is bit-exact
against numpy. Median of 5 loads, each about 1 s of frames; card 1
([raster](projects/raster/README.md#the-frame-on-the-work-clusters-thread), results in
`projects/raster/results/`,
[reference](projects/raster/reference/README.md#timings-and-power)):

| Scene | Triangles | Frame | Card frame, ms | of which the X280 step | Card W | C++ 96 threads, ms | V100, ms | V100 / card |
|---|---|---|---|---|---|---|---|---|
| tiny | 398 | first | 1.051 | 0.436 | 64.0 | 2.66 | 0.29 | 0.28 |
| tiny | 398 | repeated | 1.051 | 0.436 | 61.8 | 2.66 | 0.29 | 0.28 |
| 10^4 | 11,994 | first | 1.091 | 0.440 | 65.8 | 3.32 | 0.25 | 0.23 |
| 10^4 | 11,994 | repeated | 1.041 | 0.453 | 65.4 | 3.32 | 0.25 | 0.24 |
| 10^5 | 102,308 | first | 2.555 | 0.751 | 67.2 | 4.08 | 0.79 | 0.31 |
| 10^5 | 102,308 | repeated | 2.454 | 0.771 | 65.9 | 4.08 | 0.79 | 0.32 |
| 10^6 | 997,806 | first | 12.125 | 0.804 | 67.7 | 17.87 | 4.42 | 0.36 |
| 10^6 | 997,806 | repeated | 10.575 | 0.823 | 67.3 | 17.87 | 4.42 | 0.42 |
| Sponza, overview | 262,266 | first | 7.223 | 0.456 | 65.7 | 7.88 | 2.31 | 0.32 |
| Sponza, overview | 262,266 | repeated | 5.285 | 0.466 | 66.1 | 7.88 | 2.31 | 0.44 |
| Bistro interior, overview | 1,020,903 | first | 15.373 | 0.733 | 65.3 | 14.10 | 5.99 | 0.39 |
| Bistro interior, overview | 1,020,903 | repeated | 10.974 | 0.752 | 65.2 | 14.10 | 5.99 | 0.55 |

Where the time goes: SETUP, the scatter of each triangle's record into its tiles' lists,
is the largest part of every frame from 10^5 up (cold frame: 10^6 5.1 ms, Bistro 7.5
ms), then COUNT (3.8 and 4.8 ms) and the fill (2.1 ms). "First" is a frame without
history; "repeated" is a later frame of the same camera, whose COUNT and SETUP ranges are
cut by the frame before's costs (10^6 3.9 and 3.4 ms, Bistro 4.3 and 3.6 ms).

The **ray tracer**: ray generation, four traces and shading are eight commands on the
120 cores; trees larger than L1 are read from DRAM as treelets. Card 1. A cold frame
has no history and is submitted by the host. A warm frame is the second or third of the
same camera, submitted by a thread on the X280. Before each trace the thread cuts the
rays into the cores' ranges by the frame before's costs; its plan time counts in the frame
([raytrace](projects/raytrace/README.md#the-frame-planned-on-the-x280)):

| Scene | Frame | Card frame, ms | of which the X280 plan | Card W | C++ all threads, ms | V100, ms | V100 / card | Pixels differing from numpy |
|---|---|---|---|---|---|---|---|---|
| tiny (tree in L1) | cold and warm | 23.20 | - | 63.0 | 20.59 | 2.59 | 0.11 | 0 |
| 10^4 | cold | 57.91 | - | 63.0 | 26.44 | 4.16 | 0.07 | 0 |
| 10^4 | warm | 70.63 | 5.27 | 61.0 | 26.44 | 4.16 | 0.06 | 0 |
| 10^5 | cold | 81.51 | - | 63.0 | 35.47 | 6.57 | 0.08 | 0 |
| 10^5 | warm | 93.01 | 5.34 | 66.9 | 35.47 | 6.57 | 0.07 | 0 |
| 10^6 | cold | 154.21 | - | 63.0 | 53.66 | 11.61 | 0.08 | 1 |
| 10^6 | warm | 192.50 | 5.32 | 70.4 | 53.66 | 11.61 | 0.06 | 1 |
| Sponza, close | cold | 359.86 | - | 63.0 | 264.78 | 102.00 | 0.28 | 244 |
| Sponza, close | warm | 323.38 | 3.64 | 71.0 | 264.78 | 102.00 | 0.32 | 244 |
| Sponza, overview | cold | 348.53 | - | 69.2 | 230.36 | 95.52 | 0.27 | 234 |
| Sponza, overview | warm | 313.63 | 4.14 | 68.2 | 230.36 | 95.52 | 0.30 | 234 |
| Bistro interior, close | cold | 596.97 | - | 61.0 | 401.87 | 211.42 | 0.35 | 2 |
| Bistro interior, close | warm | 596.86 | 4.35 | 70.0 | 401.87 | 211.42 | 0.35 | 3 |
| Bistro interior, overview | cold | 523.72 | - | 73.1 | 336.80 | 153.94 | 0.29 | 2 |
| Bistro interior, overview | warm | 477.48 | 4.57 | 70.5 | 336.80 | 153.94 | 0.32 | 2 |

Where the time goes: the four traces take 67 % (10^4) to 97 % (Bistro) of the frame,
the bounce trace alone 36 to 59 %; each core sorts its bounce and second-shadow rays
(about 15 ms a frame) and reads the treelets its rays need from DRAM. In the traces,
box tests on packets with few live broadcasts come first (61 % on Sponza's overview), then
the wait for each command's slowest core. The differing pixels are ties between two
triangles at the same float32 distance, but for one or two on each Bistro view. The
plan by the frame before takes 35 to 46 ms off Sponza and Bistro's overview, nothing off
Bistro's close view, and costs 12 to 38 ms on 10^4 to 10^6: it cuts the traces into up
to 8 commands, and the cores read more treelets.

<img src="docs/images/raster-sponza.jpg" width="256" alt="Crytek Sponza, overview, rasterized on the card"> <img src="docs/images/raster-bistro.jpg" width="256" alt="Amazon Lumberyard Bistro interior, overview, rasterized on the card">
<img src="docs/images/raytrace-sponza.jpg" width="256" alt="Crytek Sponza, close view, ray traced on the card with shadows and one diffuse bounce"> <img src="docs/images/raytrace-bistro.jpg" width="256" alt="Amazon Lumberyard Bistro interior, close view, ray traced on the card with shadows and one diffuse bounce">

*Read back from the card, 512 x 512 downscales of 1024 x 1024 frames. Top: rasterized,
both bit-exact against numpy, Sponza overview (7.22 ms a cold frame above) and Bistro
interior overview (15.37 ms). Bottom: ray traced, Sponza close (359.86 ms a frame
above) and Bistro interior close (596.97 ms). Crytek Sponza, (c) 2010 Frank Meinl,
Crytek, CC BY 3.0; Amazon Lumberyard Bistro, (c) 2017 Amazon Lumberyard, CC BY 4.0
(Open Research Content Archive); both from the McGuire Computer Graphics Archive.*

### Matrix product, attention and SwiGLU against Kohaku-TT

The same card, ours and Kohaku-TT (at ac81ff2) one after the other in one session;
bf16 operands, fp32 accumulation; time per call in a loop of calls, uploads and
read-back outside it. "Kohaku-TT / ours" above 1 means ours is faster.

[Matrix product](projects/dense/README.md#results), HiFi2, Kohaku-TT's fastest of its
three routes (`results/<preset>-ours.json`, `-ktt-<route>.json`):

| M x K x N | Ours, µs | W | Kohaku-TT, µs | W | Kohaku-TT / ours |
|---|---|---|---|---|---|
| 128 x 896 x 128 | 6.3 | 67 | 28.7 | 63 | 4.58 |
| 16,384 x 384 x 384 | 61.2 | 130 | 94.1 | 116 | 1.54 |
| 16,384 x 1024 x 1024 | 393.5 | 128 | 389.9 | 136 | 0.99 |
| 4096 cubed | 958.3 | 138 | 957.0 | 151 | 1.00 |
| 8192 cubed | 7985.9 | 147 | 11797.6 | 139 | 1.48 |
| 16,384 cubed | 84018.0 | 133 | 75522.5 | 142 | 0.90 |
| 32 x 4096 x 4096 | 197.8 | 75 | 81.0 | 91 | 0.41 |

From 4096 cubed up both sides are bound by the card's power limit in a loop; the last
row is bound by reading W from DRAM on 72 cores.

[Attention](projects/attention/README.md#time-and-power), K and V cast down broadcasts of
cores, the same compute body as Kohaku-TT's flash kernel
(`results/ours-<preset>.json`, `ktt-<preset>.json`):

| B x H x S x D | Ours, µs | W | Kohaku-TT, µs | W | Kohaku-TT / ours |
|---|---|---|---|---|---|
| 1 x 32 x 4096 x 64 | 2675.3 | 138 | 2918.5 | 135 | 1.09 |
| 1 x 24 x 4096 x 128 | 2328.3 | 144 | 3001.8 | 132 | 1.29 |
| 8 x 16 x 256 x 64 | 91.7 | 122 | 80.7 | 110 | 0.88 |
| 1 x 8 x 1024 x 64, causal | 73.6 | 102 | 53.7 | 99 | 0.73 |
| 2 x 8 x 1024 x 128, 2 K / V heads | 163.3 | 143 | 261.8 | 105 | 1.60 |

The rows behind are behind per call and ahead per pass: the cost of a command.

[SwiGLU](projects/dense/SWIGLU.md#results), `silu(x W1) * (x W3)` then `W2`, HiFi2,
ours in its faster form (`results/swiglu-<preset>-<form>-hifi2.json`, `-ktt-<route>.json`):

| M x D x H | Ours, µs | W | Kohaku-TT, µs | W | Kohaku-TT / ours |
|---|---|---|---|---|---|
| 256 x 256 x 512 | 42.6 | 76 | 181.6 | 73 | 4.26 |
| 4096 x 256 x 512 | 73.7 | 114 | 183.5 | 95 | 2.49 |
| 16,384 x 256 x 512 | 180.2 | 137 | 542.1 | 109 | 3.01 |
| 16,384 x 384 x 384 | 161.7 | 142 | 458.1 | 108 | 2.83 |

The examples' own timings (commands per microsecond, broadcasts, hops, streams, graphs and
threads) are in each [example's README](examples/README.md).

## Install

Nothing is released: no package on PyPI and no GitHub release. Komorebi is used from a
clone, or from a wheel built from one with its dispatch image prebuilt
([packaging](packaging/README.md), [from a wheel](docs/install.md#from-a-wheel)), on
Linux x86-64 with Python 3.13:

```
git clone <this repository> komorebi && cd komorebi
uv venv -p 3.13 && source .venv/bin/activate && uv pip install -e ".[dev]"
kmr doctor                      # the Python, the tt-metal tree and its X280 runtime
```

The CPU tests and the stream and graph tools need nothing else. To run on a card: a
Blackhole P150A whose X280 memory is prepared for the X280 runtime, a tt-metal source
tree at the version in [runtime/tt-metal.pin](runtime/tt-metal.pin) (v0.81.1) built
with its X280 runtime, and `TT_METAL_HOME` naming that tree (neither `ttnn` nor
`torch` is needed). The dispatcher image and the native module are
built from the tree on first use (`python -m komorebi.native` builds them ahead).
Details: [install](docs/install.md).

The distribution is named `komorebi-tt` (`pip install komorebi-tt` once a release is
published); the package is imported as `komorebi` and the command is `kmr`.

## Usage

Every example runs on its own, checks its output against numpy and prints one JSON
line with `ok` ([examples](examples/README.md)); each program's page lists its
commands ([projects](projects/README.md)):

```
cd examples/02_merge_tree
python merge.py --jobs 20 --count 8 --form both
python merge.py --group-list 0:8,0:119 --form both --stall-rank 3
cd ../..
python -m projects.raytrace.frame --scene tiny --reps 16
```

The second command adds a load in which member 3 of job 1 never sends: the host raises
`km.CommandTimeout` naming the handle and the members that did not report, and the next
load in the same session runs exact.

## Repository map

```text
src/komorebi/                  the Python package: Program, Session, Run, layout, power, kmr
  engines/                     micro-op streams and graphs: builders, encodings, checks,
                               simulators; the host-fed executor
  thread.py                    X280 threads: declaration, regions, results, host fallback
  manifest/  lang/             reserved: homes and handlers declared; the thread language
src/komorebi_native/
  include/komorebi/abi/        the command, the event and the stream format, shared by all
  include/komorebi/device/     the persistent kernel, groups, ports, groups, broadcasts, dynamic
                               handlers, the trace
  kernels/                     the generic reader and writer of dynamic handlers
  x280/kernel/                 the dispatcher for X280 cluster 0
  x280/engines/  x280/kmx/     the plain-C graph engine; the X280 thread library
  csrc/host/                   the native module: X280 runtime, commands, events
examples/                      01_mapcollect to 08_parallel_thread
projects/                      raster, raytrace, dense (with SwiGLU), attention
experiments/                   hardware and design probes (see its README)
ref/                           CPU references in numpy
tests/core/  tests/projects/   tests with no card: the framework; the programs
tests/hardware/                card tests (KMR_CARD=1)
docs/                          concepts, references, install, development
```

## Limitations

- One Blackhole P150A, in slow dispatch. Grids of up to 128 cores in at most two
  multicast panels; one harvesting pattern has been run.
- One thread per X280 work cluster, three at most. A cluster reads only its own DRAM
  bank (6 for cluster 1, 7 for clusters 2 and 3); data for the other bank is pushed
  (`kmx_push`), so twelve cores gain over four only where the work computes more than
  it moves.
- Dynamic handlers: one dynamic run at a time; items travel through queues in DRAM,
  also to a core's own queue; an item cannot ask for a DRAM fetch; no order between
  items ([dynamic handlers](docs/reference/dynamic.md#not-in-this-version)).
- At most 16 commands in flight, ten 32-bit argument words per command, handlers 1 to
  239.
- Reading back from the card runs at about 1 to 2 MB per second
  (`Session.read_chunks` reads in 2 MB chunks with a line per chunk); a rendered frame
  of 20 MB takes 10 to 18 s to read back.
- The X280 runs at 1,000 MHz by default: at the 1,750 MHz opt-in the dispatcher
  trapped in 9 of 18 runs under full matrix load (the card lowers its core voltage),
  and in 0 of 23 at 1,000 MHz. One PLL clocks all four clusters.
- A stuck RISC thread is recovered by the next device open. A kernel that leaves a
  compute unit running, a packer that writes past its buffer, an X280 thread that never
  returns and polls nothing, or a dispatcher whose loop has stopped costs a card
  reset. The compute kernel's waits between its own three threads and the NoC barriers
  are not bounded.
- tt-metal's watcher cannot run while the X280 runtime is open; debugging is by the
  cores' state words, "where" codes and the per-core trace, read after the stop.
- `kmr build host` and `kmr build pymodule` take their compile flags from two targets
  that the [tt-x280-dispatch](https://github.com/KohakuBlueleaf/tt-x280-dispatch)
  project adds to a tt-metal build
  ([development](docs/development.md#known-limitations)).

## Roadmap

Planned, in order:

1. The rasterizer and the ray tracer faster on the card, where their pages say the
   time goes.
2. A first release: the wheel published, with its prebuilt dispatch image.
3. The manifest and the thread language.

## Documentation

| Page | Read it for |
|---|---|
| [docs index](docs/README.md) | every concept and reference page |
| [programming model](docs/concepts/model.md) | threads, breakpoints, homes, space and time |
| [Tensix runtime](docs/reference/runtime.md) | the device headers, the command, bounded waits, the control block |
| [ports](docs/reference/ports.md), [broadcasts](docs/reference/broadcast.md), [dynamic handlers](docs/reference/dynamic.md) | crossings between cores |
| [streams](docs/streams.md), [graphs](docs/graphs.md) | deciding on the card in the dispatcher |
| [X280 threads](docs/threads.md) | C on a work cluster with kmx |
| [host API](docs/reference/host.md) | `Program`, `Session`, `Run`, errors, builds |
| [examples](examples/README.md), [projects](projects/README.md) | runnable examples; the five programs and their results |
| [install](docs/install.md), [development](docs/development.md) | building from a clone; layout, checks, conventions |

## Contributing

Open an issue first for a feature. Before a pull request, from the repository in the
environment of [install](docs/install.md):

```
python -m pytest tests/core
python -m pytest tests/projects
python -m komorebi.tools.cxx_format --check
python -m komorebi.tools.comment_lint --check src tests examples experiments packaging
ruff check . && black --check . && blacken-docs -t py313 -l 88 $(git ls-files '*.md')
```

Conventions: [development](docs/development.md).

## Licence

Apache-2.0; see [LICENSE](LICENSE).
