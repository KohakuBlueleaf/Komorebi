# Komorebi

A programming model and a runtime for the Tensix grid of Tenstorrent Blackhole cards
(developed on a p150a). A program is written as straight-line threads; a thread has a
program counter and a place, runs on its core until it needs state held on another
core, and then moves there or sends a value there. On the card, a program is one set of
resident kernels on every Tensix core, fed 64-byte records by a dispatch kernel on one
of the card's X280 clusters. It is for kernel authors whose work does not fit one fixed
program per shape: jobs whose team of cores changes from one job to the next, and work
such as rendering, where items need different code and meet state held on other cores.

**Status: early development; nothing is released.** On a card today: a program's three
resident kernels on every Tensix core, run by the dispatch kernel on X280 cluster 0,
records issued from Python to teams that change from record to record, completion per
team, and a deadline that stops the run and names the cores that did not report
([examples/01_mapcollect](examples/01_mapcollect/)); values sent from core to core and
reduced by owner or by tree ([02_merge_tree](examples/02_merge_tree/)); blocks broadcast
along the rows and columns of a team ([03_broadcast](examples/03_broadcast/)); a graph
run host-fed ([06_graph](examples/06_graph/)); streams and graphs uploaded from Python
and run by the dispatch kernel, with no host step between their start and their end
([engines](docs/reference/host.md#engines-uploads-start-events)), such as a stream on
the X280 that draws each job's team size ([04_uop_stream](examples/04_uop_stream/)); a
thread on a second
X280 cluster that issues records and decides from their result words
([07_work_cluster](examples/07_work_cluster/)); and dynamic segments, in which items
move from core to core as their data decides and the run ends by itself
([05_hop](examples/05_hop/)). Threads that move (`at`) and the
thread language are [planned](#roadmap).

```python
import komorebi as km

program = km.Program(
    "mapcollect",
    kernels="kernels/",  # reader.cpp, compute.cpp, writer.cpp
    buffers=[
        km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
        km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
    ],
    compile_args={"reader": reader_args, "writer": writer_args},  # tensor accessors
    segments="kernels/mapcollect.h",  # kSegMap = 1, kSegStall = 2, ...
)

with km.Session(card=0) as session:
    run = session.load(program)  # the grid's kernels first, then the dispatch kernel
    tickets = [
        run.issue("map", team=(8 * i, 8), args=[x_addr, 64, out_addr, 8 * i])
        for i in range(15)  # 15 disjoint teams of 8 cores
    ]
    run.wait(tickets, timeout=5.0)
    run.stop()  # EXIT to every core, stop the kernel, then read back
```

The `MAP` segment of the compute kernel
([compute.cpp](examples/01_mapcollect/kernels/compute.cpp)); the runtime owns the loop
and calls the body once for each record whose team holds this core:

```cpp
km::Compute cp;
compute_kernel_hw_startup(kCbIn, kCbOut);
cp.run([&](const km_record_t& r) {
    if (km::seg(r) != kSegMap) {
        return;
    }
    const km::Share s = km::share(r.a1, cp.me() - km::first(r), km::count(r));
    tile_regs_acquire();
    // ... add this member's s.len input tiles into one tile in DST (fp32), pack it
});
```

## The programming model

### The card as Komorebi runs it

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/card-dark.svg">
  <img src="docs/images/card-light.svg" width="830" alt="The P150A tile grid in NoC0 coordinates, 17 columns by 12 rows: DRAM tiles in columns 0 and 9, Ethernet in row 1, PCIe and ARC in row 0, the security tile and four X280 tiles in column 8, Tensix cores in columns 1 to 6 and 11 to 16 of rows 2 to 11 (columns 7 and 10 fused off on this card), each numbered with its Komorebi core index 0 to 119. Host commands and events pass through PCIe to X280 0 at (8, 3), which runs the dispatch kernel. Its record goes out to two panels, columns 1 to 6 and 11 to 16, one multicast window each. A team of cores 26 to 55 is filled; its leader, core 26 at (3, 4), acknowledges on NoC1. Boxes below list what the dispatch kernel and each Tensix core do, and what is planned.">
</picture>

The figure is the card's real tile grid in NoC0 coordinates; the numbers in the Tensix
cores are Komorebi's core index, row by row over the 12 x 10 grid of usable cores, so
index and position differ where the grid skips the fused-off, X280 and DRAM columns.
The session finds the two multicast panels from the card at start-up.

A **record** is one 64-byte line: a sequence number, the segment, the team and ten
32-bit argument words; the sequence number is repeated in each of the four 16-byte
chunks, so a core accepts the line only when all of it has landed. A **team** is a
contiguous range of cores `[first, first + count)` in the grid's core order (row by
row, x fastest), named in the record; its first member is the team's **leader**. Every
core sees every record and skips those whose team does not hold it. Members report to
the leader, and the leader writes one word to the dispatch kernel, so several teams can
be busy at once and a team's size is free from record to record.

### A thread has a program counter and a place

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/breakpoints-dark.svg">
  <img src="docs/images/breakpoints-light.svg" width="830" alt="Left: a thread program of two threads. Thread path(pixel) computes a camera ray, loops over the treelets along the ray with 'at treelet[k]:' around the walk, shades the hit, and ends with 'emit film[pixel] += c'. Thread resolve(t) waits for every pixel of film tile t, then writes the tile. The lines with the breakpoints at, emit and wait are marked. Right: a grid of cores on which the path thread starts on one core (lines 1 to 3), moves its live variables to the core holding treelet 2 (line 4), walks it there (line 5), moves to the core holding treelet 5, walks and shades there without crossing the network (lines 5 and 6), and emits the value (pixel, c) to the core holding film tile t (line 7), where the resolve thread waits (line 10).">
</picture>

A **breakpoint** is a point where a thread must deal with another core. Between two
breakpoints the thread runs on its core without interruption, and after a move it stays
where it landed. The code between two breakpoints is a **segment**; on the card it is
one case of the switch a record selects. State has a **home**: the core that holds it,
such as a film tile or a treelet of a box tree. The author names state, never a core;
`at X[i]:` continues the thread where `X[i]` lives, and is no breakpoint when `X[i]` is
already here. The grid in this figure is a schematic. None of these three breakpoints
exists between cores yet; what runs today is records, teams and completion
(figures 1, 3 and 4).

| The author writes | What crosses the network | What the thread does |
|---|---|---|
| `at X[i]: block` | the thread's live variables | continues on the core that holds `X[i]` |
| `emit X[i] += v`, `emit v -> thread(j)` | the value | keeps running |
| `v = wait(n)` | nothing | pauses until `n` values have landed |

When the sender, the receiver and the count are known before the run, a breakpoint
becomes a write into a fixed slot and a counted wait. A matrix product with K split
across cores, merged by a tree; every thread runs the same program, and only its index
differs:

```text
thread mm(j, n):                                 # one per core, pinned
    part = A[:, K(j)] @ B[K(j), :]               # one segment; no breakpoint
    for r in 0 .. levels(n) - 1:
        if j % 2^(r+1) == 0:                     # I merge in this round
            if j + 2^r < n:
                part += wait(1)                  # breakpoint: my partner has handed over
        else:
            emit part -> thread(j - 2^r)         # breakpoint: hand my partial to my merger
            return
    write C = part                               # only thread 0 arrives here
```

A **lane** is a run of cores in one grid column; a group of threads such as
`threads(lane, *)` is a slice of the index space, and a broadcast to it is one
multicast. Space is the thread's index and time is a loop in its body; the
[model](docs/concepts/model.md) works through ray tracing, attention and matrix product
in this form.

### Teams and records in time

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/teams-dark.svg">
  <img src="docs/images/teams-light.svg" width="830" alt="Time runs left to right over cores 0 to 11, the first row of the grid (NoC0 row 2, columns 1 to 6 in panel 0 and 11 to 16 in panel 1). The dispatch kernel issues R1 to cores 0 to 4 and R2 to cores 6 to 11; each record lands on every core, the members work at the same time, report to their leader, and each leader acknowledges its own record, so R1 and R2 are in flight together. R3, for cores 3 to 8, overlaps both teams. In the drawn case it lands after cores 3, 4 and 6 to 8 have begun R1 or R2, so they take it next, while core 5, in neither team, takes it at once; each member's report ends on the lane of the leader, core 3, which acknowledges after the last report. A note says this order is the drawn case, not a guarantee: a shared core may run two records in flight in either order unless the later one is issued exclusive.">
</picture>

A record is in flight from the moment the dispatch kernel stores it until its leader's
acknowledgement arrives; it holds one of up to 16 completion slots (`depth`, 4 by
default). Two records in flight together on overlapping teams may be run by a shared
core in either order: a record of which no chunk has landed yet is invisible to the
core. The figure draws one case, in which R3 lands after the shared cores have begun R1
or R2, so they take it next. A record that must run after an earlier one is issued with
`run.issue(..., exclusive=True)`, which holds it on the host until no team in flight
overlaps it.

### One program on the grid

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mapcollect-dark.svg">
  <img src="docs/images/mapcollect-light.svg" width="830" alt="The map/collect example on the real tile grid: the team of cores 0 to 7 sits at NoC0 (1, 2) to (6, 2), (11, 2) and (12, 2); core 3 at (4, 2) is highlighted. A table lists each member's share: core m reads tiles 8m to 8m+7 and writes output page a3 + m. Members report to core 0, which acknowledges to X280 0. Inside core 3: the reader reads tiles 24 to 31 from DRAM on NoC0 into circular buffer 0 in L1, compute copies each tile to DST and the vector unit adds them in fp32, pack writes the sum to circular buffer 16, the writer writes it to DRAM on NoC1, and the core reports to core 0.">
</picture>

In the implemented example nothing but the record and the completion travels between
cores: each member reads its own share of the input from DRAM, works in its own L1 and
vector unit, and writes its own output tile.

### What exists and what is planned

| Construct | Today |
|---|---|
| Records, the ring of 16 mailbox slots on each core, the resident reader, compute and writer | implemented ([km_worker.hpp](src/komorebi_native/include/komorebi/device/km_worker.hpp)) |
| Teams per record; completion per team, one level or through row leaders | implemented ([km_team.hpp](src/komorebi_native/include/komorebi/device/km_team.hpp), [km_record.hpp](src/komorebi_native/include/komorebi/device/km_record.hpp)) |
| Host-fed dispatch: `Session`, `Run`, the dispatch kernel's host ring, slots, deadlines and events | implemented ([host API](docs/reference/host.md), [x280/kernel](src/komorebi_native/x280/kernel/)) |
| `emit`, `arrive` and `take` between cores, groups, `reduce` by owner and by tree | implemented ([ports, emit and reduce](docs/reference/ports.md), [02_merge_tree](examples/02_merge_tree/)) |
| Broadcast in a lane along a row or a column | implemented ([lane broadcast](docs/reference/lanes.md), [03_broadcast](examples/03_broadcast/)) |
| Micro-op streams, the job queue and the graph executor | in the dispatch kernel: streams and graphs on core 1, the job queue on core 0; builders, checks and simulators on the CPU; on a card, `Run.upload`, `start` and `wait_engine` run a stream or a graph in the kernel, and `run_hostfed` runs the same one host-fed ([streams](docs/streams.md), [graphs](docs/graphs.md), [engines](docs/reference/host.md#engines-uploads-start-events)) |
| User threads on X280 clusters 1 to 3 | implemented: `kmx` calls, a command ring to core 0, a host fallback ([X280 threads](docs/threads.md), [07_work_cluster](examples/07_work_cluster/)) |
| `at`: threads that move, inboxes and packets of records | dynamic segments through a `Session` and on the test feeder: per-core queues in DRAM, quota, the quiet end found by the dispatch kernel ([dynamic segments](docs/reference/dynamic.md), [05_hop](examples/05_hop/)); reservations on counter tiles, 10^6 hops exact in every run measured (91 of 91 through a `Session`, 3 of 3 under the test feeder) |
| Homes and segments declared, and the thread language compiled into them | planned ([manifest](src/komorebi/manifest/README.md), [lang](src/komorebi/lang/README.md)) |

## Highlights

- **Resident workers**: `km::Reader`, `km::Compute` and `km::Writer` own each kernel's
  loop; a program supplies the body, a switch on the segment
  ([km_worker.hpp](src/komorebi_native/include/komorebi/device/km_worker.hpp)).
- **A dispatch kernel on the X280**: one bare-metal image on core 0 of cluster 0 that
  takes host commands, multicasts records with one vector store per panel, tracks up
  to 16 in flight and checks each against its deadline
  ([x280/kernel](src/komorebi_native/x280/kernel/)).
- **Failures that say where**: the reader's and the writer's waits are bounded and
  observe a stop word, and the dispatch kernel checks a deadline per record; a missed
  deadline raises `km.RecordTimeout` with the command, slot, sequence,
  team and each member that did not report, with where its reader and writer gave up
  ([host API](docs/reference/host.md#errors)).
- **A per-core trace**: wall-clock stamps per record (taken, outputs written,
  completion sent), read back after the stop (`run.trace()`).
- **Micro-op streams**: the upload format, a Python builder, the check and a simulator
  for the small programs the dispatch kernel will run to decide which records to
  issue, in passes in which every issuer (each stream, then the job queue) takes a
  turn; 89 reference streams for the card's interpreter to replay, and a test that
  applies about a hundred small patches to the simulator and requires each to change a
  saved answer ([streams](docs/streams.md)).
- **Whole-card power beside timings**: `kmr power` prints each card's board power once a
  second while a benchmark runs, and `komorebi.power.PowerSampler` wraps a section of a
  program with mean, peak and energy per marked phase. It reads the card's `blackhole`
  hwmon sensor, which updates every 0.1 s, opens no card and takes no lock
  ([install](docs/install.md#watching-power)).
- **Measured hardware facts** behind the design: how a multicast record lands, whether
  record multicasts and in-column broadcasts collide, and what one Tensix tile costs
  for a 32-ray box-tree walk ([experiments](experiments/README.md)).
- **Render references**: rasterization and ray tracing in numpy, threaded C++ and CUDA,
  bit for bit equal on four scenes, for the card versions to be compared with
  ([projects](projects/README.md)).

## Measured

The examples on one p150a, AICLK 1350 MHz, X280 at 1000 MHz (the session's default;
the 1750 MHz opt-in's figure in brackets), every output checked against numpy; each
run, with its command, is in the example's README. The test feeder's rows use no X280:

| Example | What runs | Result |
|---|---|---|
| [01_mapcollect](examples/01_mapcollect/) | 1,000 jobs of 64 input tiles on random teams (61,720 member records), up to 4 in flight, issued from Python through the dispatch kernel | 1.751 µs per job by the kernel's clock [1.655], 2.078 by the host's [1.985]; a record from issue to completion 4.2 µs (mean) [4.5]; 0 of 1,000 jobs wrong |
| [02_merge_tree](examples/02_merge_tree/) | parts merged by owner or by tree, under the test feeder | 7.3 µs per job on 8 cores, 24.0 on 119; 1,000 random jobs exact; both forms bit-equal |
| [03_broadcast](examples/03_broadcast/) | 24 KiB blocks to a 12 x 9 team along 27 lanes, under the test feeder | 1,355 GB/s delivered from 169 GB/s read from DRAM; every member reading DRAM itself: 457 GB/s |
| [04_uop_stream](examples/04_uop_stream/) | 1,000 jobs of 64 input tiles on teams whose size a micro-op stream draws at random, decided on the host (`run_hostfed`), by a stream that issues each record on the X280, or by one that submits to the dispatch kernel's job queue | 163 µs per job host-fed [161], 13.4 issued on the X280 [11.5], 9.3 submitted [8.4] (up to 3 in flight); 59,456 output tiles bit-exact each way |
| [07_work_cluster](examples/07_work_cluster/) | a loop of 204 rounds of two records, each round's team chosen from the last result word | an X280 thread on cluster 1: 10.8 ms (about 53 µs per round) [10.3]; host-fed: 48 ms [50] |

Landing a 64-byte record multicast by one X280 core to every Tensix core: p150a, 120
Tensix cores (12 x 10) at AICLK 1350 MHz, X280 at 1750 MHz (these experiments' own
setting; a session defaults to 1000), slow dispatch, two multicast panels; the experiment's own X280 image and a Tensix kernel that checks every
payload word and does no work; microseconds per record by the X280's tick counter over
10^6 records, issue to the last acknowledgement. Komorebi's records use echo words
([landing](experiments/landing/README.md)).

| Protocol | Records in flight | µs per record | Torn records / core-records checked | Check on the core, cycles (mean) |
|---|---|---|---|---|
| sequence echoed in every 16-byte chunk | 1 | 1.285 | 0 / 733,212,120 | 39.8 |
| sequence echoed in every 16-byte chunk | 4 | 0.772 | 0 / 733,212,120 | 39.2 |
| record, fence, then a sequence word | 1 | 1.310 | 0 / 733,212,120 | 29.6 |
| record, fence, then a sequence word | 4 | 0.728 [^1] | 0 / 733,212,120 | 29.2 |

The same card and clocks, records at 4 in flight while lanes of 2, 5 and 10 cores
broadcast 64 B to 24 KiB blocks down every grid column: no lost or wrong word and no
hang in about 7.0 x 10^9 checked core-records; records took 0.7752 µs alone and 0.7719
to 0.7780 µs with lanes on either NoC ([collision](experiments/collision/README.md)).

One Tensix core, tt-metal v0.81.1, AICLK 1350 MHz, median of three runs: the vector
unit runs a multiply-add in 1.000 cycle with 32, 1 or no lanes enabled; one walk of a
255-node box tree, in cycles per ray, every best distance bit-equal to the CPU
reference ([tile probes](experiments/tile_probes/README.md)):

| Rays, ray set | Ray-first (each lane at its own node) | Node-first (rays binned by node) |
|---|---|---|
| 256, camera | 2273 | 1232 |
| 1024, camera | 2269 | 1008 |
| 256, random | 3500 | 2199 |
| 1024, random | 3412 | 2110 |

The render references, 1024 x 1024: numpy, C++ and CUDA give the same arrays bit for
bit on all four scenes. Milliseconds per frame (wall time of a frame loop of at least
3 s) on a shared 96-thread AMD EPYC 7K62 (load average 5 to 18) and one Tesla V100
32 GB; no card version is landed yet
([raster](projects/raster/reference/README.md),
[ray tracing](projects/raytrace/reference/README.md)).

| Scene | Triangles | Raster, C++ 1 thread | C++ 96 threads | V100 | Ray tracing, C++ 1 thread | C++ 96 threads | V100 |
|---|---|---|---|---|---|---|---|
| tiny | 398 | 11.21 | 2.66 | 0.29 | 726.4 | 23.1 | 2.3 |
| 10^4 | 11,994 | 16.79 | 3.32 | 0.25 | 1078.4 | 34.9 | 3.7 |
| 10^5 | 102,308 | 39.90 | 4.08 | 0.79 | 1452.1 | 37.7 | 6.1 |
| 10^6 | 997,806 | 152.78 | 17.87 [^2] | 4.42 | 2088.9 | 54.3 | 11.3 |

<img src="projects/raster/reference/images/s10k_colour_256.png" width="256" alt="Rasterized 10^4 scene: spheres of mixed size over a ground grid"> <img src="projects/raytrace/reference/images/s10k_image_256.png" width="256" alt="Ray-traced 10^4 scene with shadows and one diffuse bounce">

*The 10^4 scene, rasterized and ray traced (256 x 256 downscales of 1024 x 1024).*

[^1]: To be revisited: at 4 in flight the record-then-sequence-word protocol is
    0.044 µs faster end to end than echo words, against 0.025 µs slower at 1; the cause
    was not separated.
[^2]: To be revisited: the timed stages add up to 12.99 ms (median; 11.99 to 18.25)
    against 17.87 ms for the frame loop; the 96-thread rows have the widest spread on
    the shared host, not investigated.

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

The CPU tests and the stream tools need nothing else. To run on a card: a Blackhole
p150a whose X280 memory is prepared for the X280 runtime, a tt-metal source tree at the
version in [packaging/tt-metal.pin](packaging/tt-metal.pin) (v0.81.1) built with its
X280 runtime, `ttnn` and `torch` from it in the same environment, and `TT_METAL_HOME`
naming that tree. The dispatch kernel image and the native module are built from the
tree on first use (`python -m komorebi.native` builds them ahead). Details:
[install](docs/install.md).

## Usage

### Running the example on a card

```
cd examples/01_mapcollect
python mapcollect.py --jobs 20 --count 8
python mapcollect.py --jobs 1000 --teams random --out-tensors 32
python mapcollect.py --jobs 10 --count 8 --first 4 --stall 3 --stall-member 2 --deadline 0.02
```

Each run checks every member's output tile against numpy (bit-exact on integer inputs)
and prints one JSON line: microseconds per job by the dispatch kernel's clock and the
host's, the issue-to-completion latency and every core's end state. The third command
makes member 2 of job 3 never report, which ends in `km.RecordTimeout` naming that core
([01_mapcollect](examples/01_mapcollect/README.md)).

### Writing a segment

A program is three kernels with a switch on the segment inside the runtime's loops.
Segment numbers (1 to 239) are `constexpr uint32_t kSegName = n;` lines of a header the
host reads; the record's ten argument words are `r.a0` to `r.a9`. The writer's body
returns whether this member reports the record
([writer.cpp](examples/01_mapcollect/kernels/writer.cpp), without its test segment):

```cpp
km::Writer wr;
wr.run([&](const km_record_t& r) {
    const uint32_t index = wr.me() - km::first(r);
    switch (km::seg(r)) {
        case kSegMap: {
            if (!wr.wait_front(kCbOut, 1)) {
                return true;
            }
            const auto out = TensorAccessor(out_args, r.a2);
            noc_async_write_page(r.a3 + index, out, get_read_ptr(kCbOut));
            noc_async_writes_flushed();
            cb_pop_front(kCbOut, 1);
            return true;
        }
        case kSegStall:
            return index != r.a0;
        default:
            return true;
    }
});
```

After the body the runtime flushes the writes and reports: one increment on the
leader's counter, or, on the leader, one acknowledgement word to the dispatch kernel
once all members have reported.

### A micro-op stream, simulated without a card

A stream is a small program for the dispatch kernel that decides which records to
issue. Eight jobs of 30 cores through the job queue, which picks each job's first core
([docs/streams.md](docs/streams.md)):

```python
from komorebi.engines.stream import Stream

SEG_MAP = 1  # the program's segment number

s = Stream(deadline_us=200_000, seed=1)
(j,) = s.regs(1)
top = s.label()
s.submit(SEG_MAP, count=30, args=[j])  # a job of 30 cores; the queue picks `first`
s.addi(j, j, 1)
s.jlt(j, s.const(8), top)
s.wait(0)  # until every record of the stream has completed
s.halt(s.const(0))
upload = s.encode()  # the bytes run.upload(s) writes on a card

done = s.simulate(n_cores=120, delay=2)  # a record completes 2 passes after issue
print(len(upload), done.status, done.halt)
for r in done.records:
    print(r.seq, r.args[0], (r.first, r.count), r.pass_index)
```

```text
320 ended 0
1 0 (0, 30) 0
2 1 (30, 30) 0
3 2 (60, 30) 0
4 3 (90, 30) 0
5 4 (0, 30) 3
6 5 (30, 30) 3
7 6 (60, 30) 3
8 7 (90, 30) 3
```

Four disjoint teams fill the 120 cores in the first pass; the next four are issued when
the first ones complete. The simulator is the definition the card's interpreter is
tested against: `run.upload(s)` and `run.start` run the same stream in the dispatch
kernel, and the card tests compare its registers with `s.simulate`.

## Repository map

```text
src/komorebi/                  the Python package: Program, Session, Run, layout, power, kmr
  engines/                     micro-op streams and graphs: builders, encodings, checks,
                               simulators; the host-fed graph executor
  manifest/  lang/             reserved: records, homes and segments; the thread language
src/komorebi_native/
  include/komorebi/abi/        the record, the event and the stream format, shared by all
  include/komorebi/device/     the resident worker, teams, ports, groups, lanes, the trace
  x280/kernel/                 the dispatch kernel for X280 cluster 0
  x280/engines/  x280/kmx/     the plain-C graph engine; the X280 thread library
  csrc/host/                   the native module: X280 runtime, commands, events
examples/                      01_mapcollect, 02_merge_tree, 03_broadcast, 04_uop_stream,
                               05_hop, 06_graph, 07_work_cluster
experiments/                   hardware and design probes (see its README)
projects/  ref/                render references; CPU references in numpy
tests/core/  tests/hardware/   tests with no card; card tests (KMR_CARD=1)
tests/ref/                     the reference renderers against each other
docs/                          concepts, host API, streams, install, development
```

## Limitations

- One Blackhole p150a, in slow dispatch. Grids of up to 128 cores in at most two
  multicast panels; one harvesting pattern has been run.
- Host-fed, each record is one Python call to `run.issue`; deciding on the card goes
  through an X280 thread today (streams and graphs on the card wait for their host
  calls).
- At most 16 records in flight, ten 32-bit argument words per record, segments 1 to
  239.
- Between cores the runtime has ports (`emit`, `arrive`, `take`), groups, `reduce` and
  lanes; threads that move to another core are not built yet.
- tt-metal's watcher cannot run while the X280 runtime is open. Debugging is by the
  cores' state words, "where" codes and the per-core trace, all read after the stop.
- A compute body's waits on its buffers are not bounded yet, so a stop while a long
  record is in flight can leave a core's compute kernel waiting.
- Reading tensors back is slow: a 4 MB tensor takes about 2 s bank by bank
  (`session.read_bf16`) and about 3 s through `ttnn.to_torch` (host load about 8;
  [dram_read](experiments/dram_read/README.md)).
- `kmr build host` and `kmr build pymodule` take their compile flags from two targets
  that the [tt-x280-dispatch](https://github.com/KohakuBlueleaf/tt-x280-dispatch)
  project adds to a tt-metal build
  ([development](docs/development.md#known-limitations)).

## Roadmap

1. Threads that move (`at`) on top of dynamic segments.
2. A first release: the wheel published, with its prebuilt dispatch image.
3. Programs on the card against the references: matrix product, attention, SwiGLU,
   rasterization, ray tracing.
4. The manifest and the thread language.

## Documentation

| Page | Read it for |
|---|---|
| [docs index](docs/README.md) | every concept and reference page |
| [programming model](docs/concepts/model.md) | threads, breakpoints, homes, space and time |
| [dispatch kernel](docs/concepts/dispatch.md) | the kernel on X280 cluster 0 and the work clusters |
| [Tensix runtime](docs/concepts/runtime.md) | the resident worker, records, teams, breakpoints |
| [host API](docs/reference/host.md) | `Program`, `Session`, `Run`, errors, builds |
| [micro-op streams](docs/streams.md) | the upload format, builder, check and simulator |
| [install](docs/install.md), [development](docs/development.md) | building from a clone; layout, checks, conventions |

## Contributing

Open an issue first for a feature. Before a pull request, from the repository in the
environment of [install](docs/install.md):

```
python -m pytest tests/core
python -m komorebi.tools.cxx_format --check
python -m komorebi.tools.comment_lint --check
ruff check . && black --check . && blacken-docs -t py313 -l 88 $(git ls-files '*.md')
```

Conventions: [development](docs/development.md).

## Licence

Apache-2.0; see [LICENSE](LICENSE).
