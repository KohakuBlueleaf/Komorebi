# 01 map/collect

One program runs as a resident worker on every Tensix core of the card. The dispatch
kernel on X280 cluster 0 takes records from the host and sends each one to every core.
A record names a segment and a team, cores `[first, first + count)` of the core order
(row by row, x fastest); cores outside the team ignore it. When every member of the team
has finished, the team's first member tells the dispatch kernel, which tells the host.

| Segment | Arguments | What a member does |
|---|---|---|
| `MAP` | a0 input, a1 tiles T, a2 output, a3 output page of member 0, a6 first input tile | reads its share of input tiles a6 to a6 + T - 1, adds them into one tile (fp32), stores it at page a3 + member, reports |
| `STALL` | a0 member | member a0 waits for the stop word and does not report; the others report |
| `ALIGN` (a test) | a9 release time (low word of the wall clock; the test feeder sets it) | every member reports at time a9 |
| `LATE` (a test) | a0 member, a1 cycles | member a0 reports a1 cycles after it took the record; the others report at once |
| `NOP` | none | reports |
| `BAD_ITEM` (a test) | a0 member, a1 side (0 writer, 1 reader) | member a0 asks for an item that passes the end of its buffer and gives up with `bad_item`; the others report |

The host checks every member's tile against numpy: bit-exact on integer inputs (even
jobs), and the relative RMS error of the total on uniform random inputs (odd jobs).

## The program, from the host

```python
import komorebi as km

program = km.Program(
    "mapcollect",
    kernels="kernels/",  # reader.cpp, compute.cpp, writer.cpp
    buffers=[
        km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
        km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
    ],
    compile_args={"reader": reader_args, "writer": writer_args},
    compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
    segments="kernels/mapcollect.h",  # kSegMap = 1, kSegStall = 2, ...
)

with km.Session(card=0) as session:
    run = session.load(program)  # launch the grid, then the dispatch kernel
    t = run.issue("map", team=(0, 8), args=[x_addr, 64, out_addr, 0])
    run.wait(t, timeout=5.0)  # sends what is queued, waits for the event
    run.stop()  # EXIT to every core, stop the kernel, close the X280 runtime
    print(run.state()[0], run.trace())  # read back only now
```

`reader_args` and `writer_args` are the tensor accessor arguments of the input and
output tensors (`ttnn.TensorAccessorArgs(tensor).get_compile_time_args()`); `x_addr`
and `out_addr` are their `buffer_address()`. `mapcollect.py` is this program with the
tensors, the teams and the checks around it.

If a member never reports, the dispatch kernel's deadline (one second unless `load`
says otherwise) ends the run: `run.wait` raises `km.RecordTimeout`, which names the
command, the slot, the sequence number, the team, and the members that did not report
with where each was waiting. The session is already closed cleanly by then.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels: a switch on the segment inside the runtime's loops (`komorebi/device/km_worker.hpp`) |
| `kernels/mapcollect.h` | segments, buffers and record arguments shared by the kernels |
| `mapcollect.py` | builds the records, runs them, checks the outputs, prints one JSON line |

## What it needs

- this repository installed in the Python environment (`pip install -e .` from its
  root);
- a tt-metal v0.81.1 tree with the X280 runtime and the tt-x280-dispatch project's
  targets, `ttnn` and `torch` from it in the same environment, `TT_METAL_HOME` naming
  it, and a Blackhole p150a whose X280 LIM ECC was primed once: the steps are in
  [Running on a card](../../docs/install.md#running-on-a-card). The dispatch kernel
  image and the package's native module are built from the tree on first use (about
  ten seconds) into `build/komorebi/`. The session runs in slow dispatch (it sets
  `TT_METAL_SLOW_DISPATCH_MODE=1` if unset).

```
python mapcollect.py --jobs 20 --count 8
python mapcollect.py --jobs 1000 --teams random --out-tensors 32
python mapcollect.py --jobs 40 --teams pair --depth 4 --trace
python mapcollect.py --jobs 10 --count 8 --first 4 --stall 3 --stall-member 2 --deadline 0.02
python mapcollect.py --jobs 100000 --seg nop --count 120 --no-check
```

`--teams` is `fixed` (`--first`, `--count`), `random` (first and count uniform),
`pair` (two halves of the grid, alternating) or `rotate` (disjoint teams of `--count`
cores in turn). `--depth` is the number of mailbox slots the cores scan and the
dispatch kernel uses (default 4). Outputs go to a ring of `--out-tensors` tensors of
`--out-pages` tiles; only jobs whose pages no later job overwrote are checked, and
reading back runs at about 1 MB/s in slow dispatch. `--trace` adds the per-record
phases (microseconds after the first member took the record: last member took it,
last outputs written, leader pushed the completion) and counts consecutive jobs on
disjoint teams that were in flight at once. A team inside one grid row completes in
one level (every member reports to the first member); a team that spans rows completes
through row leaders (the runtime chooses per record).

The JSON line reports microseconds per job by the dispatch kernel's clock and by the
host's, the issue-to-completion latency, the Python time per record in `issue` and in
the batched stores, and every core's end state.
