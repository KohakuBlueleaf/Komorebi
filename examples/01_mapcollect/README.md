# 01 map/collect

One program runs as a persistent kernel on every Tensix core of the card. The dispatcher on X280 cluster 0 takes commands from the host and sends each one to every core.
A command names a handler and a group, cores `[first, first + count)` of the core order
(row by row, x fastest); cores outside the group ignore it. When every member of the group
has finished, the group's first member tells the dispatcher, which tells the host.

| Handler | Arguments | What a member does |
|---|---|---|
| `MAP` | a0 input, a1 tiles T, a2 output, a3 output page of member 0, a6 first input tile | reads its share of input tiles a6 to a6 + T - 1, adds them into one tile (fp32), stores it at page a3 + member, reports |
| `STALL` | a0 member | member a0 waits for the stop flag and does not report; the others report |
| `ALIGN` (a test) | a9 release time (low word of the wall clock; the test feeder sets it) | every member reports at time a9 |
| `LATE` (a test) | a0 member, a1 cycles | member a0 reports a1 cycles after it took the command; the others report at once |
| `NOP` | none | reports |
| `BAD_ITEM` (a test) | a0 member, a1 side (0 writer, 1 reader) | member a0 asks for an item that passes the end of its buffer and gives up with `bad_item`; the others report |
| `RESULT` (a test) | a0 value | each member gives a0 plus its index as the result word; with `KM_F_RESULT` the command is acknowledged with the leader's, a0 |

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
    handlers="kernels/mapcollect.h",  # kHandlerMap = 1, kHandlerStall = 2, ...
)

with km.Session(card=0) as session:
    run = session.load(program)  # launch the grid, then the dispatcher
    t = run.submit("map", group=(0, 8), args=[x_addr, 64, out_addr, 0])
    run.wait(t, timeout=5.0)  # sends what is queued, waits for the event
    run.stop()  # EXIT to every core, stop the kernel, close the X280 runtime
    print(run.state()[0], run.trace())  # read back only now
```

`reader_args` and `writer_args` are the tensor accessor arguments of the input and
output tensors (`tensor.accessor_args()` of tensors made by
`session.tensor(array, "bfloat16", "tile")`); `x_addr` and `out_addr` are their
`address`. `mapcollect.py` is this program with the tensors, the groups and the checks
around it.

If a member never reports, the dispatcher's deadline (one second unless `load`
says otherwise) ends the run: `run.wait` raises `km.CommandTimeout`, which names the
command, the slot, the sequence number, the group, and the members that did not report
with where each was waiting. The session is already closed cleanly by then.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels: a switch on the handler inside the runtime's loops (`komorebi/device/km_worker.hpp`) |
| `kernels/mapcollect.h` | handlers, buffers and command arguments shared by the kernels |
| `mapcollect.py` | builds the commands, runs them, checks the outputs, prints one JSON line |

## What it needs

- this repository installed in the Python environment (`pip install -e .` from its
  root);
- a tt-metal v0.81.1 tree with the X280 runtime and the tt-x280-dispatch project's
  targets, `TT_METAL_HOME` naming it, and a Blackhole p150a whose X280 LIM ECC was
  primed once: the steps are in [Running on a card](../../docs/install.md#running-on-a-card). The dispatcher
  image and the package's native module are built from the tree on first use (about
  ten seconds) into `build/komorebi/`. The session runs in slow dispatch (it sets
  `TT_METAL_SLOW_DISPATCH_MODE=1` if unset).

```
python mapcollect.py --jobs 20 --count 8
python mapcollect.py --jobs 1000 --groups random --out-tensors 32
python mapcollect.py --jobs 40 --groups pair --depth 4 --trace
python mapcollect.py --jobs 10 --count 8 --first 4 --stall 3 --stall-rank 2 --deadline 0.02
python mapcollect.py --jobs 100000 --handler nop --count 120 --no-check
```

`--groups` is `fixed` (`--first`, `--count`), `random` (first and count uniform),
`pair` (two halves of the grid, alternating) or `rotate` (disjoint groups of `--count`
cores in turn). `--depth` is the number of mailbox slots the cores scan and the
dispatcher uses (default 4). Outputs go to a ring of `--out-tensors` tensors of
`--out-pages` tiles; only jobs whose pages no later job overwrote are checked, and
reading back runs at about 1 MB/s in slow dispatch. `--trace` adds the per-command
phases (microseconds after the first member took the command: last member took it,
last outputs written, leader pushed the completion) and counts consecutive jobs on
disjoint groups that were in flight at once. A group inside one grid row completes in
one level (every member reports to the first member); a group that spans rows completes
through row leaders (the runtime chooses per command).

The JSON line reports microseconds per job by the dispatcher's clock and by the
host's, the submit-to-completion latency, the Python time per command in `submit` and in
the batched stores, and every core's end state. Its `ok` is true when every core ended,
no job and no compute check is wrong, and the run ended cleanly (with `--stall`: in the
deadline, naming the stalled core); the script exits non-zero without it.

## Measured

On one p150a (card 0), 2026-11-13, AICLK 1350 MHz, X280 at 1000 MHz (the session's
default), with

```
python mapcollect.py --jobs 1000 --groups random --out-tensors 32
```

1,000 jobs of 64 input tiles on random groups, 61,720 member commands, up to 4 in flight:
1.751 µs per job by the dispatcher's clock and 2.078 by the host's; from submit to
completion 4.2 µs per command on average (13.3 at most); every one of the 1,000 jobs
checked against numpy, 0 wrong (500 bit-exact, the others within a relative RMS error
of 0.0019 of the total); every core ended cleanly. The wall time, 74 s, is mostly the
outputs' read-back in slow dispatch: 32 outputs of 4 MB, about 71 s, read in chunks of
2 MB with a line per chunk (about one a second). At the 1750 MHz opt-in (host load 8 to 14): 1.655
µs per job by the kernel's clock, 1.985 by the host's, 4.5 µs per command (14.5 at
most).
