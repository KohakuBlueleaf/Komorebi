# 01 map/collect

One program runs as a persistent kernel on every Tensix core of the card. The dispatcher on X280 cluster 0 takes commands from the host and sends each one to every core.
A command names a handler and a group, cores `[first, first + count)` of the core order
(row by row, x fastest); cores outside the group ignore it. When every core of the group
has finished, the group's first core tells the dispatcher, which tells the host.

| Handler | Arguments | What a core does |
|---|---|---|
| `MAP` | a0 the input (its reference word, `tensor.address`), a1 `tiles` (T), a2 the output (its reference word), a3 `first_page` (output page of rank 0), a4 `slow_cycles`, a5 `slow_rank`, a6 `first_tile` | reads its partition of input tiles `first_tile` to `first_tile` + T - 1 with `km::read_pages` (tiles another island holds are asked for), adds them into one tile (fp32), stores it at page `first_page` + rank, reports (the rank `slow_rank` first holds its tile `slow_cycles` cycles: a test of slow compute) |
| `STALL` | a0 `stalled_rank` | rank a0 waits for the stop flag and does not report; the others report |
| `ALIGN` (a test) | a9 `report_time` (low word of the wall clock; the test feeder sets it) | every core of the group reports at time a9 |
| `LATE` (a test) | a0 `late_rank`, a1 `delay` (cycles) | rank a0 reports a1 cycles after it took the command; the others report at once |
| `NOP` | none | reports |
| `BAD_ITEM` (a test) | a0 `faulty_rank`, a1 `faulty_role` (0 writer, 1 reader) | rank a0 asks for an item that passes the end of its buffer and gives up with `bad_item`; the others report |
| `RESULT` (a test) | a0 `result_base` | each core of the group gives a0 plus its index as the result word; with `KM_FLAG_RESULT` the command is acknowledged with the leader's, a0 |
| `HANG` (a test) | a0 `hung_rank`, a1 `hang_mode` | rank a0 waits for what never comes: 1 its reader for an arrival at port 0; 2 compute for 2 input tiles (the reader sends 1); 3 the writer for 2 output tiles (compute sends 1); the dispatcher gives the command up at its deadline (below, "A command given up") |

The host checks every core's tile against numpy: bit-exact on integer inputs (even
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
    handle = run.submit(
        "map", group=(0, 8), args=[input_address, 64, output_address, 0]
    )
    run.wait(handle, timeout=5.0)  # sends what is queued, waits for the event
    run.stop()  # EXIT to every core, stop the kernel, close the X280 runtime
    print(run.state()[0], run.trace())  # read back only now
```

`reader_args` and `writer_args` are the tensor accessor arguments of the input and
output tensors (`tensor.accessor_args()` of tensors made by
`session.tensor(array, "bfloat16", "tile")`); `input_address` and `output_address` are their
`address`. `mapcollect.py` is this program with the tensors, the groups and the checks
around it.

If a core never reports, the dispatcher's deadline (one second unless `load` says
otherwise) gives the command up: `run.wait` raises `km.CommandTimeout` with `given_up`
True, naming the command, the slot, the sequence number, the group, and where each
core was waiting at the deadline, and the run goes on. A core that waits in a loop of
its own, which polls nothing, cannot give up: the deadline then ends the run, and the
error names the cores that did not report. The session is already closed cleanly by
then.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the program's three kernels: a switch on the handler inside the runtime's loops (`komorebi/device/km_worker.hpp`) |
| `kernels/mapcollect.h` | handlers, buffers and command arguments shared by the kernels |
| `mapcollect.py` | builds the commands, runs them, checks the outputs, prints one JSON line |
| `block.py` | DOT commands that each carry a block of weights (below, "More than ten words") |
| `give_up.py` | HANG commands inspected and given up, MAP commands after them; the cost of an inspect (below, "A command given up") |

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
phases (microseconds after the first core took the command: last core took it,
last outputs written, leader pushed the completion) and counts consecutive jobs on
disjoint groups that were in flight at once. A group inside one grid row completes in
one level (every core of the group reports to the first core); a group that spans rows completes
through row leaders (the runtime chooses per command).

The JSON line reports microseconds per job by the dispatcher's clock and by the
host's, the submit-to-completion latency, the Python time per command in `submit` and in
the batched stores, and every core's end state. Its `ok` is true when every core ended,
no job and no compute check is wrong, and the run ended cleanly (with `--stall`: in the
deadline, naming the stalled core); the script exits non-zero without it.

## More than ten words

A command has ten argument words; `block.py` gives each DOT command a table of
`--words` (default 64) random 32-bit weights w as its block
(`run.submit(..., block=weights)`, [host](../../docs/reference/host.md#a-commands-block)).
Every core of the group reads the table with `reader.block(command)` and
`writer.block(command)`. The writer gives sum of w[i] * (x0 + i), mod 2^32, as the
result (x0 in `a0`); the reader gives up unless the words sum to `a1`. The host checks
every leader's result against numpy, then times 200 commands with and without a block,
submitted before one wait and one at a time.

```
python block.py --jobs 200 --words 64 --count 12
```

Card 0, 2026-11-14, X280 at 1000 MHz, 200 commands on 2, 12 and 120 cores: every
result right; a block costs the host 160 to 210 µs a command when 200 are submitted
before one wait, and 2.0 to 3.0 ms when each is waited for (the DRAM barrier), and the
cores about 1 µs (submit to acknowledgement 2.7 to 3.0 µs, against 1.8 to 2.0 without).

## A command given up

`give_up.py` hangs rank 1 of a group of 2 in each of the three ways HANG has. A
quarter of the deadline later it inspects the group (`run.inspect`,
[host](../../docs/reference/host.md#inspecting-a-run)). After the deadline it expects
`km.CommandTimeout` with `given_up` True, then runs `--maps` MAP commands on the same
cores and checks their tiles bit for bit. Last, it times an inspect of every core, and
four NOP commands in flight alone and with an inspect queued behind them.

```
python give_up.py --deadline 0.5 --rounds 50
```

Card 0, 2026-11-14, X280 at 1000 MHz: the inspect names the wait on rank 1 (`port` for
the reader, `cinput` for unpack, `data` for the writer) and each HANG is given up. An
inspect of 120 cores takes the dispatcher 2.9 ms (about 24 µs a core); NOP commands in
flight take 2.4 µs alone and 8.6 µs with an inspect behind them. Not passing yet: after
the writer's HANG, run third, the first MAP on those cores is given up in turn.

## Measured

On one p150a (card 0), 2026-11-13, AICLK 1350 MHz, X280 at 1000 MHz (the session's
default), with

```
python mapcollect.py --jobs 1000 --groups random --out-tensors 32
```

1,000 jobs of 64 input tiles on random groups, 61,720 core commands, up to 4 in flight:
1.751 µs per job by the dispatcher's clock and 2.078 by the host's; from submit to
completion 4.2 µs per command on average (13.3 at most); every one of the 1,000 jobs
checked against numpy, 0 wrong (500 bit-exact, the others within a relative RMS error
of 0.0019 of the total); every core ended cleanly. The wall time, 74 s, is mostly the
outputs' read-back in slow dispatch: 32 outputs of 4 MB, about 71 s, read in chunks of
2 MB with a line per chunk (about one a second). At the 1750 MHz opt-in (host load 8 to 14): 1.655
µs per job by the kernel's clock, 1.985 by the host's, 4.5 µs per command (14.5 at
most).
