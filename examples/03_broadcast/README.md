# 03 broadcast

One rectangle of cores shares two kinds of blocks without every core reading them from
DRAM: A blocks travel along the rectangle's rows, W blocks down its columns. The first
core of each row (and the last core of each column) reads its blocks from DRAM and
multicasts them to the rest of its broadcast; every core combines the two blocks it holds. This
is the data movement of a matrix product that reads every byte from DRAM once.

## What each core does

Each core of the rectangle has a row and a column, and loops over `blocks` blocks:

1. The first core of its row reads a block of A from DRAM and multicasts it to the other
   cores of the row; every other core of the row waits for it.
2. The last core of its column reads a block of W from DRAM and multicasts it to the other
   cores of the column; every other core of the column waits for it.
3. It adds the tiles of the two blocks it holds to its sum.

After the `blocks` blocks, each core writes its sum to the output.

Each multicast of a row's block is one broadcast: one multicast of the block
into a slot of the receivers' broadcast buffer, then a VALID marker; each receiver writes a marker to the
sender when its compute kernel has finished with the block, and the sender reuses a
slot only after every receiver has marked it. A broadcast holds B blocks in flight.

## How it maps onto the runtime

| Step | Kernel | Call (`komorebi/device/movement/km_broadcast.hpp`) |
|---|---|---|
| rows: the block of A goes out, the others wait | reader, NoC0 (runs +x: the sender is the row's left end) | `broadcast.reserve(block)`, DRAM reads into the slot, `broadcast.cast(block)`; `broadcast.arrive(block)`; `broadcast.finish()` |
| columns: the block of W goes out, the others wait | writer, NoC1 (runs -y: the sender is the column's bottom end) | the same calls on the column broadcast |
| adding the tiles | compute | `km::broadcast_receive`, tiles by index, `km::broadcast_done`; `km::broadcast_end` after the last block |

A row broadcast never crosses a panel boundary (a run of columns with consecutive NoC0 x):
on this card a row of 12 columns is two broadcasts of 6, each with its own sender. The host
computes the broadcasts, their senders and multicast rectangles (`komorebi.layout.broadcast_args`)
and passes each core its broadcast words as program arguments.

| Handler | Arguments | What a core does |
|---|---|---|
| `CAST` | a0 `row_address` (A), a1 `column_address` (W), a2 `output_address`, a3 `blocks`, a4 `stalled_rank` (a test), a5 `stall_block`, a6 `first_page` (output page of rank 0), a7 `data_blocks` per broadcast (block k reads data block k mod `data_blocks`) | the steps above; cores outside the rectangle only report |
| `READ` | the same | every core of the group reads its own blocks from DRAM (the comparison) |

## Run

The driver loads the program with `komorebi.Session` and submits one `CAST` (or `READ`)
command per job with `Run.submit(..., exclusive=True)`: the commands reach the grid
through the dispatcher on X280 cluster 0. A command that uses broadcasts is exclusive,
so the kernel never starts it while another command on an overlapping group runs (the
two would share the broadcast buffers). Setup is that of
[01_mapcollect](../01_mapcollect/README.md#what-it-needs).

```
python broadcast.py --rect 0,0,12,10 --blocks 16 --pages 4 --window 2 --jobs 8
python broadcast.py --rect 0,0,12,10 --blocks 16 --pages 4 --window 2 --read
python broadcast.py --rect 4,0,4,3 --blocks 6 --pages 2 --jobs 2 --stall-job 0 \
    --stall-rank 13 --stall-block 2 --wait 0.05 --deadline 0.5 --again
python broadcast.py --rect 4,0,4,3 --blocks 5 --pages 2 --jobs 3 --empty-job 1
python broadcast.py --rect 4,0,2,2 --blocks 4 --pages 16 --jobs 2 --first-blocks 4100
```

`--rect first_x,first_y,columns,rows` places the rectangle; its group is the smallest range of the
core order that covers it. Inputs are integers in [-2, 2], so every checksum is exact
and the driver compares each core's tile with numpy bit for bit. With
`--stall-job`, that job's core (group index) stops taking row blocks at the given block
and waits for the stop flag: the broadcast's sender gives up waiting for its marker, the
other cores of the broadcast give up waiting for the next block, and at the command's
deadline the dispatcher gives it up, resets the cores still in it or whose kernels ended
and loads them again (`recoveries` in the JSON names them); the next job runs on them.
A run without the reset lists (`recovery_refused` says why), or whose core to reset
other commands in flight also hold, ends in `CommandTimeout` instead, which names the
handle and the cores that did not report. Every kernel returns;
`broadcast_gave_up` names, per core, the broadcast, where it waited and the block. `--again` then loads the program again in the
same session and runs the clean case. With `--empty-job`, that job multicasts zero blocks on
every broadcast (every checksum is zero) between jobs that multicast the full count;
`--empty-every n` does that for every nth job. With
`--first-blocks N`, job 0 multicasts N blocks per broadcast, block k being data block k mod
`--blocks`: a command that moves many pages through a broadcast from little host data.

The JSON line reports, per run, the dispatcher's error and command count, the
host's microseconds per job, the blocks every broadcast delivered, every core's end
state, `ranks_bad`, and `ok` per run and for all of them: every core ended, every job
but the stalled one right on every core of the group, a run with a stalled job recovered
the stalled core (or ended in the deadline), a clean one with no
core and no broadcast given up. The script exits non-zero without it.

`--feeder` runs the same commands on the test feeder (`tests/hardware/feeder/`): the
last core of the grid submits them and no X280 is involved, so the rectangle must leave
that core out (12 x 9 at most). The measurements below were taken that way.

## Measured: broadcasts against every core of the group reading DRAM itself

One p150a, AICLK 1350 MHz, the 12 x 9 rectangle (108 cores; 9 column broadcasts of 9 cores
and 18 row broadcasts of 6), 64 blocks per broadcast and job, 4 jobs, every core of the group checked
(0 wrong), the feeder's clock per job. "Delivered" counts both blocks into every core of the group;
in `READ` each core of the group reads them from DRAM, in `CAST` only the 27 senders do.

| block | window | `CAST` us/job | per broadcast | delivered | DRAM read | `READ` us/job | delivered (= DRAM read) |
|---|---|---|---|---|---|---|---|
| 2 KiB | 1 | 78.8 | 1.66 GB/s | 359 GB/s | 45 GB/s | 74.1 | 382 GB/s |
| 2 KiB | 4 | 71.4 | 1.84 GB/s | 397 GB/s | 50 GB/s | 69.5 | 407 GB/s |
| 24 KiB | 1 | 289.3 | 5.44 GB/s | 1,174 GB/s | 147 GB/s | 738.8 | 460 GB/s |
| 24 KiB | 4 | 250.8 | 6.27 GB/s | 1,355 GB/s | 169 GB/s | 744.0 | 457 GB/s |

At 24 KiB blocks the broadcasts deliver 2.5 to 3 times what the cores get by reading DRAM
themselves, which stays near 460 GB/s whatever the window. At 2 KiB both take the same
time within 6% (`READ` slightly ahead): the job is then bound by something both modes
share, the per-block steps and this example's compute (two tile adds per page), not by
DRAM (an inference; not separated here). A window of 4 gains 9 to 15% over 1 in `CAST`.
So a program broadcasts when blocks are large and shared by many cores; with blocks of
one tile, reading DRAM directly costs no more.

## Over two cards

`islands.py` runs the same kernels on one flat group of both islands
(`Session(cards=[0, 1], links=1)`, 238 ranks: each island's last core is its link core).
The host lays the rectangle over the two grids (`komorebi.layout.island_broadcast_args`):
"below" stands island 1's grid under island 0's, so the rectangle's columns span the
islands; "beside" puts it to the right, so its rows do. A broadcast that spans islands is
a broadcast an island part: the sender casts in its part and sends each block once
by `km::write_l1` to the first core of the other part, whose `arrive` takes it and casts
it in that part as the sender would; the markers of that core go back the same way. The
kernels call the same `reserve`, `cast`, `arrive` and `finish`. The tensors are a copy
on each island; each island writes its cores' checksums into its own copy of the output.
On one card the script runs the same 6 x 4 rectangle at the bottom of the grid.

```
python islands.py              # cards 0 and 1: "below" and "beside"
python islands.py --cards 0    # card 0 alone
```

Blackhole p150a cards, one link, a 6 x 4 rectangle, 8 blocks of 2 tiles (4 KiB) a
broadcast, window 2, 8 `CAST` commands a case on the whole group, each submitted by the
host (a part to each island's dispatcher) and waited; the median from submit to the last
event, every core's checksum exact:

| Case | Spanning broadcasts | µs a round |
|---|---|---|
| cards 0 and 1, "below" (rows 8 and 9 of island 0, 0 and 1 of island 1) | the 6 columns | 705.7 |
| cards 0 and 1, "beside" (columns 9 to 11 of island 0, 0 to 2 of island 1) | the 4 rows | 535.5 |
| card 0 alone (120 ranks) | none | 186.1 |

Each crossing block costs two `km::write_l1` calls (the block, then its VALID line) and a
marker back, each landed before the call returns (about 12 µs a call between two cards),
so a broadcast across islands is paced by those calls, not by the multicast (an
inference; not separated here).

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `writer.cpp`, `compute.cpp` | the three kernels: the row broadcast, the column broadcast and the output, the checksum |
| `kernels/broadcast.h` | handlers, buffers, arguments, and the two block loops (multicast, read) shared by reader and writer |
| `broadcast.py` | builds the broadcasts and commands, runs them through a session (or the feeder), checks every core of the group |
| `islands.py` | the same kernels on one flat group over two cards (and on one), checked against numpy |
