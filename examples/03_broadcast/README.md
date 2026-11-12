# 03 broadcast

One rectangle of cores shares two kinds of blocks without every core reading them from
DRAM: A blocks travel along the rectangle's rows, W blocks down its columns. The first
core of each row (and the last core of each column) reads its blocks from DRAM and
casts them to the rest of its lane; every core combines the two blocks it holds. This
is the data movement of a matrix product that keeps every byte crossing DRAM once.

## The thread program

```
thread combine(row, col):                     # one thread per core of an R x C rectangle
    sum = 0
    for k in blocks(n):                        # time: n blocks per lane
        if col == first:  a = read_dram(A[row, k]);  emit a -> threads(row, *)  # breakpoint
        else:             a = wait(1)                                           # breakpoint
        if row == last:   w = read_dram(W[col, k]);  emit w -> threads(*, col)  # breakpoint
        else:             w = wait(1)                                           # breakpoint
        sum += tiles(a) + tiles(w)
    write out[row, col] = sum
```

Each `emit ... -> threads(row, *)` is one lane broadcast: one multicast of the block
into a slot of the receivers' lane buffer, then a VALID stamp; each receiver stamps the
sender when its compute kernel has finished with the block, and the sender reuses a
slot only after every receiver has stamped it. A lane holds B blocks in flight.

## How it maps onto the runtime

| Thread program | Kernel | Call (`komorebi/device/km_lane.hpp`) |
|---|---|---|
| rows: `emit a`, `wait(1)` | reader, NoC0 (runs +x: the sender is the row's left end) | `ln.reserve(k)`, DRAM reads into the slot, `ln.cast(k)`; `ln.arrive(k)`; `ln.finish()` |
| columns: `emit w`, `wait(1)` | writer, NoC1 (runs -y: the sender is the column's bottom end) | the same calls on the column lane |
| `sum += ...` | compute | `km::lane_take`, tiles by index, `km::lane_done`; `km::lane_end` after the last block |

A row lane never crosses a panel boundary (a run of columns with consecutive NoC0 x):
on this card a row of 12 columns is two lanes of 6, each with its own sender. The host
computes the lanes, their senders and multicast rectangles (`komorebi.layout.lane_args`)
and passes each core its lane words as program arguments.

| Segment | Arguments | What a member does |
|---|---|---|
| `CAST` | a0 A, a1 W, a2 output, a3 blocks n, a4 member that stalls (a test), a5 its block, a6 output page of member 0 | as the thread program; members outside the rectangle only report |
| `READ` | the same | every member reads its own blocks from DRAM (the comparison) |

## Run

The example runs on the test feeder (`tests/hardware/feeder/`): the last core of the
grid issues the records, no X280 is involved.

```
python broadcast.py --rect 0,0,12,9 --blocks 16 --pages 4 --window 2 --jobs 8
python broadcast.py --rect 0,0,12,9 --blocks 16 --pages 4 --window 2 --read
python broadcast.py --rect 0,0,6,4 --jobs 2 --stall-job 0 --stall-member 1 \
    --stall-block 3 --wait 0.05 --deadline 0.5 --again
```

`--rect x0,y0,columns,rows` places the rectangle; its team is the smallest range of the
core order that covers it. Inputs are integers in [-2, 2], so every checksum is exact
and the driver compares each member's tile with numpy bit for bit. With
`--stall-job`, that job's member (team index) stops taking row blocks at the given block
and waits for the stop word: the lane's sender gives up waiting for its stamp, the
other cores of the lane give up waiting for the next block, the feeder reports the
record's timeout, and every kernel returns; `lane_gave_up` names, per core, the lane,
where it waited and the block. `--again` then runs the clean case in the same device
open.

The JSON line reports, per run, the feeder's status, microseconds per job, the blocks
every lane handed over, every core's end state, and `members_bad`.

## Files

| File | What it is |
|---|---|
| `kernels/reader.cpp`, `writer.cpp`, `compute.cpp` | the three kernels: the row lane, the column lane and the output, the checksum |
| `kernels/broadcast.h` | segments, buffers, arguments, and the two block loops (cast, read) shared by reader and writer |
| `broadcast.py` | builds the lanes and records, runs them on the feeder, checks every member |
