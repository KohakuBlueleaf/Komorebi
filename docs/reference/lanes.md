---
title: Lane broadcast
summary: One sender hands a sequence of blocks to the other cores of a row or a column of a team, with a window of blocks in flight; the device calls, the host layout, the rules a program follows, and what a stalled member leaves.
tags:
  - reference
  - device
---

# Lane broadcast

Status: implemented for the resident runtime; run on one card under the test feeder
(`examples/03_broadcast`).

A lane is a run of cores of one team in one grid column, or in one grid row inside one
panel. One core of the lane, the sender, reads block k (P pages) from DRAM into slot
k mod B of its lane buffer and multicasts it to the same slot on every other core of
the lane, then stamps a VALID word on each. Each receiver hands the block to its
compute kernel and, when compute has popped it, stamps its own word in the sender's
DONE-BY table. The sender writes slot k mod B again only when every receiver has
popped the block that was there. The sender's own compute takes every block too.

A program has up to two lanes at once (lane id 0 and 1), each with its own buffer,
window B, VALID word and DONE-BY table: a matrix product sends A blocks along rows and
W blocks down columns.

## Which kernel runs a lane

| NoC | Kernel | Sender | Direction of the multicast |
|---|---|---|---|
| NoC0 | reader (RISCV_0) | top of a column, left end of a row | +y, +x |
| NoC1 | writer (RISCV_1) | bottom of a column, right end of a row | -y, -x |

Every core of a lane runs the lane in the same kernel, so a lane buffer has one
producer per core. The sender sits where its NoC's direction starts; a multicast from
the other end would go the long way round the torus. The receivers' 4-byte DONE stamps
go against the direction; a window of two or more hides them.

## Device calls

```c++
#include "komorebi/device/km_lane.hpp"

// Reader or writer, inside the segment body, once per record:
km::Lane ln(rd, r, kArgLane);        // rd: the km::Reader or km::Writer; r: the record;
                                     // kArgLane: index of the lane's first program word
if (!ln.ok()) return;                // the lane's words do not fit this record or buffer
if (ln.member()) {
    for (uint32_t k = 0; k < n; k++) {
        if (ln.sender()) {
            uint32_t addr = ln.reserve(k);   // slot k mod B, the same address everywhere
            if (addr == 0) return;
            // read block k's P pages into addr (noc_async_read_page ...)
            if (!ln.cast(k)) return;         // reads landed, receivers done with k - B,
                                             // multicast, VALID, push to own compute
        } else if (!ln.arrive(k)) {          // VALID >= k + 1, push to own compute
            return;
        }
    }
    if (!ln.finish()) return;        // compute popped every block, padding, last stamps
}
```

```c++
// Compute:
km::lane_take(cp, cb, P);            // block k's P pages, read by index 0 .. P - 1
km::lane_done(cb, P);                // done with block k (the DM kernel stamps for it)
km::lane_end(cp, cb, P, B, n);       // after block n - 1: pops the padding
```

Every call is bounded by the run's wait limit and checks the tile's stop word. A call
that returns false or 0 has recorded the give-up; the kernel leaves the body.

## Rules for a program

1. Blocks of a record go in order k = 0, 1, ..., n - 1, on every core of the lane, and
   compute takes and pops every one of them, then calls `lane_end`.
2. The lane buffer is a circular buffer of exactly B * P pages, used only through
   these calls. `finish` pads it to whole windows, so every record starts at slot 0
   and slot addresses are the same on every core.
3. A record with lanes names exactly the team the lane words were built for, and
   carries `KM_F_EXCL`: overlapping teams are never in flight together.
4. Window B: 1 to 8. Receivers per lane: up to 15. Blocks per record: up to 32,767.
5. A member reports only after its compute has popped every block (`finish` checks it
   for its own kernel).

## Host: lane words

`komorebi.layout` computes the lanes as pure functions:

| Function | Gives |
|---|---|
| `lanes(grid, panels, rect, along, noc)` | the lanes of a rectangle (x0, y0, columns, rows), `along` "column" or "row", cores in rank order (rank 0 sends) |
| `lane_rect(lane, noc0, noc_xy, noc)` | the multicast corners of the receivers; refuses a lane whose cores are not consecutive in NoC0 |
| `lane_args(grid, panels, team, rect, along, noc, noc0, noc_xy, (lane, cb, B, P))` | per member of the team, the 11 words of `km::LaneArgs` (rank 0xFF outside the rectangle) |
| `blackhole_noc0_x(harvest_mask)` | NoC0 x of each logical column, from the chip's Tensix harvesting mask |
| `panels(noc0_x)` | runs of columns with consecutive NoC0 x |

The words go into the reader's or writer's program arguments, in the field order of
`km::LaneArgs`; `examples/03_broadcast/broadcast.py` shows the whole host side.

## When a member stalls

A receiver's stamps are sent from inside its lane calls (`arrive`, `finish`),
so a receiver whose data-movement kernel stops calling them stops stamping, even for
blocks its compute pops afterwards. The sender's wait for its stamp
passes the wait limit: it records the give-up and stops its tile; the other cores of
the lane give up waiting for the next VALID. Each core's lane area keeps the first
give-up: where (`lane_done`: the sender waiting for stamps or for its own NoC barriers,
`lane_valid`: a receiver waiting for a block, `lane_free`: waiting for its own compute
or for its last stamps to leave, `lane_begin`: bad words), the lane and the block. The record misses its deadline,
the feeder or the dispatch kernel reports the timeout, and every kernel returns; no
reset is needed.

## Control block

Two lane areas of 0x140 bytes from `KM_CTL_LANE` (0xA00), in 16-byte slots: VALID,
the source slot of this core's outgoing stamp, the first give-up (where | lane << 8,
block, sequence number), statistics (blocks, wait cycles), and the DONE-BY words of
ranks 1 to 15. Stamps carry the record's sequence number in the high half, so a word
left by an earlier record never counts.
