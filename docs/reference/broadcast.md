---
title: Broadcast
summary: One sender hands a sequence of blocks to the other cores of a row or a column of a group, with a window of blocks in flight; the device calls, the host layout, the rules a program follows, and what a stalled core leaves.
tags:
  - reference
  - device
---

# Broadcast

Status: implemented for the persistent-kernel runtime; run on one card through a `Session` and
`Run`, under the dispatcher, and under the test feeder
([03_broadcast](../../examples/03_broadcast/): `python broadcast.py --rect 0,0,12,10
--blocks 16 --pages 4 --window 2 --jobs 8`).

A broadcast is a run of cores of one group in one grid column, or in one grid row inside one
panel. One core of the broadcast, the sender, reads a block (`pages` pages) from DRAM into
slot `block % window` of its broadcast buffer and multicasts it to the same slot on every
other core of the broadcast, then writes a VALID marker on each. Each receiver hands the
block to its compute kernel and, when compute has popped it, writes its own marker in the
sender's DONE-BY table. The sender writes a slot again only when every receiver has
popped the block that was there. The sender's own compute takes every block too.

A program has up to two broadcasts at once (broadcast id 0 and 1), each with its own buffer,
window, VALID word and DONE-BY table: a matrix product sends A blocks along rows and
W blocks down columns.

## Which kernel runs a broadcast

| NoC | Kernel | Sender | Direction of the multicast |
|---|---|---|---|
| NoC0 | reader (RISCV_0) | top of a column, left end of a row | +y, +x |
| NoC1 | writer (RISCV_1) | bottom of a column, right end of a row | -y, -x |

Every core of a broadcast runs the broadcast in the same kernel, so a broadcast buffer has one
producer per core. The sender sits where its NoC's direction starts; a multicast from
the other end would go the long way round the torus. The receivers' 4-byte DONE markers
go against the direction; a window of two or more hides them.

## Device calls

```c++
#include "komorebi/device/km_broadcast.hpp"

// Reader or writer, inside the handler body, once per command:
// reader: the km::Reader or km::Writer; command: the command;
// kArgBroadcast: index of the broadcast's first program word
km::Broadcast broadcast(reader, command, kArgBroadcast);
if (!broadcast.ok()) return;  // the broadcast's words do not fit this command or buffer
if (broadcast.included()) {
    for (uint32_t block = 0; block < blocks; block++) {
        if (broadcast.sender()) {
            uint32_t address = broadcast.reserve(block);  // slot block % window, the same
                                                          // address everywhere
            if (address == 0) return;
            // read the block's pages into address (noc_async_read_page ...)
            if (!broadcast.cast(block)) return;  // reads landed, receivers done with the
                                                 // block `window` before, multicast, VALID,
                                                 // push to own compute
        } else if (!broadcast.arrive(block)) {   // VALID >= block + 1, push to own compute
            return;
        }
    }
    if (!broadcast.finish()) return;  // compute popped every block, padding, last markers
}
```

```c++
// Compute:
km::broadcast_receive(compute, cb, pages);  // a block's pages, read by index 0 .. pages - 1
km::broadcast_done(cb, pages);              // done with the block (the DM kernel writes its marker)
km::broadcast_end(compute, cb, pages, window, blocks);  // after the last block: pops the padding
```

Every call times out at the run's wait limit and checks the tile's stop flag. A call
that returns false or 0 has recorded the give-up; the kernel leaves the body.

## Rules for a program

1. Blocks of a command go in order, block 0 to the last, on every core of the broadcast,
   and compute takes and pops every one of them, then calls `broadcast_end`.
2. The broadcast buffer is a circular buffer of exactly `window * pages` pages, used only through
   these calls. `finish` pads it to whole windows, so every command starts at slot 0
   and slot addresses are the same on every core.
3. A command with broadcasts names exactly the group the broadcast words were built for, and
   carries `KM_FLAG_EXCLUSIVE`: overlapping groups are never in flight together.
4. Window (`window`): 1 to 8. Receivers per broadcast: up to 15. Blocks per command: 0 to
   32,767, with no limit on the pages they carry in all; the buffer, `window * pages`
   pages, holds at most 65,535. A broadcast that multicasts no block in a command still calls `finish` on every
   core.
5. A core reports only after its compute has popped every block (`finish` checks it
   for its own kernel).

## Host: broadcast words

`komorebi.layout` computes the broadcasts as pure functions:

| Function | Gives |
|---|---|
| `broadcasts(grid, grid_panels, rect, along, noc)` | the broadcasts of a rectangle (x0, y0, columns, rows), `along` "column" or "row", cores in rank order (rank 0 sends) |
| `broadcast_rect(broadcast, noc0, noc_xy, noc)` | the multicast corners of the receivers; rejects a broadcast whose cores are not consecutive in NoC0 |
| `broadcast_args(grid, panels, group, rect, along, noc, noc0, noc_xy, (broadcast, cb, window, pages))` | per core of the group, the 11 words of `km::BroadcastArgs` (rank 0xFF outside the rectangle) |
| `blackhole_noc0_x(harvest_mask)` | NoC0 x of each logical column, from the chip's Tensix harvesting mask |
| `panels(noc0_x)` | runs of columns with consecutive NoC0 x |

The words go into the reader's or writer's program arguments, in the field order of
`km::BroadcastArgs`; `examples/03_broadcast/broadcast.py` shows the whole host side.

## When a core stalls

A receiver's markers are sent from inside its broadcast calls (`arrive`, `finish`),
so a receiver whose data-movement kernel stops calling them stops sending markers, even for
blocks its compute pops afterwards. The sender's wait for its marker
passes the wait limit: it commands the give-up and stops its tile; the other cores of
the broadcast give up waiting for the next VALID. Each core's broadcast area keeps the first
give-up: where (`broadcast_done`: the sender waiting for markers or for its own NoC barriers,
`broadcast_valid`: a receiver waiting for a block, `broadcast_free`: waiting for its own compute
or for its last markers to leave, `broadcast_begin`: bad words), the broadcast and the block. The command misses its deadline,
the feeder or the dispatcher reports the timeout, and every kernel returns; no
reset is needed.

## Control block

Two broadcast areas of 0x140 bytes from `KM_CONTROL_BROADCAST` (0xA00), in 16-byte slots: VALID,
the source slot of this core's outgoing marker, the first give-up (where | broadcast << 8,
block, sequence number), statistics (blocks, wait cycles), and the DONE-BY words of
ranks 1 to 15. Markers carry the command's sequence number in the high half, so a word
left by an earlier command never counts.
