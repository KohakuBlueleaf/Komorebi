---
title: Broadcast
summary: One sender hands a sequence of blocks, and with each one a word if it wants, to the other cores of a row or a column of a group, with a window of blocks in flight; a broadcast without a buffer casts words alone; the device calls, the host layout, the rules a program follows, and what a stalled core leaves.
tags:
  - reference
  - device
---

# Broadcast

Status: implemented for the persistent-kernel runtime; run on one card through a `Session` and
`Run`, under the dispatcher, and under the test feeder
([03_broadcast](../../examples/03_broadcast/): `python broadcast.py --rect 0,0,12,10
--blocks 16 --pages 4 --window 2 --jobs 8`). A cast that carries a word, with a block or
alone: [15_leader](../../examples/15_leader/) (`python leader.py --rect 0,0,12,10
--casts 25000 --jobs 40`).

A broadcast is a run of cores of one group in one grid column, or in one grid row inside one
panel. One core of the broadcast, the sender, reads a block (`pages` pages) from DRAM into
slot `block % window` of its broadcast buffer and multicasts it to the same slot on every
other core of the broadcast, then writes a VALID marker on each. Each receiver hands the
block to its compute kernel and, when compute has popped it, writes its own marker in the
sender's DONE-BY table. The sender writes a slot again only when every receiver has
popped the block that was there. The sender's own compute takes every block too.

A cast may also carry one 32-bit word, which every receiver takes when it arrives: the
leader of a group decides from data it has seen what the group does next and tells it
in the same cast. A broadcast without a buffer casts words alone: no tiles, no
circular buffer, no compute.

A program has up to two broadcasts at once (broadcast id 0 and 1), each with its own buffer,
window, VALID word and DONE-BY table: a matrix product sends A blocks along rows and
W blocks down columns.

## Which kernel runs a broadcast

| NoC | Kernel (by default) | Sender | Direction of the multicast |
|---|---|---|---|
| NoC0 | reader (RISCV_0) | top of a column, left end of a row | +y, +x |
| NoC1 | writer (RISCV_1) | bottom of a column, right end of a row | -y, -x |

A program that gives its kernels the other NoCs (`km.Program(noc=...)`,
[Movement](movement.md#the-noc-of-each-cpu)) runs a NoC0 broadcast in its writer and a
NoC1 broadcast in its reader; the words name the NoC, and a broadcast started on a
kernel of the other NoC gives up at its start.

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

A cast with a word, and the word on every core of the broadcast:

```c++
if (broadcast.sender()) {
    // reserve and read the block as above (none on a broadcast without a buffer)
    if (!broadcast.cast(block, word)) return;  // the block and the word, VALID
} else if (!broadcast.arrive(block)) {         // VALID >= block + 1, and its word
    return;
}
uint32_t taken = broadcast.word();  // the word of this block, on the sender too;
                                    // 0 when the block was cast without one
```

```c++
// Compute:
km::broadcast_receive(compute, cb, pages);  // a block's pages, read by index 0 .. pages - 1
km::broadcast_done(cb, pages);              // done with the block (the DM kernel writes its marker)
km::broadcast_end(compute, cb, pages, window, blocks);  // after the last block: pops the padding
```

Every call times out at the run's wait limit and checks the tile's stop flag. A call
that returns false or 0 has recorded the give-up; the kernel leaves the body.

| Call | What it does |
|---|---|
| `cast(block)` | as before: the block, then VALID; the block carries no word |
| `cast(block, word)` | the same, and `word` rides in the VALID write. Without a buffer: the word alone, no `reserve` before it |
| `arrive(block)` | waits for VALID >= `block + 1`; takes the block's word when it carries one |
| `word()` | the word of the block cast or arrived last; 0 when that block carried none |

## Valid when sent

```c++
km::Broadcast broadcast(reader, command, kArgBroadcast);
broadcast.valid_when_sent();  // before the first cast of the command
```

By default the sender waits for every receiver's acknowledgement of a block before it
multicasts the block's VALID. After `valid_when_sent()` the block's multicast and its
VALID are one linked NoC transaction, the VALID its last request, with no wait between
them; before the next block's multicast the sender waits for the last one's
acknowledgements, so one transaction of the broadcast is in flight at a time. Word-only
casts are unchanged. It is refused at build in dynamic NoC mode; the block and its
VALID always go from the same CPU, on its one NoC, to the same rectangle.

What it rests on: the chip's NoC documentation orders the two packets in transit (same
source, rectangle and NoC, one static virtual channel) but states no order in which a
receiver's L1 takes two writes to different addresses. tt-metal's matrix product and
Kohaku-TT's kernel rely on it, and Komorebi offers it as an option; the acknowledged
form is the one with a documented guarantee.

Measured on the dense linear kernel (one p150a, single calls; the output after a loop
of about 9 s checked, its sampled rows the same as the acknowledged form's): 4,480.4 us
a call at 8192 cubed against 4,488.1, 34,522.4 at 16384 cubed against 34,544.8. The
product is bound by compute, so the acknowledgement's round trip was not on its path.

## A word

The VALID marker and the word go in one 16-byte multicast write into the receivers'
VALID slot: the marker, the word, and whether the cast carries one. A receiver reads the
three in one poll and takes them only when the marker is the same before and after, so
a word is never seen without its block. A receiver that took a word marks it in the
sender's area (a 4-byte write beside its DONE-BY word), and the sender writes the next
VALID only after every receiver has marked the last word: a word is never overwritten
before every receiver took it, whatever the window. A cast without a word adds no such
wait, so a program that casts no words runs exactly as before.

Measured on one p150a (AICLK 1,350 MHz), 10^6 casts of a word alone, every word of every
core checked: 0.54 µs a cast along a row of 6, 0.60 µs down a column of 10 (9
receivers), 0.60 µs with twelve columns casting at once; with a tile (2 KiB) and a word
along a row, 1.31 µs. No word was wrong or late in 1.5 × 10^8 checks
([15_leader](../../examples/15_leader/)).

## A broadcast without a buffer

`window` and `pages` 0 in the broadcast words (`(broadcast, 0, 0, 0)` on the host) make a
broadcast that casts words alone. Its casts are `cast(block, word)` with no `reserve`;
`cast(block)` and `reserve` give up (`broadcast_begin`). Receivers `arrive` and read
`word()`; nothing reaches compute, and the program needs no circular buffer for it.
`finish` is still called on every core. Each cast waits until every receiver took the
last word, so one cast costs a round trip to the farthest receiver.

**The hazard.** At `finish` a receiver clears its VALID. The caster does not wait for
that clear, so the first cast of its next command on the same cores can land before it:
the clear then wipes the new VALID, that receiver never sees the cast, and the command
gives up at the wait limit.

**The rule.** A command that casts a word on a broadcast is not followed at once by a
command that casts a word on the same broadcast's cores; something between them (the
host's wait for the first command's event, or a command of other cores) gives every
receiver the time to clear.

**The guard.** `km.Program(..., broadcast_wait=True)` (and the same keyword at a class's
making) builds the kernels with `KM_BROADCAST_WAIT` 1: each receiver marks the caster
after its clear, and the caster waits at `finish` for every mark under the core's wait
limit and stop, so the rule is not needed. It is off by default (`KM_BROADCAST_WAIT` 0:
the kernels as before it). Its cost on one p150a
([17_leader_overlay](../../examples/17_leader_overlay/README.md)): the cast itself
unchanged, a command of 2,000 casts 0.2 % longer, a command that is one cast of a word
4.94 against 5.38 µs (+0.44 µs); the bytes it adds, 176 B of example 17's reader text
(23,632 against 23,808 B) and 576 B of the [dense class](../../experiments/language/dense_class/README.md)'s
five kernels (90,624 against 91,200 B, 64 B over its kernel buffer of 91,136 B).

## Casting an overlay

The word can name an overlay ([Overlays](overlays.md)): the leader picks the code its
group runs next, and every core of the broadcast loads that overlay when it does not hold
it and calls it. A receiver takes the word in `arrive`, before its own load and call, so
the leader's next cast does not wait for them and the leader runs up to one round ahead.
From the cast to the last call done this took 2.26 µs down a column of 10 when every
core loaded the overlay and 1.67 µs when every core held it
([17_leader_overlay](../../examples/17_leader_overlay/README.md)).

## Rules for a program

1. Blocks of a command go in order, block 0 to the last, on every core of the broadcast,
   and compute takes and pops every one of them, then calls `broadcast_end` (on a
   broadcast with a buffer).
2. The broadcast buffer is a circular buffer of exactly `window * pages` pages, used only through
   these calls. `finish` pads it to whole windows, so every command starts at slot 0
   and slot addresses are the same on every core.
3. A command with broadcasts names exactly the group the broadcast words were built for, and
   carries `KM_FLAG_EXCLUSIVE`: overlapping groups are never in flight together.
4. Window (`window`): 1 to 8, or 0 with `pages` 0 for a broadcast without a buffer.
   Receivers per broadcast: up to 15. Blocks (casts) per command: 0 to 32,767, with no
   limit on the pages they carry in all; the buffer, `window * pages` pages, holds at
   most 65,535. A word is 32 bits, one per cast. A broadcast that multicasts no block in
   a command still calls `finish` on every core.
5. A core reports only after its compute has popped every block (`finish` checks it
   for its own kernel).

One caster a rectangle is the protocol: the sender alone multicasts each block, and the
other cores hand it their rows first. Several casters into one rectangle at once were
measured in [experiments/noc/casters](../../experiments/noc/casters/README.md), with the counted
VALID kept inside that experiment. The chip's NoC documentation
(`BlackholeA0/NoC/MemoryMap.md`, `NOC_CMD_PATH_RESERVE`) warns that such unlinked
broadcasts "can lose a flit and hang the NoC". The probe was clean (800,000 casts, 1.44
million multicasts and as many multicast atomic increments into one column, every word
right), which is a count, not that guarantee; in the dense linear layer's prefill the
step was slower at every shape measured, 1.31 to 2.07 times the hand-over's.

## Host: broadcast words

`komorebi.layout` computes the broadcasts as pure functions:

| Function | Gives |
|---|---|
| `broadcasts(grid, grid_panels, rect, along, noc)` | the broadcasts of a rectangle (x0, y0, columns, rows), `along` "column" or "row", cores in rank order (rank 0 sends) |
| `broadcast_rect(broadcast, noc0, noc_xy, noc)` | the multicast corners of the receivers; rejects a broadcast whose cores are not consecutive in NoC0 |
| `broadcast_args(grid, panels, group, rect, along, noc, noc0, noc_xy, (broadcast, cb, window, pages))` | per core of the group, the 11 words of `km::BroadcastArgs` (rank 0xFF outside the rectangle); `(broadcast, 0, 0, 0)` for a broadcast without a buffer |
| `blackhole_noc0_x(harvest_mask)` | NoC0 x of each logical column, from the chip's Tensix harvesting mask |
| `panels(noc0_x)` | runs of columns with consecutive NoC0 x |

The words go into the reader's or writer's program arguments, in the field order of
`km::BroadcastArgs`; `examples/03_broadcast/broadcast.py` shows the whole host side.

## When a core stalls

A receiver's markers, and its marks of the words it took, are sent from inside its
broadcast calls (`arrive`, `finish`), so a receiver whose data-movement kernel stops
calling them stops sending markers, even for blocks its compute pops afterwards. The sender's wait for its marker
passes the wait limit: it commands the give-up and stops its tile; the other cores of
the broadcast give up waiting for the next VALID. Each core's broadcast area keeps the first
give-up: where (`broadcast_done`: the sender waiting for markers or for its own NoC barriers,
`broadcast_valid`: a receiver waiting for a block, `broadcast_free`: waiting for its own compute
or for its last markers to leave, `broadcast_begin`: bad words), the broadcast and the block. The command misses its deadline,
the feeder or the dispatcher reports the timeout, and every kernel returns; no
reset is needed.

## Control block

Two broadcast areas of 0x140 bytes from `KM_CONTROL_BROADCAST` (0xAC0), in 16-byte slots: VALID
(the marker, the word at `KM_BROADCAST_WORD`, 1 at `KM_BROADCAST_CARRIES` when the cast
carries one), the source slot of this core's outgoing writes, the first give-up (where |
broadcast << 8, block, sequence number), statistics (blocks, wait cycles), and the
DONE-BY slots of ranks 1 to 15 (the DONE-BY word, and at `KM_BROADCAST_ARRIVED` the
marker of the blocks whose word the rank took). Markers carry the command's sequence
number in the high half, so a word left by an earlier command never counts. The host's
`state.broadcasts` shows them as `valid`, `word`, `done` and `arrived`.
