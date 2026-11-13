---
title: "Broadcast"
part: 4
summary: "one core sends each of a series of tiles to the whole row of cores with one write"
---

# Broadcast

A broadcast sends the same tile from one core to every other core of a row (or a column)
with a single write over the NoC. This lesson relies on [[port]], [[reduce]] and
[[circular-buffers]].

## Why a broadcast

Sometimes every core needs the same data: the same weights, the same row of a matrix. If
each core read it from DRAM, it would be read from DRAM as many times as there are cores. The NoC can do better: the sender reads a tile once and writes
it to a whole **rectangle** of cores in one go, a **multicast**.

A port ([[port]]) goes to one receiver. A broadcast goes to many, and it handles a
whole series of tiles, called **blocks**, so that the sender can read the next block while the
receivers work on the last. The receivers tell the sender when they have finished with a
block; the sender writes into a slot of the receivers' buffer again only when all have.
The number of blocks that may be in flight at once is the **window**.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "dram", "x": 0, "y": 0, "w": 150, "h": 44, "t": "DRAM: 4 tiles"}, {"id": "c0", "x": 0, "y": 84, "w": 150, "h": 56, "t": "core 0, rank 0\nthe sender", "gem": "amber", "here": true}, {"id": "rest", "x": 260, "y": 84, "w": 200, "h": 56, "t": "cores 1 to 5\nthe receivers", "gem": "amber"}], "arrows": [{"from": "dram", "to": "c0", "kind": "data", "t": "read", "dx": 8, "dy": 2}, {"from": "c0", "to": "rest", "kind": "broadcast", "t": "one multicast write", "dy": -10, "dx": -60}]}
</script>
<figcaption>Core 0 reads from DRAM; the NoC copies each block to cores 1 to 5 with one write.</figcaption>
</figure>

## The general form

Reader of every core of the broadcast, in the function given to `run`:

```cpp
km::Broadcast NAME(reader, command, FIRST_WORD);
if (!NAME.ok() || !NAME.included()) {
    return;
}
for (uint32_t block = 0; block < COUNT; block++) {
    if (!NAME.sender()) {
        if (!NAME.arrive(block)) { return; }
        continue;
    }
    const uint32_t address = NAME.reserve(block);
    ... read the block into address ...
    if (!NAME.cast(block)) { return; }
}
NAME.finish();
```

Compute kernel of every core:

```cpp
km::broadcast_receive(compute, BUFFER, PAGES);
... use the block, in BUFFER ...
km::broadcast_done(BUFFER, PAGES);
...
km::broadcast_end(compute, BUFFER, PAGES, WINDOW, COUNT);
```

- `FIRST_WORD` is where, among the program's arguments for this kernel, the eleven numbers
  of the broadcast start; the host makes them.
- `ok()` says that the numbers match the command, and `included()` that this core is in the
  broadcast. Both are checks; a core that fails either has nothing to do.
- `sender()` is true on rank 0, the core where the blocks start. Every other core
  `arrive`s: waits until block `block` has been written to its buffer.
- `reserve(block)` gives the sender the address of the slot for this block, after waiting
  until every receiver has finished with the block that was there before. `cast(block)`
  waits until the reads have landed, writes the block to every receiver with a multicast,
  and marks it valid.
- `finish()` waits until the compute kernel has taken every block, and ends the broadcast.
- `broadcast_receive` waits in the compute kernel until the block is in the buffer.
  `broadcast_done` says it has finished with it, which is what lets the sender go on, and
  `broadcast_end` runs after the last block.
- A broadcast runs in one row inside one panel of the card ([[the-card]]), or in one column.

## One line at a time

The host first works out the eleven numbers for each core:

```{ .python }
--8<-- "tutorial/26_broadcast/broadcast.py:plan"
```

`session.nocs()` is the card's map of the NoC. `layout.broadcast_args(...)` computes, for
each core of the group, the numbers `km::Broadcast` needs: its rank, where the sender
is and the corners of the rectangle of receivers on the NoC. The arguments are
`(0, 0, 6, 1)`, the rectangle (first column, first row, columns, rows), `"row"`, the NoC the
reader uses (0), and `(0, 0, 2, 1)`: broadcast number 0, buffer 0, a window of 2 blocks,
1 page a block. `arguments_of(core)` returns the three argument lists of a core, in the
order reader, writer, compute. The reader's are these eleven numbers.

```{ .cpp }
--8<-- "tutorial/26_broadcast/kernels/reader.cpp:run"
```

The reader is the general form: the sender reads each block of the source into the slot
that `reserve` gives it and casts it; the others arrive.

```{ .cpp }
--8<-- "tutorial/26_broadcast/kernels/compute.cpp:run"
```

Every core, the sender too, adds the blocks it receives into one tile and packs the sum.

```{ .python }
--8<-- "tutorial/26_broadcast/broadcast.py:cast"
```

One command, `exclusive=True`, to the six cores.

## The whole program

=== "broadcast.py"

    ``` { .python }
    --8<-- "tutorial/26_broadcast/broadcast.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/26_broadcast/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/26_broadcast/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/26_broadcast/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/26_broadcast/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/26_broadcast/output.txt"
```

Only core 0 read the source. The script checks, for each of the six cores, that the tile it
wrote to the destination is the sum of the four blocks of the source: every core received
every block.

## Common mistakes

- A group that is not the rectangle the host's numbers were made for. The numbers name the
  group (`km::Broadcast::ok()` is false otherwise), and the core does nothing.
- A broadcast across two panels of the card, or across a row that is not in one panel. The
  host's `broadcast_args` refuses a row whose cores are not next to each other on the NoC.
- Forgetting `broadcast_done` in the compute kernel. The sender never gets the markers
  that free a slot, and gives up waiting: the card stops the command at its deadline.
- A buffer that does not hold `WINDOW * PAGES` pages. The buffer of a broadcast is used only
  through these calls and must have exactly that many.

Next: [[routed-message]] sends to a core that the data picks.
