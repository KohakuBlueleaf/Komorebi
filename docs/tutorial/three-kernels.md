---
title: "The three kernels"
part: 3
summary: "the two data-movement kernels as two loops that run at the same time"
---

# The three kernels

The data-movement kernel `reader.cpp`, the compute function and the data-movement kernel
`writer.cpp` are three separate programs that run at the
same time on one core, each in a loop of its own, and they meet only in the buffers in
L1. This lesson relies on [[reading-from-dram]] and [[writing-to-dram]].

## Why three

The core has five CPUs ([[the-card]]) and each can run only its own code. If one CPU had
to read a tile, work on it and write it out, it would do the three in turn, and while it
waited for DRAM the arithmetic would stand still. With a CPU for each job, `reader.cpp`
can be fetching tile 3 while the arithmetic works on tile 2 and `writer.cpp` sends out tile
1.

The price is that the three have to be told when to go: `writer.cpp` must not write a tile
`reader.cpp` has not finished reading. That is what the buffer's four calls do. The
`reserve` and `cb_push_back` of `reader.cpp` and the `wait_front` and `cb_pop_front` of
`writer.cpp` are the handshake between them ([[reading-from-dram]], [[writing-to-dram]]). None of the
three knows the others' code; each knows only the buffer.

So far the compute kernel did nothing. [[computing-on-a-tile]] puts it between `reader.cpp`
and `writer.cpp`. In this lesson only `reader.cpp` and `writer.cpp` work, and the new thing
is that each has a loop.

## The general form

In `reader.cpp`, and in `writer.cpp` in the same way:

```cpp
for (uint32_t index = 0; index < COUNT; index++) {
    STEP
}
```

- `COUNT` is how many tiles this command moves: here an argument of the command.
- `STEP` in `reader.cpp` is `reserve`, `noc_async_read_page`, `noc_async_read_barrier`,
  `cb_push_back`; in `writer.cpp` `wait_front`, `noc_async_write_page`,
  `noc_async_write_barrier`, `cb_pop_front`.
- The two loops have the same `COUNT`. Each tile that `reader.cpp` pushes is a tile that
  `writer.cpp` pops. If the counts differ, one of them waits for ever.
- The loops run at the same time. `reader.cpp`'s second iteration can start while `writer.cpp`
  is still writing the first tile, as soon as the buffer is free.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "reader", "x": 0, "y": 0, "w": 170, "h": 72, "t": "reader.cpp\nfor each tile:\nreserve, read, push", "gem": "amber"}, {"id": "buffer", "x": 230, "y": 10, "w": 130, "h": 52, "t": "the buffer\nin L1: 1 tile"}, {"id": "writer", "x": 420, "y": 0, "w": 170, "h": 72, "t": "writer.cpp\nfor each tile:\nwait, write, pop", "gem": "amber"}], "arrows": [{"from": "reader", "to": "buffer", "kind": "data", "t": "push", "dy": -10, "dx": -14}, {"from": "buffer", "to": "writer", "kind": "data", "t": "wait", "dy": -10, "dx": -14}], "narrow": {"boxes": [{"id": "reader", "x": 0, "y": 0, "w": 190, "h": 72, "t": "reader.cpp\nfor each tile:\nreserve, read, push", "gem": "amber"}, {"id": "buffer", "x": 0, "y": 122, "w": 190, "h": 52, "t": "the buffer\nin L1: 1 tile"}, {"id": "writer", "x": 0, "y": 224, "w": 190, "h": 72, "t": "writer.cpp\nfor each tile:\nwait, write, pop", "gem": "amber"}], "arrows": [{"from": "reader", "to": "buffer", "kind": "data", "t": "push", "dy": 2, "dx": 8}, {"from": "buffer", "to": "writer", "kind": "data", "t": "wait", "dy": 2, "dx": 8}]}}
</script>
<figcaption>Two loops, one buffer. `reader.cpp` pushes a tile when the buffer is free, `writer.cpp` waits for one.</figcaption>
</figure>

## One line at a time

```{ .cpp }
--8<-- "tutorial/18_three_kernels/kernels/reader.cpp:run"
```

Compared to [[reading-from-dram]], the body is inside a `for` loop, and the tile read is
`first_tile + index`. The first tile and the count are arguments 1 and 4 of the command.

```{ .cpp }
--8<-- "tutorial/18_three_kernels/kernels/writer.cpp:run"
```

`writer.cpp`'s loop is the same shape: the destination's first tile is argument 3 and the
count is argument 4, the same count as `reader.cpp`'s.

```{ .python }
--8<-- "tutorial/18_three_kernels/three_kernels.py:copy"
```

`copy(first_tile, tile_count)` is a small function of the program that sends the command
and returns the microseconds the command took, from the event's `microseconds`. The first line sums
eight commands of one tile each. The second sends one command of eight tiles.

## The whole program

=== "three_kernels.py"

    ``` { .python }
    --8<-- "tutorial/18_three_kernels/three_kernels.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/18_three_kernels/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/18_three_kernels/kernels/reader.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/18_three_kernels/kernels/writer.cpp"
    ```

The compute file is the empty one.

```{ .text .km-output }
--8<-- "tutorial/18_three_kernels/output.txt"
```

All eight tiles arrive, in both ways, which the script checks. The times differ a little on
every run. The one command of eight tiles took less than half the time of the eight
commands of one tile: the tiles took the same work either way, and each command costs a
trip from the dispatcher to the core and back, which the eight commands paid eight times.

## Common mistakes

- Different counts in `reader.cpp` and `writer.cpp`. If `reader.cpp` reads 8 and `writer.cpp`
  writes 4, the command ends with `reader.cpp` waiting for room: `CommandTimeout`
  ([[when-it-goes-wrong]]).
- Expecting `reader.cpp` to finish all its tiles before `writer.cpp` starts. With a buffer of
  one tile they take turns, tile by tile; the next lesson is about how to let them overlap.

Next: [[circular-buffers]] gives the buffer more than one tile of room.
