---
title: "A whole map"
part: 3
summary: "read, compute and write over a core group: every core maps its share of the tiles"
---

# A whole map

A map applies the same operation to every tile of a tensor. Here the cores of a group each
take their share of the tiles ([[partitions]]) and run the whole read, compute and write
path of Part 3 on it, at the same time. This lesson relies on [[core-group]],
[[partitions]], [[computing-on-a-tile]] and [[circular-buffers]].

## Why a map

Until now one core worked on a handful of tiles. A real tensor has thousands. Each tile
is independent of the others, so the tiles can be shared among the cores of a group and all
the cores work at once: a group of 12 does the work of one core in about a twelfth of the
time. This is the pattern most programs on the card are made of.

## The general form

In every one of the three kernels, inside the function you give to `run`:

```cpp
const uint32_t rank = name.me() - km::first(command);
const km::Partition share = km::partition(TILES, rank, km::count(command));
for (uint32_t index = 0; index < share.len; index++) {
    STEP for tile share.start + index
}
```

- `TILES` is how many tiles the tensor has. It travels as a command argument.
- The three lines are lessons 11 and 12 together. Every core works out the same share for
  itself, with no message to any other core.
- `STEP` is the reader's, the compute kernel's or the writer's step: [[three-kernels]] and
  [[computing-on-a-tile]].
- The three kernels run the same loop count, `share.len`, so that every tile the reader
  pushes is one the compute kernel and the writer take.
- A command's group chooses how many cores share the tiles. The kernels do not change.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "c0", "x": 0, "y": 0, "w": 140, "h": 56, "t": "core 0, rank 0\nshare: tiles 0 to 14", "small": true, "gem": "amber"}, {"id": "c1", "x": 150, "y": 0, "w": 140, "h": 56, "t": "core 1, rank 1\nshare: tiles 15 to 29", "small": true, "gem": "amber"}, {"id": "c2", "x": 300, "y": 0, "w": 140, "h": 56, "t": "core 2, rank 2\nshare: tiles 30 to 44", "small": true, "gem": "amber"}, {"id": "c3", "x": 450, "y": 0, "w": 140, "h": 56, "t": "core 3, rank 3\nshare: tiles 45 to 59", "small": true, "gem": "amber"}, {"id": "src", "x": 0, "y": 96, "w": 590, "h": 40, "t": "source tensor, first half: tiles 0 to 59 (second half: 60 to 119)", "small": true}, {"id": "dst", "x": 0, "y": 176, "w": 590, "h": 40, "t": "destination tensor: tiles 0 to 59", "small": true}], "arrows": []}
</script>
<figcaption>A group of 4 cores and 60 tiles: each core's share is 15 tiles, and they work on their own part of the tensor at the same time.</figcaption>
</figure>

## One line at a time

```{ .cpp }
--8<-- "tutorial/22_a_whole_map/kernels/reader.cpp:run"
```

The reader works out its share, then for each tile of it reads the tile from the first
half of the source and the tile `tile_count` further on, from the second half, into the two
input buffers.

```{ .cpp }
--8<-- "tutorial/22_a_whole_map/kernels/compute.cpp:run"
```

The compute kernel has the same loop around the body of [[computing-on-a-tile]]. It does
not need the tile numbers, only how many tiles: `share.len`.

```{ .cpp }
--8<-- "tutorial/22_a_whole_map/kernels/writer.cpp:run"
```

The writer writes the tile `first_tile + share.start + index` of the destination. The
`first_tile` is an argument that shifts the output, so the program can keep several results
in one tensor.

```{ .python }
--8<-- "tutorial/22_a_whole_map/a_whole_map.py:map"
```

The same command with the group of 1, 4 and then 12 cores. Each writes its result to a
different third of the destination: a tensor write needs the cores to be idle, and a loaded
program never is, so the program keeps the three results apart and checks them at the end.

## The whole program

=== "a_whole_map.py"

    ``` { .python }
    --8<-- "tutorial/22_a_whole_map/a_whole_map.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/22_a_whole_map/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/22_a_whole_map/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/22_a_whole_map/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/22_a_whole_map/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/22_a_whole_map/output.txt"
```

Sixty tiles took 38 microseconds on one core, 11 on four and 5.4 on twelve: about 3.4 and
7 times faster. Not 4 and 12 times: every command also pays a fixed cost to reach the
cores and to hear that they are done. The last line is the program's check: all three
results are the sum of the two source tensors.

## Common mistakes

- Different loop counts in the three kernels. If the reader's share and the writer's share
  differ, one waits for a tile that never comes, and the command ends with
  `CommandTimeout`.
- Writing the tensor while the program is loaded. `destination.write(...)` waits for every
  core to be idle, and never returns: it fails after a minute. Make tensors before `load`.
- More cores than tiles. The last cores get `share.len == 0` and their loops run zero times;
  the command still finishes.

That is the end of Part 3. A core can now read tiles from DRAM, work on them with the
compute kernel and the matrix unit, write them back, and a group can do that for a
whole tensor. Part 4 lets cores pass tiles to each other, without a trip to DRAM.

Next: [[why-cores-talk]].
