---
title: "Partitions"
part: 2
summary: "splitting a range of work by rank"
---

# Partitions

A partition is the share of a job that one core of a group takes: the items of the job are
cut into one contiguous piece for each rank, and `km::partition` says where the piece of
a rank starts and how long it is. This lesson relies on [[core-group]] and [[arguments]].

## Why partitions

A group of cores has the same kernel code and the same command, so the cores have to find
different work in it by the one thing that differs, the rank ([[core-group]]). Take a job of
10 items, ten numbers or ten rows of a table, to be done by 4 cores. The cores need rules that
every core can follow alone, without asking the others, and that together give each item
to exactly one core. The rule: cut the items into 4 pieces, in order, as equal as they can
be. 10 does not divide by 4, so the first two ranks take one item more.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "i0", "x": 0, "y": 0, "w": 40, "h": 34, "t": "item\n0", "small": true, "gem": "aquamarine"}, {"id": "i1", "x": 44, "y": 0, "w": 40, "h": 34, "t": "item\n1", "small": true, "gem": "aquamarine"}, {"id": "i2", "x": 88, "y": 0, "w": 40, "h": 34, "t": "item\n2", "small": true, "gem": "aquamarine"}, {"id": "r0", "x": 0, "y": 50, "w": 128, "h": 34, "t": "rank 0", "small": true, "gem": "aquamarine"}, {"id": "i3", "x": 132, "y": 0, "w": 40, "h": 34, "t": "item\n3", "small": true, "gem": "taaffeite"}, {"id": "i4", "x": 176, "y": 0, "w": 40, "h": 34, "t": "item\n4", "small": true, "gem": "taaffeite"}, {"id": "i5", "x": 220, "y": 0, "w": 40, "h": 34, "t": "item\n5", "small": true, "gem": "taaffeite"}, {"id": "r1", "x": 132, "y": 50, "w": 128, "h": 34, "t": "rank 1", "small": true, "gem": "taaffeite"}, {"id": "i6", "x": 264, "y": 0, "w": 40, "h": 34, "t": "item\n6", "small": true, "gem": "coral"}, {"id": "i7", "x": 308, "y": 0, "w": 40, "h": 34, "t": "item\n7", "small": true, "gem": "coral"}, {"id": "r2", "x": 264, "y": 50, "w": 84, "h": 34, "t": "rank 2", "small": true, "gem": "coral"}, {"id": "i8", "x": 352, "y": 0, "w": 40, "h": 34, "t": "item\n8", "small": true, "gem": "sage"}, {"id": "i9", "x": 396, "y": 0, "w": 40, "h": 34, "t": "item\n9", "small": true, "gem": "sage"}, {"id": "r3", "x": 352, "y": 50, "w": 84, "h": 34, "t": "rank 3", "small": true, "gem": "sage"}], "arrows": []}
</script>
<figcaption>Ten items shared by four ranks: three, three, two, two. Colours show which rank takes which item.</figcaption>
</figure>

## The general form

```cpp
const km::Partition share = km::partition(ITEMS, RANK, COUNT);
```

- `ITEMS` is how many items the whole job has.
- `RANK` is the rank of the core that asks, from 0 to `COUNT - 1`.
- `COUNT` is the number of cores in the group. In a kernel it is `km::count(command)`.
- `share.start` is the number of the first item of this rank's piece, counted from 0.
- `share.len` is how many items the piece holds. The pieces of the ranks follow one
  another with no gap and no overlap, and the first `ITEMS % COUNT` ranks have one item
  more than the others.

A kernel then does its work for the items from `share.start` up to, not including,
`share.start + share.len`.

## One line at a time

```{ .cpp }
--8<-- "tutorial/12_partitions/kernels/writer.cpp:run"
```

The writer reads the number of items and a rank from the arguments. Normally the rank is
the one the core has itself (`writer.me() - km::first(command)`, from the last lesson),
but a Python program sees only the answer of the group's first core, which has rank 0
([[the-leader]] explains why). To look at every rank's piece from Python, this program
sends the rank as an argument, and the first core works out the piece of that rank with the
same function. The result word has two parts packed into one 32-bit number: `first_item << 16`
moves the first item to the upper 16 bits, and `| item_count` puts the count in the
lower 16.

```{ .python }
--8<-- "tutorial/12_partitions/partitions.py:shares"
```

Four commands to the group of four cores, one for each rank. Python takes the word apart
again: `word >> 16` is the first item, `word & 0xFFFF` the count. The loop prints the piece
of each rank and collects the items in `covered`.

## The whole program

=== "partitions.py"

    ``` { .python }
    --8<-- "tutorial/12_partitions/partitions.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/12_partitions/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/12_partitions/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/12_partitions/output.txt"
```

The first two ranks take three items, the other two take two, and the last line is the
script's own check: the pieces, put one after the other, are exactly the items 0 to 9.

## Common mistakes

- Using `ITEMS / COUNT` for the piece. With 10 items and 4 cores that is 2 for each, and
  two items are left over. `km::partition` places the remainder.
- More cores than items. With 3 items and 4 cores the last rank gets `share.len == 0`, and
  a kernel that does not check for it works on an item that does not exist.
- Using a core's number where the rank belongs. A group that starts at core 40 has no rank
  40; `km::partition(ITEMS, 40, COUNT)` is a piece that does not exist.

Next: [[the-leader]] is about the one core that speaks for the whole group.
