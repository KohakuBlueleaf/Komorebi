---
title: "Reduce"
part: 4
summary: "many cores send a tile to one core, which adds them: by an owner, and along a tree"
---

# Reduce

A reduce combines one tile from each core of a group into a single tile on one core, here by
adding them, and Komorebi has two ways to route the tiles. This lesson relies on [[port]],
[[core-group]] and [[computing-on-a-tile]].

## Why a reduce

When a job is shared among cores ([[a-whole-map]]), each core ends up with a part of the
answer, and often the answer is the sum of the parts: a total, a dot product, the last
step of a matrix product. The parts have to meet in one place and be added. That is a
reduce: every core sends its part, and one core, rank 0, ends with the sum.

Two ways to route it:

| Form | Who sends to whom | Steps | Slots in the arrival buffer |
|---|---|---|---|
| owner | every core but rank 0 sends straight to rank 0 | 1, rank 0 adds `size - 1` tiles | `size - 1` |
| tree | in round 0 odd ranks send to the even rank before them; in round 1 ranks 2, 6, ... send; and so on | `log2(size)` rounds, each core adds at most one tile per round | one per round, at most 7 |

The owner is the simplest and does all the adding on one core. The tree spreads the adding:
no core adds more than once a round, and the whole takes 3 rounds for 8 cores and 7 for
128. Both use the ports of [[port]]; the sums are the same.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "tree, round 0: ranks 1, 3, 5, 7 send\nto ranks 0, 2, 4, 6", "gem": "amber"}, {"t": "round 1: ranks 2 and 6 send to ranks 0\nand 4"}, {"t": "round 2: rank 4 sends to rank 0", "gem": "amber"}, {"t": "rank 0 holds the sum of all 8 tiles"}]}
</script>
<figcaption>The tree form for 8 cores: three rounds. Each round halves the number of cores that hold a part.</figcaption>
</figure>

## The general form

In `reader.cpp`, `writer.cpp` and the compute kernel, the group and the form:

```cpp
const km::Group group = km::group_of(command, NAME.me());
const km::Arrival arrival = {BUFFER, PAGES, 1};
```

Reader, after reading the core's own tile:

```cpp
km::reduce_arrive(reader, command, arrival, group, FORM);
```

Compute:

```cpp
ok = km::reduce(compute, arrival, group, FORM, [&](uint32_t slot) {
    ...  // add the tile in slot `slot` of the arrival buffer
});
```

Writer:

```cpp
if (group.rank != 0) {
    return km::reduce_emit(writer, command, arrival, group, FORM, OUTPUT, 1);
}
km::reduce_result(writer, OUTPUT, 1);
```

- `km::group_of(command, core)` is the command's group as an object: `group.rank` and
  `group.size` ([[core-group]]).
- `FORM` is `km::kReduceOwner` (0) or `km::kReduceTree` (1).
- `reduce_arrive` makes `reader.cpp` wait at every port a core has to wait at, as in
  [[port]], according to the form.
- `reduce_emit` makes every core but rank 0 send its sum to the core the form says, and
  `reduce_result` makes rank 0 wait for the total.
- The function given to `km::reduce` is the one thing that is yours: how to combine. It is
  called for each tile that arrives, in rank order, whatever order they came in, so the sum
  is the same on every run.

## One line at a time

```{ .cpp }
--8<-- "tutorial/26_reduce/kernels/reader.cpp:run"
```

Each core reads the tile of its rank ([[partitions]], one tile each), then waits for what
the others send.

```{ .cpp }
--8<-- "tutorial/26_reduce/kernels/compute.cpp:run"
```

The compute kernel copies its own tile into register 0. The function given to `reduce`
copies the arrived tile into register 1 and adds it to register 0. When the reduce is
done, register 0 holds this core's sum (the total, on rank 0) and `pack_tile` puts it in
the output buffer.

```{ .cpp }
--8<-- "tutorial/26_reduce/kernels/writer.cpp:run"
```

`writer.cpp` of every core but rank 0 sends the output tile on; rank 0 writes it to the
destination tensor.

```{ .python }
--8<-- "tutorial/26_reduce/reduce.py:reduce"
```

The same command twice, with the group of 8 cores, once with each form. The two forms
write to different tiles of the destination (the argument after its address), so both can
be checked at the end.

## The whole program

=== "reduce.py"

    ``` { .python }
    --8<-- "tutorial/26_reduce/reduce.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/26_reduce/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/26_reduce/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/26_reduce/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/26_reduce/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/26_reduce/output.txt"
```

Both commands put the sum of the 8 tiles in the destination, exactly: the script compares
every number with numpy. With 8 cores there is little to choose between the forms; the tree
pays off with many cores, where one core would have to add a hundred tiles.

## Common mistakes

- An arrival buffer with too few pages. The owner form needs `size - 1` slots, so a group of
  8 needs at least 7 pages; with fewer, `reader.cpp` gives up.
- Forgetting `exclusive=True` ([[port]]).
- A combine function that depends on the order of the tiles, such as a subtraction. The
  order is fixed (rank order), but the owner and the tree combine in different groupings,
  so only an operation that does not care, like addition of whole numbers, gives the same
  result from both.

Next: [[broadcast]] sends one tile from one core to many.
