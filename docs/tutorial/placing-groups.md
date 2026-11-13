---
title: "Placing groups"
part: 2
summary: "choosing the cores of a group: a row, two rows, a panel, a rectangle"
---

# Placing groups

A group is a run of core numbers, so you place it by choosing where the run starts and how
long it is. This lesson relies on [[core-group]], [[the-leader]] and [[the-card]].

## Why placing matters

Until now every group started at core 0. A program that uses the card well chooses its
cores: a whole row for work that is the same everywhere, the left panel for one stage of
a job and the right for the next, a block of cores for a block of data. A group is always
the numbers `FIRST` up to `FIRST + COUNT - 1`, so what shapes are possible follows from how
the cores are numbered. [[the-card]] gave the numbering: 12 cores in a row, 10 rows, 0 to
119 row by row, the first six of each row in the left panel and the last six in the right.

## The general form

```python
first = ROW * 12 + COLUMN
handle = run.submit(KIND, group=(first, COUNT))
```

- `ROW` is the row, 0 to 9, from the top.
- `COLUMN` is the place in the row, 0 to 11: 0 to 5 is the left panel, 6 to 11 the right.
- `first` is the number of the core at the start of the group. It is the leader.
- `COUNT` is how many cores follow it, as long as the group ends at core 119 or before.
  Within a row `COLUMN + COUNT` must not pass 12 if the group is to stay in its row.
- A shape that is not one run of core numbers is several groups: one command for each,
  all submitted before the first `wait` ([[many-commands]]).

## Pictures of the placements

<figure class="km-grid">
<script type="application/json">
{"steps": [{"label": "A row: group=(24, 12). Core 24 is the first core of row 2, and the leader, marked L.", "lit": ["cores:24-35"], "marks": {"core:24": "L"}}, {"label": "Two rows in one group: group=(24, 24), cores 24 to 47. The leader is core 24 again.", "lit": ["cores:24-47"], "marks": {"core:24": "L"}}, {"label": "Half a row is a group too. The left panel of row 2: group=(24, 6).", "lit": ["cores:24-29"], "marks": {"core:24": "L"}}, {"label": "The right panel of row 2: group=(30, 6). It starts in the middle of the row, with core 30 as leader.", "lit": ["cores:30-35"], "marks": {"core:30": "L"}}, {"label": "A rectangle is not one group: a group is a run of core numbers, and a rectangle's rows are not next to each other in that numbering. It is three groups, one for each row: (30, 6), (42, 6), (54, 6).", "lit": ["cores:30-35", "cores:42-47", "cores:54-59"], "marks": {"core:30": "L", "core:42": "L", "core:54": "L"}}]}
</script>
<figcaption>Five groups on the card. Lit cores are in the group; L marks a leader.</figcaption>
</figure>

## One line at a time

```{ .cpp }
--8<-- "tutorial/14_placing_groups/kernels/writer.cpp:run"
```

Each core of the group sets its own number as the result. Python reads the leader's, which
is the group's first core, so the result shows where a group started.

```{ .python }
--8<-- "tutorial/14_placing_groups/placing_groups.py:groups"
```

`CORES_PER_ROW` is 12. The dictionary lists five placements by name: for each, a
list of groups, each a pair `(first, count)`. The loop sends one command to each group of
the placement, waits for all of them, and prints the leaders' core numbers. The last line
of the loop is the program's own check: each leader is the first core of its group.

## The whole program

=== "placing_groups.py"

    ``` { .python }
    --8<-- "tutorial/14_placing_groups/placing_groups.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/14_placing_groups/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/14_placing_groups/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/14_placing_groups/output.txt"
```

Each command was done, which means every core of every group reported. The last placement
has three groups, so three events and three leaders. The group of two rows, `(24, 24)`,
works like the group of one row: the leader is its first core.

## Common mistakes

- Expecting a rectangle from one group. `group=(30, 18)` is not "6 columns of 3 rows": it
  is cores 30 to 47: the right panel of row 2 and then all of row 3, because the run goes
  on to the next number.
- A group that does not fit the card. `group=(110, 20)` stops at `submit` with
  `ValueError: group (110, 20) is not inside 120 cores`.
- Counting rows or columns from 1. Both start at 0.

That is the end of Part 2. A command can now go to any run of cores, each core knows its
rank, takes its piece of the work by it, and the leader speaks for the group. Part 3 gives
the cores something to work on: data.

Next: [[tiles-and-tensors]].
