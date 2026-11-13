---
title: "A core group"
part: 2
summary: "the same command on several cores, and each core's rank"
---

# A core group

A command goes to every core of its group, and every core runs the same kernel code on it;
the one thing that tells the cores apart is their **rank**, the core's position in the
group. This lesson relies on [[hello-core]], [[commands]] and [[when-it-goes-wrong]].

## Why a group

One core is a small part of the card: 120 of them are there. Work that is the same for
every core, such as the same arithmetic on a different piece of data, should go to many
cores with one command, not one command each. The group in `submit` is how: `group=(FIRST,
COUNT)` is `COUNT` cores in a row, starting at core number `FIRST`, and all of them get the
one command ([[hello-core]]).

If every core did exactly the same, nothing would be gained from having many. The kernel
needs a way to ask "which of the cores am I?". The rank is the answer.

## The general form

In the kernel, inside the function you give to `run`:

```cpp
const uint32_t rank = name.me() - km::first(command);
```

In Python:

```python
handle = run.submit(KIND, group=(FIRST, COUNT))
```

- `name` is the role object of the file, `reader`, `compute` or `writer`.
- `name.me()` is the number of the core this code runs on, 0 to 119.
- `km::first(command)` is `FIRST`, the first core of the command's group.
- Their difference is the **rank**: 0 for the first core of the group, 1 for the next,
  and so on to `COUNT - 1`. It does not depend on where in the card the group is: a group
  that starts at core 3 has the same ranks 0, 1, 2, 3 as one that starts at core 40.
- `km::count(command)` is `COUNT`, how many cores the group has. The rank is always less
  than it.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "c0", "x": 0, "y": 0, "w": 42, "h": 36, "t": "core\n0", "small": true}, {"id": "c1", "x": 46, "y": 0, "w": 42, "h": 36, "t": "core\n1", "small": true}, {"id": "c2", "x": 92, "y": 0, "w": 42, "h": 36, "t": "core\n2", "small": true}, {"id": "c3", "x": 138, "y": 0, "w": 42, "h": 36, "t": "core\n3", "small": true, "gem": "amber"}, {"id": "c4", "x": 184, "y": 0, "w": 42, "h": 36, "t": "core\n4", "small": true, "gem": "amber"}, {"id": "c5", "x": 230, "y": 0, "w": 42, "h": 36, "t": "core\n5", "small": true, "gem": "amber"}, {"id": "c6", "x": 276, "y": 0, "w": 42, "h": 36, "t": "core\n6", "small": true, "gem": "amber"}, {"id": "c7", "x": 322, "y": 0, "w": 42, "h": 36, "t": "core\n7", "small": true}, {"id": "c8", "x": 368, "y": 0, "w": 42, "h": 36, "t": "core\n8", "small": true}, {"id": "c9", "x": 414, "y": 0, "w": 42, "h": 36, "t": "core\n9", "small": true}, {"id": "c10", "x": 460, "y": 0, "w": 42, "h": 36, "t": "core\n10", "small": true}, {"id": "c11", "x": 506, "y": 0, "w": 42, "h": 36, "t": "core\n11", "small": true}, {"id": "r0", "x": 138, "y": 52, "w": 42, "h": 36, "t": "rank\n0", "small": true, "gem": "amber"}, {"id": "r1", "x": 184, "y": 52, "w": 42, "h": 36, "t": "rank\n1", "small": true, "gem": "amber"}, {"id": "r2", "x": 230, "y": 52, "w": 42, "h": 36, "t": "rank\n2", "small": true, "gem": "amber"}, {"id": "r3", "x": 276, "y": 52, "w": 42, "h": 36, "t": "rank\n3", "small": true, "gem": "amber"}], "arrows": [{"from": "c3", "to": "r0"}, {"from": "c4", "to": "r1"}, {"from": "c5", "to": "r2"}, {"from": "c6", "to": "r3"}]}
</script>
<figcaption>The command <code>group=(3, 4)</code>: four cores, from core 3. Core 3 has rank 0, core 6 has rank 3. The other cores let the command pass.</figcaption>
</figure>

## One line at a time

```{ .cpp }
--8<-- "tutorial/11_core_group/kernels/writer.cpp:run"
```

The writer of every core of the group runs this. The line with `rank` is the one from the
general form. The next line makes a limit from it: the core of rank 0 gets 0, the core of
rank 3 gets three times the first argument. The `while` loop counts up to the limit, as in
[[when-it-goes-wrong]], and takes time in proportion to it. At the end every core returns
`true`, so every core reports.

```{ .python }
--8<-- "tutorial/11_core_group/core_group.py:groups"
```

The same command, with the argument 1,000,000, goes to a group of 1 core, then of 4, then
of 12, each starting at core 0. The loop prints how long each command took from `submit` to
the report, using the event's `microseconds`. `took` is a dictionary made before the loop;
the program keeps each time in it to check them at the end.

A command is done when **every** core of its group has reported. The group of 12 contains
a core of rank 11, which counts to 11 million, about 66 milliseconds, and so the command
takes about 66 milliseconds even though most of the cores finished long before. The cores
worked at the same time; the slowest one set the time.

## The whole program

=== "core_group.py"

    ``` { .python }
    --8<-- "tutorial/11_core_group/core_group.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/11_core_group/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/11_core_group/kernels/writer.cpp"
    ```

The reader and compute files are the empty ones. Run it from the folder
`tutorial/11_core_group`:

```{ .text .km-output }
--8<-- "tutorial/11_core_group/output.txt"
```

The group of 1 core has only rank 0, which counts to 0, so its command took a few
microseconds, and the line shows 0.0 milliseconds. The group of 4 has a core of
rank 3, 3 million counts, 18 milliseconds. The group of 12: 11 million counts, 65. The
numbers differ a little on every run, and the script checks only that more cores took longer.

## Common mistakes

- Taking the rank for the core number. It is the core number minus the group's first core;
  they are equal only when the group starts at core 0.
- Expecting the command to take the average time. It takes the time of the slowest core.
- A group that runs past the last core. `group=(100, 30)` stops at `submit` with
  `ValueError: group (100, 30) is not inside 120 cores`.
- Giving every core of the group the same work by mistake: if the kernel never looks at the
  rank, all of them do everything, and the group is a way to do the same thing 12 times.

Next: [[partitions]] uses the rank to give each core its own piece of a job.
