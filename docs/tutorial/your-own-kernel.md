---
title: "Your own kernel"
part: 6
summary: "changing a working program by one thing at a time, with the answer worked out in numpy first"
---

# Your own kernel

You write a kernel of your own by changing a kernel that works, one thing at a time, with
the right answer worked out in numpy before the card runs. This lesson changes the program of
[[a-whole-program]] so that it sums the squares of the tiles. It relies on
[[a-whole-program]], [[finding-a-bug]] and [[measuring]].

## Why change, and not write

A new kernel written from nothing has many places to be wrong at once: the buffers, the
partition, the reduction, the arithmetic. If it does not work, nothing says which. A program
that works, changed in one place, has one place to look. And a check that is made before
the change tells you the change did what you meant.

## The method

1. Copy the folder of a program that works. Run it, to be sure that it works here.
2. Decide the one change, and say what the answer will be after it. Compute that answer in
   numpy, from the same input.
3. Make the change. Change the handler's name if the work is no longer what the name says.
4. Run. The program compares the card's answer with numpy's and says where they differ.
5. If they differ, go back to what you changed. If you want to change something else, go to
   step 2.

Two rules keep the steps short. Change only one thing between two runs. And check with
numbers that cannot hide a mistake: small whole numbers, whose sums `bfloat16` holds
exactly, and not random fractions that you can only compare "about".

## One line at a time

The change is in the compute kernel. In [[a-whole-program]] each tile was added to the sum as
it was. Now each tile is first multiplied by itself:

```{ .cpp }
--8<-- "tutorial/40_your_own_kernel/kernels/compute.cpp:run"
```

The three lines that are new are `const uint32_t into = ...`, `mul_binary_tile_init()` and
`mul_binary_tile(into, into, into)`. `into` is the register the tile goes to: 0 for the first
tile, 1 for the others. `mul_binary_tile(a, b, c)` multiplies the tiles in registers `a` and
`b`, number by number, into register `c`; with the same register for all three, it squares the
tile where it lies. The addition that follows is the old one. The reader, the writer, and the
reduction between cores are unchanged; only the handler's name is new,
`sum_squares`.

```{ .python }
--8<-- "tutorial/40_your_own_kernel/your_own_kernel.py:sum"
```

On the Python side, two things changed: the name of the handler, and the expected answer,
which is now the sum of the squares. The input is the integers -2 to 2, so every square is at
most 4 and the total of 48 of them is at most 192, which `bfloat16` holds exactly.

The check also changed: when the answers differ, the program says how many numbers differ
and shows the first. That is the line to read when step 4 fails.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "start from the program that works\nand run it"},
  {"t": "decide one change and work out\nthe answer in numpy"},
  {"t": "make the change and run", "gem": "amber"},
  {"t": "same as numpy: done, or the next change\ndifferent: look at the change you made", "gem": "sage"}],
 "back": {"from": 3, "to": 1, "t": "next change"}}
</script>
<figcaption>The loop of changing a kernel.</figcaption>
</figure>

## The whole program

=== "your_own_kernel.py"

    ``` { .python }
    --8<-- "tutorial/40_your_own_kernel/your_own_kernel.py"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/40_your_own_kernel/kernels/compute.cpp"
    ```

The other kernels are those of [[a-whole-program]], with the handler renamed. Run it from
the folder `tutorial/40_your_own_kernel`:

```bash
python your_own_kernel.py
```

```{ .text .km-output }
--8<-- "tutorial/40_your_own_kernel/output.txt"
```

It worked, and the times are those of the sum: squaring costs the cores almost nothing next to
reading the tiles.

## When the check fails

To see what a failed check looks like, the same program was run with the three lines of
`mul_binary_tile` taken out of the kernel, so that it only adds, and numpy still expected
the squares. The program ended with:

```text
1 cores: squares of 48 tiles added in  25.6 us
1024 of 1024 numbers differ from numpy; the first:
  the card made -17.0, numpy made 103
```

All 1024 numbers differ, and the card's first number is negative, which a sum of squares
can never be. So the squares are missing, and the place to look is the change you just made.
A wrong answer in every number usually points at the work done on every tile; a wrong answer
in a few numbers points more often at the places where the cores meet.

## Common mistakes

- Changing two things at once. If the answer is wrong, you do not know which.
- Working out the expected answer after seeing the card's. It is then easy to believe the
  card's.
- Checking with random fractions. A difference in the last digit looks like an error or like
  rounding, and you cannot tell which. Use small whole numbers.
- Keeping the old handler name for new work. `sum_tiles` that sums squares misleads the next
  reader, and you.

Next: [[every-primitive]] lists every call that this tutorial used, with its lesson.
