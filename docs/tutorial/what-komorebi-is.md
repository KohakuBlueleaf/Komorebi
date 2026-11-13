---
title: "What Komorebi is"
part: 1
summary: "a whole small program, shown and run but not yet explained, and the list of lessons"
---

# What Komorebi is

Komorebi lets you run your own C++ code on the cores of a Tenstorrent Blackhole card and
control it from Python.

A Blackhole card is not a graphics card. It holds 120 small computers, called Tensix
cores, on one chip. Each core has its own memory and runs its own program. Nothing
spreads your work over them for you: you decide what each core does and when. Komorebi is
the small set of tools for that. You write what one core does in C++, and from Python
you tell the cores when to do it.

## A whole program

Here is a complete program, and what it printed on a card. You do not need to understand
it yet: lessons 4 to 8 take it apart, one piece at a time. Read it once and look at what
it does.

=== "add.py"

    ``` { .python }
    --8<-- "tutorial/01_first_look/add.py"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/01_first_look/kernels/writer.cpp"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/01_first_look/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/01_first_look/kernels/compute.cpp"
    ```

Run `python add.py` and it prints:

```{ .text .km-output }
--8<-- "tutorial/01_first_look/output.txt"
```

Python sent two numbers to core 0 of the card, three times. The core added them, in C++,
on the card, and sent the sum back. Python printed it. The addition did not happen in
Python.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "host", "x": 0, "y": 10, "w": 150, "h": 70, "t": "Python\non your computer", "gem": ""}, {"id": "core", "x": 290, "y": 10, "w": 150, "h": 70, "t": "core 0 of the card\nadds the numbers", "gem": "amber"}], "arrows": [{"from": "host", "to": "core", "kind": "command", "t": "40 and 2", "dy": -14, "dx": -30}, {"from": "core", "to": "host", "kind": "reply", "t": "the sum, 42", "dy": 28, "dx": -62}], "narrow": {"boxes": [{"id": "host", "x": 0, "y": 0, "w": 170, "h": 56, "t": "Python\non your computer"}, {"id": "core", "x": 0, "y": 110, "w": 170, "h": 56, "t": "core 0 of the card\nadds the numbers", "gem": "amber"}], "arrows": [{"from": "host", "to": "core", "kind": "command", "t": "40 and 2", "dy": 0, "dx": -100}, {"from": "core", "to": "host", "kind": "reply", "t": "the sum, 42", "dy": 0, "dx": 10}]}}
</script>
<figcaption>What the program does: numbers go to a core of the card, the sum comes back.</figcaption>
</figure>

Four files for one addition looks like a lot. Two of the C++ files do nothing at all, and
lesson 4 says why they have to be there. The tutorial will make each file earn its place.

## What this tutorial is

The tutorial has six parts, and every part ends with something you can do:

1. **One core.** Run your own code on one core and get an answer back.
2. **Many cores.** Run one program on many cores and give each its share of the work.
3. **Data.** Move arrays between the card's big memory and the cores, and compute on them.
4. **Cores talking to cores.** Send data between cores without going through the big memory.
5. **The other cores.** Write C code for the card's other kind of core, and let it drive the first kind.
6. **Practice.** Measure, find a bug, read a whole program, and write one of your own.

The [list of all lessons](index.md) is on the left of every page.

## How a lesson works

Every lesson teaches one thing. It says what the thing is, then shows its general form
with the parts you fill in, then explains each part, and only then shows a complete
program that uses it. Under every program is what it printed on a card. The numbers that
depend on the machine (a time, for example) are different on yours, and the lesson says so.

A lesson uses only words that an earlier lesson explained. If a word is new, it is
explained where it first appears.

Next: [[setup]] gets your machine ready and runs this program yourself.
