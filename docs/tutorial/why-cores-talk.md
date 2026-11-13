---
title: "Why cores talk"
part: 4
summary: "each core's L1 is its own; sending data between cores, and the lessons of Part 4"
---

# Why cores talk

A core cannot read another core's L1 by itself, so when one core has something another
needs, it has to be sent. This lesson shows the problem, and lists the ways of solving it
that the rest of Part 4 teaches. It relies on [[the-card]], [[core-group]] and
[[the-leader]].

A program meets its data in one of two ways. Either the data comes to the core that
works on it: ports, reduce and broadcast send tiles to the cores that need them. Or the
work goes to the data: a routed message carries it to the core that holds what it needs.

## Why cores need to talk

In Part 3 every core worked on its own share of tiles, and no core needed anything from
another. That covers a map: each output tile depends on one input tile. It does not cover
a sum of all the tiles, or a result that every core needs a copy of, or a piece of data
that is found out only while the program runs. For these a core has to hand something to
another core.

Going through DRAM works: core A writes a tile, core B reads it. It costs two trips to
DRAM and needs a way for B to know that A has finished. The cores' own network, the NoC,
can carry the data from L1 to L1 directly. Part 4 is about that.

## What a core's L1 holds

First the problem, as a program. A core can keep a number in its own L1. Python can read the L1
of every core, as one tensor with a row for each core. The kernels write into it.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "c0", "x": 0, "y": 0, "w": 120, "h": 44, "t": "core 0\nL1: 0", "gem": "amber", "small": true}, {"id": "c1", "x": 140, "y": 0, "w": 120, "h": 44, "t": "core 1\nL1: 1", "gem": "amber", "small": true}, {"id": "c2", "x": 280, "y": 0, "w": 120, "h": 44, "t": "core 2\nL1: 4", "gem": "amber", "small": true}, {"id": "py", "x": 140, "y": 104, "w": 120, "h": 44, "t": "Python\nreads all", "gem": "sapphire", "small": true}], "arrows": [{"from": "c0", "to": "py", "kind": "reply"}, {"from": "c1", "to": "py", "kind": "reply"}, {"from": "c2", "to": "py", "kind": "reply"}], "narrow": {"boxes": [{"id": "c0", "x": 0, "y": 0, "w": 100, "h": 44, "t": "core 0\nL1: 0", "small": true, "gem": "amber"}, {"id": "c1", "x": 110, "y": 0, "w": 100, "h": 44, "t": "core 1\nL1: 1", "small": true, "gem": "amber"}, {"id": "c2", "x": 220, "y": 0, "w": 100, "h": 44, "t": "core 2\nL1: 4", "small": true, "gem": "amber"}, {"id": "py", "x": 110, "y": 104, "w": 100, "h": 44, "t": "Python\nreads all", "small": true, "gem": "sapphire"}], "arrows": [{"from": "c0", "to": "py", "kind": "reply"}, {"from": "c1", "to": "py", "kind": "reply"}, {"from": "c2", "to": "py", "kind": "reply"}]}}
</script>
<figcaption>Each core holds a number of its own. Python can read all of them. No core can read another's.</figcaption>
</figure>

## One line at a time

```{ .python }
--8<-- "tutorial/24_why_cores_talk/why_cores_talk.py:notes"
```

`session.tensor(..., shard_grid=session.grid)` makes a tensor that lives in the L1 of every
core, not in DRAM: one row of the array in each core, 4 numbers of 4 bytes. `session.grid`
says it is on all 120 cores. The array has 120 rows, one for each core, in the order of the
core numbers.

```{ .cpp }
--8<-- "tutorial/24_why_cores_talk/kernels/writer.cpp:run"
```

`km::l1(address)` turns the address in the command into a pointer to the core's own L1: the
same address on every core, the row of that core. Every core of the group stores its
rank times its rank there.

```{ .python }
--8<-- "tutorial/24_why_cores_talk/why_cores_talk.py:remember"
```

One command to the first six cores, with the address of the tensor as its argument.

```{ .python }
--8<-- "tutorial/24_why_cores_talk/why_cores_talk.py:read"
```

`notes.read()` returns all 120 rows. The first number of the first six rows is what the
cores stored; the others are untouched zeros. Cores 6 and 7 did not run the command.

## The whole program

=== "why_cores_talk.py"

    ``` { .python }
    --8<-- "tutorial/24_why_cores_talk/why_cores_talk.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/24_why_cores_talk/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/24_why_cores_talk/kernels/writer.cpp"
    ```

`reader.cpp` and `compute.cpp` are the empty ones.

```{ .text .km-output }
--8<-- "tutorial/24_why_cores_talk/output.txt"
```

Each core of the group holds its own number, 0, 1, 4, 9, 16 and 25. This is also the tool
that the next lessons use to see what the cores received: a place in L1 that Python can
read. The program checks that cores 6 to 119 did not change.

## The ways cores talk

Two kinds of message, and the lessons for each:

| Lesson | The one idea | Kind |
|---|---|---|
| [[port]] | one core sends a tile to one other core | the receiver is known in advance |
| [[reduce]] | many cores send to one, which combines them | the receiver is known in advance |
| [[broadcast]] | one core sends to every core of its row or column | the receivers are known in advance |
| [[routed-message]] | a message to a core chosen while the program runs | the receiver depends on the data |
| [[forward-and-consume]] | a message that visits several cores before it ends | the receiver depends on the data |
| [[full-buffers]] | what a sender does when the receiver has no room | both |
| [[streams-and-graphs]] | a whole chain of such steps described once | both |

The first three are **channels**: who sends to whom is fixed by the command and the ranks
before anything runs. The last ones are **routed messages**: the data decides.

## Common mistakes

- Reading `notes.read()` while a command is still running. Wait for the event first.
- Expecting a core to find another's number by reading the same address. Every core's
  address is its own.

Next: [[port]] sends a tile from one core to another.
