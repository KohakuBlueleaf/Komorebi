---
title: "The card"
part: 1
summary: "the parts of the card that the next lessons use: Tensix cores, L1, DRAM, the network"
---

# The card

The card is a chip with a grid of 120 small computers, a big memory around them, and a
network that joins them all. This lesson names the parts. It relies on [[what-komorebi-is]]
and has no code to write: a picture, and a short program that asks the card about itself.

## The parts, in one picture

Press **Next** to see each part in turn.

<figure class="km-grid">
<script type="application/json">
{"steps": [
 {"label": "The Tensix cores: 120 small computers in two panels of 6 columns and 10 rows. They are numbered 0 to 119, row by row. Core 0 is at the top left, core 11 at the top right, core 12 at the start of the second row.",
  "lit": ["core:0", "core:11", "core:12"]},
 {"label": "DRAM: the card's big memory, in the two grey columns. Your data lives here when no core is working on it. A core cannot compute on data in DRAM; it has to copy it into its own memory first.",
  "lit": ["dram:0", "dram:9"]},
 {"label": "The X280 cores: four clusters of four, numbered 0 to 3, in the middle. They are bigger computers than the Tensix cores. Komorebi runs a program of its own on cluster 0. A later part of this tutorial lets you run yours on clusters 1 to 3.",
  "lit": ["x280:0", "x280:1", "x280:2", "x280:3"]},
 {"label": "The host: your computer, joined to the card by a cable (PCIe). Your Python program runs here.",
  "lit": ["host"],
  "arrows": [{"from": "host", "to": "x280:0"}]},
 {"label": "The network on chip, the NoC, is not drawn: it runs between all of these. It is what carries a read or a write from a core to DRAM, or to another core.",
  "lit": []}
]}
</script>
<figcaption>The card, drawn as it sits on the chip. Amber: Tensix cores. Blue: X280 cores. Grey: DRAM.</figcaption>
</figure>

- A **Tensix core** is one of the 120 small computers. This is where your code runs, and
  all of this tutorial is about them. There are 12 in a row: the first six are the left
  panel, the last six the right panel. Core `n` is in row `n // 12` and place `n % 12` of
  that row.
- **DRAM** is the big memory, many gigabytes, split in eight banks. It is far from the
  cores and slow compared with the memory inside a core.
- The **NoC** (network on chip) is the network between every core, every DRAM bank and the
  X280 cores. A core uses it to copy data from DRAM to itself, from itself to DRAM, or
  to another core.
- The **X280** cores are the other kind of core. [[hello-thread]] is the first lesson that
  uses them. Until then they are only a place that some of Komorebi runs on.
- The **host** is your computer.

## Inside a Tensix core

A Tensix core is five small CPUs that share one memory.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "l1", "x": 0, "y": 0, "w": 430, "h": 60, "t": "L1: the core's own memory, 1.4 MB", "gem": "amber"}, {"id": "reader", "x": 0, "y": 100, "w": 100, "h": 56, "t": "CPU 1\nmoves data"}, {"id": "writer", "x": 110, "y": 100, "w": 100, "h": 56, "t": "CPU 2\nmoves data"}, {"id": "unpack", "x": 240, "y": 100, "w": 60, "h": 56, "t": "CPU 3\nunpack", "small": true}, {"id": "math", "x": 305, "y": 100, "w": 60, "h": 56, "t": "CPU 4\nmath", "small": true}, {"id": "pack", "x": 370, "y": 100, "w": 60, "h": 56, "t": "CPU 5\npack", "small": true}], "arrows": [{"from": "reader", "to": "l1", "kind": "data"}, {"from": "writer", "to": "l1", "kind": "data"}, {"from": "unpack", "to": "l1", "kind": "data"}, {"from": "pack", "to": "l1", "kind": "data"}], "narrow": {"boxes": [{"id": "l1", "x": 0, "y": 0, "w": 330, "h": 60, "t": "L1: the core's own memory,\n1.4 MB", "gem": "amber"}, {"id": "reader", "x": 0, "y": 110, "w": 100, "h": 56, "t": "CPU 1\nmoves data"}, {"id": "writer", "x": 110, "y": 110, "w": 100, "h": 56, "t": "CPU 2\nmoves data"}, {"id": "unpack", "x": 0, "y": 200, "w": 100, "h": 56, "t": "CPU 3\nunpack"}, {"id": "math", "x": 110, "y": 200, "w": 100, "h": 56, "t": "CPU 4\nmath"}, {"id": "pack", "x": 220, "y": 200, "w": 100, "h": 56, "t": "CPU 5\npack"}], "arrows": [{"from": "reader", "to": "l1", "kind": "data"}, {"from": "writer", "to": "l1", "kind": "data"}, {"from": "unpack", "to": "l1", "kind": "data"}, {"from": "pack", "to": "l1", "kind": "data"}]}}
</script>
<figcaption>One Tensix core: one memory, five CPUs. Two move data, three together do the arithmetic.</figcaption>
</figure>

- **L1** is the core's own memory. A core works on what is in its L1 and on nothing else.
  After Komorebi has taken its share, 1.4 MB are left for your data. L1 is not a cache:
  nothing is copied into it unless a program asks for it.
- Two of the five CPUs move data. Either of them can copy data from DRAM into L1, from L1
  to DRAM, and between the L1 of one core and another's. They are alike, with two
  differences that later lessons use: each of them uses its own NoC, and
  each queue of data in L1 has one CPU that fills it and one that empties it.
  Lesson 4 gives each of them a name.
- The other three CPUs, together with the matrix and vector units that they drive, do the
  arithmetic. They get a lesson of their own in Part 3.

The five CPUs run at the same time, each its own code. That is the reason for a core
having so many: while one moves the next piece of data in, the others work on the piece
before.

## Ask the card

You can ask the card for these facts from Python. One fragment at a time:

```{ .python }
--8<-- "tutorial/03_the_card/card_facts.py:grid"
```

`session.grid` is the number of cores in a row and the number of rows.

```{ .python }
--8<-- "tutorial/03_the_card/card_facts.py:panels"
```

The panels, as the first and the last column of each (two on a P150A, one on a
Wormhole chip).

```{ .python }
--8<-- "tutorial/03_the_card/card_facts.py:l1"
```

`session.l1()` gives two numbers; the second is how many bytes of each core's L1 are left
for your data. The `_` throws the first one away.

All three lines run inside `with km.Session() as session:`, which opens the card. Lesson 4
explains that line. The whole program, and what it printed:

```{ .text .km-output }
--8<-- "tutorial/03_the_card/output.txt"
```

## Common mistakes

- Thinking of a core as one CPU. It is five, and they run different code at the same time.
- Thinking L1 is a cache. Nothing is in a core's L1 unless a program put it there.
- Counting cores from one. The numbers run from 0 to 119.

Next: [[hello-core]] writes the smallest program that runs on a core.
