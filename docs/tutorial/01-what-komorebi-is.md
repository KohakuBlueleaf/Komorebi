---
title: What Komorebi is
summary: The card as a programmer sees it, what is inside a Tensix core, what a kernel and a command are.
---

# 1. What Komorebi is

Komorebi is a way to program a Tenstorrent Blackhole card. You write small C++ kernels
that run on the card's cores, and a Python file that tells the card which kernel to run
and on which cores. This page is the map: what is on the card, who does what, and what
happens between the moment Python submits a command and the moment it gets the answer.
There is no code to type yet.

## The card

A Blackhole P150A has four kinds of parts that a program uses. The figure draws them
as they sit on the chip. Press **Next** to follow one command through the card.

<figure class="km-grid">
<script type="application/json">
{"steps": [
 {"label": "The card. Left and right: two panels of 6 x 10 Tensix cores, 120 in all, numbered 0 to 119 row by row. Grey columns: DRAM, the card's big memory. Centre: the host (your PC, over PCIe) and four X280 clusters.",
  "lit": []},
 {"label": "1. Your Python program submits a command. It travels over PCIe to X280 0, the cluster that runs the dispatcher.",
  "lit": ["host", "x280:0"],
  "arrows": [{"from": "host", "to": "x280:0"}]},
 {"label": "2. The dispatcher sends the command to every Tensix core, one multicast per panel. Each core reads it and finds out whether it is in the command's core group: here cores 0 to 7. The others ignore it.",
  "lit": ["x280:0", "cores:0-7"],
  "arrows": [{"from": "x280:0", "to": "core:5"}, {"from": "x280:0", "to": "core:6"}]},
 {"label": "3. Each core of the group does its part: it reads its data from DRAM over the NoC, works on it in its own L1 memory and writes the result back to DRAM.",
  "lit": ["cores:0-7"],
  "arrows": [{"from": "dram:0:2", "to": "core:27", "kind": "data"}, {"from": "core:6", "to": "dram:9:0", "kind": "data"}]},
 {"label": "4. When a core has finished, it tells the first core of the group (the leader, core 0). Core 0 waits until all eight have reported.",
  "lit": ["cores:0-7"],
  "marks": {"core:0": "0"},
  "arrows": [{"from": "core:3", "to": "core:0", "kind": "reply"}, {"from": "core:7", "to": "core:0", "kind": "reply"}]},
 {"label": "5. The leader acknowledges to the dispatcher, the dispatcher posts an event for the host, and your Python program's wait returns.",
  "lit": ["core:0", "x280:0", "host"],
  "arrows": [{"from": "core:0", "to": "x280:0", "kind": "reply"}, {"from": "x280:0", "to": "host", "kind": "reply"}]}
]}
</script>
</figure>

| Part | How many | What it is |
|---|---|---|
| **Tensix core** | 120 | A small computer that has its own memory and runs its own kernels. The cores are the workers: a program is mostly a set of kernels on them. The next section opens one. |
| **NoC** | 2 networks | The network on the chip: it carries reads and writes between any two tiles of the grid, such as a core and a DRAM bank. Its name is Network on Chip. Every tile has an address on it, two numbers `(x, y)`. |
| **DRAM** | 8 banks | The card's big memory, the place for your input and output data. It sits in the two grey columns. Tensix cores reach it through the NoC. |
| **X280 cluster** | 4 | Each cluster is four 64-bit RISC-V cores with vector units, programmed in C. Cluster 0 runs Komorebi's **dispatcher**, which sends commands to the Tensix cores. Clusters 1, 2 and 3 are free for your own C code, called an X280 thread (chapter 16). |
| **Host** | 1 | Your computer. It runs Python, compiles the kernels, puts data on the card and submits commands. |

## Inside a Tensix core

A Tensix core is not one processor but five small RISC-V cores that share a
memory called **L1**, plus two arithmetic units that the RISC-V cores drive.

| Part | What it does |
|---|---|
| **L1** | 1.5 MB of memory inside the core, as fast as a cache but managed by you. Everything a kernel works on has to be in L1. After the kernels and the runtime take their share, a Komorebi program has about 1.4 MB for its data. |
| **Reader core** (RISC-V core 0) | Moves data into L1: it starts reads on NoC 0. Runs the **reader kernel**. |
| **Writer core** (RISC-V core 1) | Moves data out of L1: it starts writes on NoC 1. Runs the **writer kernel**. |
| **Three compute cores** (unpack, math, pack) | Together they run the **compute kernel**. *Unpack* moves a tile from L1 into the arithmetic unit's registers, *math* tells the unit what to compute, *pack* moves the result back into L1. |
| **Matrix unit and vector unit** | The arithmetic. They work on **tiles**, blocks of 32 x 32 numbers. Chapter 9 explains them. |

Why five cores? So that moving data and computing happen at the same time: the reader
brings the next block while the compute cores work on this one and the writer sends the
last one out. Each core has its own small program, a kernel, and they hand data to each
other through queues in L1 called **circular buffers** ([chapter 6](06-buffers-and-data.md)).

## What a kernel is

A **kernel** is a C++ function `kernel_main` compiled for one of those core types. A
Tensix core runs three of them in a Komorebi program: a reader, a compute kernel and a
writer. Komorebi uses tt-metal, Tenstorrent's open-source runtime, to open the card and
to compile and load kernels.

In plain tt-metal a kernel starts, does one job and ends, and the host launches it again
for the next job. Komorebi's kernels are **persistent**: they are loaded once, and each
runs a loop that waits for a command, runs the part of your code that the command
names, reports that it is done, and waits again. Submitting a command costs a
microsecond or two instead of a launch.

## What a command is

A **command** is a 64-byte message. It names three things:

- a **handler**, a number that picks which piece of your kernel code runs;
- a **core group**, a range of consecutive cores (the first core and how many);
- ten 32-bit **words** of arguments, for example the address of the data.

Every core receives every command. A core that is not in the group ignores it. A core
that is in the group runs the handler in all three of its kernels and reports back.
[Chapter 5](05-handlers-and-commands.md) is about commands and handlers.

## Where the rest of the tutorial goes

| Chapters | What they cover |
|---|---|
| 1 to 4 | The map (this page), setting up, the files of a program, a first kernel |
| 5 to 9 | Tensix cores: handlers and commands, buffers and data, core groups, partitions, the compute cores |
| 10 to 13 | Messages between cores: ports, broadcast, reduce, routed messages |
| 14 to 20 | Deciding on the card: streams, graphs, X280 threads and the messages between X280 cores and Tensix cores |
| 21 to 24 | Errors and hangs, measuring, one whole program, a table of every primitive |

## Run it

The card is the thing to look at. This script opens it and prints what a program can
ask about it:

=== "card_facts.py"

    ``` { .python }
    --8<-- "tutorial/01_the_card/card_facts.py:facts"
    ```

```bash
python card_facts.py
```

```{ .text .km-output }
120 Tensix cores: 12 per row, 10 rows
panels (first and last column): [(0, 5), (6, 11)]
core 0 at NoC0 (1, 2), core 119 at (16, 11)
L1 of a core: 70656 bytes for kernels, 1461120 for buffers
```

These numbers are from a P150A. Another card can have another number of working cores:
a card has some Tensix columns switched off, and the grid reads 12 x 10 on this one.
Chapter 2 sets up the machine so that you can run this.

## Notes

- Komorebi targets the Blackhole P150A. There is no emulator: every example needs a card.
- Cores are numbered 0 to 119 row by row in the order of the figure. A command's core
  group is a range of these numbers, so a group can start at the end of one row and end
  on the next.
- The two panels are separate on the chip, so a command takes two multicasts. You do not
  see this; the dispatcher does it.
- Reference: [the Tensix runtime](../concepts/runtime.md) (design), [the dispatcher](../concepts/dispatch.md),
  [host API](../reference/host.md).
