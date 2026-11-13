---
title: Tutorial
summary: A step-by-step tutorial of the Komorebi programming model, for a reader who has never written a Tenstorrent kernel.
---

# Komorebi tutorial

This tutorial teaches the Komorebi programming model from zero. It assumes you know C++
and Python and what a multi-core machine is. It does not assume you have ever written a
kernel for a Tenstorrent card: the first page explains what a Tensix core, a tile and a
circular buffer are before any code uses one.

You need a Blackhole P150A card and a machine set up as the [install guide](../install.md)
says. The examples run on the card and check their results against numpy; there is no
online run button and no emulator. Each page shows the output of a real run.

## How to read it

Read top to bottom the first time, and type the examples. Each page is one topic and
stands alone afterwards, so you can come back to the one you need. Every page has the
same parts:

1. what the topic is, in a few sentences, with a figure where a picture is shorter;
2. the calls of the topic in a table;
3. one or two small, complete examples. **Click the `+` marks on the code** for the
   explanation of that line; the tabs show the kernels, the thread and the Python file
   of one program;
4. **Run it**: the command and the output of a real run, with what differs on your card;
5. **Notes**: limits and common mistakes.

All code on these pages is included from the folders of the repository's `tutorial/`
directory, where each example runs as it is shown. The figures with **Previous** and
**Next** buttons draw the card and walk through who sends what to whom.

## The chapters

| | Chapter | |
|---|---|---|
| **Getting started** | [1. What Komorebi is](01-what-komorebi-is.md) | the card, a Tensix core, a kernel, a command |
| | [2. Environment setup](02-environment-setup.md) | `kmr doctor`, the first run |
| | [3. Program structure](03-program-structure.md) | the folder, `Program`, `Session`, `Run` |
| | [4. A first kernel](04-a-first-kernel.md) | one core, one handler, one result |
| **Tensix cores** | [5. Handlers and commands](05-handlers-and-commands.md) | several handlers, many commands in flight |
| | [6. Buffers and data](06-buffers-and-data.md) | DRAM, L1, circular buffers, NoC reads and writes |

The chapters on core groups and partitions, the compute cores, messages between cores,
streams and graphs, X280 threads, errors, measuring, a whole program and a table of
every primitive follow.
