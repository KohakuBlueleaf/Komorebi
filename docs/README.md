---
title: Komorebi
summary: Komorebi runs your own C++ and C code on the cores of a Tenstorrent Blackhole card from Python; the tutorial is the place to start.
tags:
  - overview
  - docs
---

# Komorebi

**Move the data, move the thread, or move the code.**

A programming model for a network of independently programmable processors, in which a
thread reads what it needs or goes to where it is. Today it runs on the Tensix grid and
the X280 cores of a Tenstorrent Blackhole card.

Komorebi lets you run your own C++ code on the Tensix cores of a Tenstorrent Blackhole card, and C code on its X280 cores, driven from Python.
At the end of the [tutorial](tutorial/index.md) you can write a program that spreads work over the cores, moves tensors between the card's memory and the cores, and measure and debug it.
You need to know C++ and Python. You do not need to have written a kernel for a card like this one.
Start with [lesson 1, What Komorebi is](tutorial/what-komorebi-is.md); every lesson teaches one thing and ends with a link to the next.
Komorebi 0.1.0 is on PyPI (`pip install "komorebi-tt[runtime]"`), a first release for one card model. Programs run on one P150A card, and every example in the tutorial shows what it printed on one.

## Where things are

- **Getting started**: [install](install.md) Komorebi and check the host and the card.
- **[Tutorial](tutorial/index.md)**: 50 lessons in six parts, from one core to a whole program.
- **Concepts**: why the framework is the way it is.
  [The dispatcher](concepts/dispatch.md), [the Tensix runtime](concepts/runtime.md) and
  [islands](concepts/islands.md) (two cards as one session), then the longer design pages on [micro-op streams](streams.md), [graphs](graphs.md) and [X280 threads](threads.md).
- **Guides**: how to do one thing. [Debug a command that does not finish](guides/debug-a-command-that-does-not-finish.md), [recover from one that does not return](guides/recovery.md), [measure a command](guides/measure-a-command.md) and [development](development.md).
- **Reference**: exact descriptions of what is built: the [host API](reference/host.md), the
  [Tensix runtime](reference/runtime.md), [ports, emit and reduce](reference/ports.md),
  [broadcast](reference/broadcast.md), [Movement](reference/movement.md),
  [dynamic handlers](reference/dynamic.md), the
  [X280 threads](reference/threads.md), [overlays](reference/overlays.md) and [islands](reference/islands.md).
- **Examples**: twenty-two programs of the repository, each with its own page, from the
  [first one](../examples/01_mapcollect/README.md) on; the
  [list](../examples/README.md) says what each one shows.
- **Releases**: [0.1.0](releases/0.1.0.md).

The repository map is in the [README](../README.md#repository-map); the full layout and the
checks are in [development](development.md). Measurements are in the
[experiments](../experiments/README.md) and the [projects](../projects/README.md).
