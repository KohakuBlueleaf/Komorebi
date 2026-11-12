---
title: Komorebi Documentation
summary: Home for the programming model, the design of the dispatch kernel and the Tensix runtime, installation from a clone, and development conventions.
tags:
  - overview
  - docs
---

# Komorebi Documentation

Komorebi is a programming model and a runtime for the Tensix grid of Tenstorrent
Blackhole cards. A program is written as straight-line threads; a thread has a program
counter and a place, stays on its core until it needs state held on another core, then
moves there or sends a value there. A resident worker on every Tensix core runs the
threads, and a small kernel on the card's X280 cores decides what to run.

Nothing here is released. The concept pages describe the design and each says whether
it is implemented; today none of it is. Everything that exists is the repository
skeleton: the Python package with its environment check, build and layout helpers, and
the empty directories of the native layer.

## Pick a path

| You are... | Start here |
|---|---|
| **Trying to build it** | [install](install.md) |
| **Learning the model** | [the programming model](concepts/model.md) |
| **Reading the design of the card side** | [dispatch kernel](concepts/dispatch.md) · [Tensix runtime](concepts/runtime.md) |
| **Running a program on a card** | [host API](reference/host.md) · [01_mapcollect](../examples/01_mapcollect/) |
| **Changing the code** | [development](development.md) |

## Documentation structure

### Start

- [Install](install.md): from a clone, against a tt-metal source tree.
- [Development](development.md): repository layout, checks, conventions and known limits.

### Reference

- [Host API](reference/host.md): `Program`, `Session` and `Run`; errors; builds.

### Concepts

Mental models: why things are the way they are. Every page carries a status line.

- [The programming model](concepts/model.md): threads with a program counter and a
  place, the breakpoints move, emit and wait, homes, and how space and time are both
  written.
- [The dispatch kernel](concepts/dispatch.md): the fixed kernel on one X280 cluster,
  its services and engines, command sources and the work clusters.
- [The Tensix runtime](concepts/runtime.md): the resident worker, the 64-byte record,
  teams and completion, and static and dynamic breakpoints.

### Reference

- [Micro-op streams](streams.md): the upload format, the builder, the check, the
  simulator and the error reasons of the programs the dispatch kernel runs.

## Codebase map

```text
src/komorebi/                  the package you import
  layout.py                    the grid, its panels and teams of cores
  build.py                     out-of-tree builds against a tt-metal tree
  stats.py  power.py           statistics; power sampling
  cli.py  tools/               the kmr command; the C and C++ formatter
  manifest/  lang/             reserved: records, homes and segments; the thread language
src/komorebi_native/           the native layer (README inside)
examples/  projects/           runnable examples; larger programs
experiments/  ref/             standalone proofs; CPU references
tests/                         core/ (CPU only)
docs/                          this tree
packaging/                     the tt-metal version Komorebi is developed against
```

The full layout and the checks: [development](development.md).
