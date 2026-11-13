---
title: Komorebi Documentation
summary: Home for the programming model, the design of the dispatch kernel and the Tensix runtime, the host API, micro-op streams, installation from a clone, and development conventions.
tags:
  - overview
  - docs
---

# Komorebi Documentation

Komorebi is a programming model and a runtime for the Tensix grid of Tenstorrent
Blackhole cards. The [repository README](../README.md) draws the model and says what
exists today and what is planned; this page lists where each part is described.

Nothing here is released. Host-fed runs on one card are implemented: a program's
resident kernels on every Tensix core, the dispatch kernel on X280 cluster 0, and the
Python host API that issues records to teams and waits for their completion. Micro-op
streams exist as a format, a builder, a check and a simulator on the CPU. Breakpoints
between cores, deciding on the card and the thread language are planned.

## Pick a path

| You are... | Start here |
|---|---|
| **Trying to build it** | [install](install.md) |
| **Learning the model** | [the programming model](concepts/model.md) |
| **Running a program on a card** | [host API](reference/host.md) · [01_mapcollect](../examples/01_mapcollect/) |
| **Writing an X280 thread** | [X280 threads](threads.md) · [reference](reference/threads.md) · [07_work_cluster](../examples/07_work_cluster/) · [08_parallel_thread](../examples/08_parallel_thread/) |
| **Reading the design of the card side** | [dispatch kernel](concepts/dispatch.md) · [Tensix runtime](concepts/runtime.md) · [reference](reference/runtime.md) |
| **Changing the code** | [development](development.md) |

## Documentation structure

### Start

- [Install](install.md): from a clone or a wheel, against a tt-metal source tree;
  running on a card; watching a card's power with `kmr power`.
- [Development](development.md): repository layout, checks, conventions and known
  limits.

### Reference

- [Host API](reference/host.md): `Program`, `Session` and `Run`; errors; builds.
- [Tensix runtime](reference/runtime.md): the device headers; the three kernels'
  classes and calls, the record, every bounded wait, the gave-up and where codes, the
  control block map, and what a program may and may not do.
- [Ports, emit and reduce](reference/ports.md): the device calls that send a value to
  one core, groups of a team, and reduce by owner and by tree.
- [Lane broadcast](reference/lanes.md): one sender to a row or a column of a team;
  the device calls, the host layout, the rules and the give-up.
- [Dynamic segments](reference/dynamic.md): items that move between cores as the data
  decides, through per-core queues in DRAM; quota; the quiet end; give-up reasons.
- [Micro-op streams](streams.md): the upload format, the builder, the check, the
  simulator and the error reasons of the programs the dispatch kernel is to run.
- [Graphs](graphs.md): the graph executor's upload format, builder, check, the
  executor in plain C and in Python, and the simulator.
- [X280 threads](threads.md): the irregular part of a program as C on a work cluster,
  with the kmx library; its regions, its host declaration and its host fallback.
- [X280 threads reference](reference/threads.md): every kmx call and return code,
  the thread's memory layout and result block, the ring line, and the Python side.

### Concepts

Mental models: why things are the way they are. Every page carries a status line.

- [The programming model](concepts/model.md): threads with a program counter and a
  place, the breakpoints `at`, `emit` and `wait`, homes, and how space and time are
  both written.
- [The dispatch kernel](concepts/dispatch.md): the fixed kernel on one X280 cluster,
  its services and engines, command sources and the work clusters.
- [The Tensix runtime](concepts/runtime.md): the resident worker, the 64-byte record,
  teams and completion, and static and dynamic breakpoints.

### Pages by part

One page per part of the framework, or the part listed as missing with the source
directory whose authors write it. "Concept" pages describe the design; "reference"
pages describe what is built.

| Part | Page | State | Written with |
|---|---|---|---|
| Resident worker on the Tensix cores: records, mailbox slots, teams, completion, trace | [Tensix runtime](concepts/runtime.md) (concept); [reference](reference/runtime.md) | written | `src/komorebi_native/include/komorebi/device/` |
| Static breakpoints: lane broadcast along a row or a column | [lane broadcast](reference/lanes.md) | written | `src/komorebi_native/include/komorebi/device/km_lane.hpp` |
| Static breakpoints: `emit`, `arrive`, `take`, reduce by owner and by tree | [ports, emit and reduce](reference/ports.md) | written | `src/komorebi_native/include/komorebi/device/km_port.hpp`, `km_group.hpp` |
| Dynamic breakpoints: `at`, inbox rings, packets, credits | [Tensix runtime](concepts/runtime.md) (concept) | reference page missing | `src/komorebi_native/include/komorebi/device/` |
| Dispatch kernel on X280 cluster 0: rings, slots, deadlines, events, upload region | [dispatch kernel](concepts/dispatch.md) (concept); [kernel README](../src/komorebi_native/x280/kernel/README.md) (reference) | written | `src/komorebi_native/x280/kernel/` |
| Deciding on the card: micro-op streams | [micro-op streams](streams.md) | written | `src/komorebi_native/x280/kernel/`, `src/komorebi/engines/` |
| Deciding on the card: the job queue | [dispatch kernel](concepts/dispatch.md) (concept); [kernel README](../src/komorebi_native/x280/kernel/README.md) (reference, "Job queue") | written | `src/komorebi_native/x280/kernel/` |
| Graph executor | [graphs](graphs.md); [dispatch kernel](concepts/dispatch.md) (concept) | written | `src/komorebi_native/x280/engines/`, `src/komorebi/engines/` |
| User threads on the X280 work clusters | [X280 threads](threads.md); [reference](reference/threads.md) | written | `src/komorebi_native/x280/kmx/`, `src/komorebi/thread.py` |
| Host launcher: `Program`, `Session`, `Run` | [host API](reference/host.md) | written | `src/komorebi/`, `src/komorebi_native/csrc/host/` |
| Python package and C++ headers: install, wheel, builds | [install](install.md), [packaging](../packaging/README.md) | written; the generic kernels (`kernels/`) and CMake helpers (`cmake/`) are not written yet | `packaging/` |
| The runtime on the CPU (harness for the Tensix runtime) | [development](development.md), [runtime harness](../experiments/runtime_harness/README.md) | written | `experiments/runtime_harness/` |

### Measurements

- [Experiments](../experiments/README.md): record landing, record multicasts against
  in-column broadcasts, and the cost of one Tensix tile for a box-tree walk.
- [Projects](../projects/README.md): the rasterization and ray tracing references the
  card versions are compared with.

The repository map is in the [README](../README.md#repository-map); the full layout
and the checks are in [development](development.md).
