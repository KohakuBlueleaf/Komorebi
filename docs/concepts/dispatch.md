---
title: The dispatch kernel
summary: A fixed kernel on one X280 cluster that issues work to the Tensix grid, its services and engines, where commands come from, and the other X280 clusters as work clusters.
tags:
  - concepts
  - dispatch
  - x280
---

# The dispatch kernel

Status: implemented on X280 cluster 0 of one card: the host ring, slots, deadlines
and events, micro-op streams and the job queue ([streams](../streams.md)), and graphs
([graphs](../graphs.md)); user threads on work clusters 1, 2 and 3 issue records
through a ring each ([X280 threads](../threads.md)). What is built, in detail, is in the
[kernel README](../../src/komorebi_native/x280/kernel/README.md).

The Blackhole card has, besides its Tensix grid, four clusters of X280 cores (RISC-V
with vector units). Komorebi uses one cluster to decide what the grid runs, and the
others for work that suits a general-purpose core. This page describes the dispatch
kernel and the clusters around it; [runtime](runtime.md) describes the Tensix side.

## The whole machine

```text
 host (Python / C++)                         builds programs and graphs, uploads, starts, reads back
   |  commands into the dispatch cluster's host ring          ^ results pushed to host-visible memory
   v                                                          |
 X280 cluster 0: DISPATCH KERNEL (one fixed image; decides inside)
   kernel loop on core 0, cores 1 to 3 as its threads
   services: grid port, completion table, rings, timers, memory
   engines:  micro-op interpreter, job queue, graph executor      CPU tasks registered in the image
   |  64-byte records, multicast per panel                    ^ completion words and counters, pushed
   v                                                          |
 Tensix grid, 120 cores: ONE RESIDENT WORKER per program
   next record -> segment switch -> run -> complete
   static breakpoints (fixed slot + counter)   dynamic breakpoints (inbox, packets, credits)

 X280 clusters 1 to 3: WORK CLUSTERS (the user's images)
   X280 threads: heavy irregular work, or a decider the engines cannot express
   talk to the dispatch kernel through 64-byte message rings with credits
```

The image on cluster 0 is fixed: it does not change from program to program. What
changes is the data the host uploads for it to run.

## How the kernel behaves

- **It is never interrupted.** It loops. When idle it polls its sources, gated so it
  does not scan every source on every pass.
- **Messages are 64 bytes:** a header word and seven payload words. Messages carry
  request ids, replies are parked at the receiver, senders use credits, and every
  sender has an outbox so two senders never deadlock. Nothing is dropped. This holds
  for the host ring and for the rings to clusters 1 to 3.
- **A unit of work is opaque to the kernel.** A unit is a team of Tensix cores with a
  depth; an instruction is one 64-byte record that the kernel treats as eight opaque
  words. Nothing in the kernel knows what a program computes.
- **Where a stuck node is can be seen.** The kernel keeps an error word and a "where"
  code per loop pass.

The four cores of one X280 cluster are coherent and share memory, and the image is one
bare-metal program, so the kernel needs no memory management, system calls or
preemption.

## Structure of the kernel

- **Kernel loop (core 0).** Single-threaded in its decisions. It owns every resource,
  including the grid port: it is the only writer to the grid, because several cores
  storing add no bandwidth.
- **Kernel threads (cores 1 to 3).** Parallel work for the loop: sweeping completion
  words and counters, draining and answering rings, preparing records, decoding
  micro-ops. They pass work through single-producer, single-consumer queues in the
  cluster's coherent memory.
- **Services.**
  - Grid port: panel windows, and optional per-core windows (183 usable windows per
    cluster).
  - Completion table: for a static program, a team is done when all its members have
    reported; for a dynamic one, when the sent and consumed counts are equal on two
    successive sweeps.
  - Rings, timers and deadlines.
  - Memory: fast local memory for hot state, the cluster's DRAM bank for graphs and
    command streams.
- **Engines.** They are programmed by data the host uploads; the image stays fixed.

| Engine | Uploaded data | What it does |
|---|---|---|
| Micro-op interpreter | a command stream: issue, wait, loop, set, branch on a pushed value | runs the stream; it is also the job-list player |
| Job queue | a stream of jobs and a sizing rule for teams and pools | issues each job to a team sized by the rule |
| Graph executor | task records: record node, CPU-task node, join; predecessor counts; bindings patched at submit | runs a node when its predecessors are done; issues within each unit's credit; supports deadlines and abort; keeps a graph so it can run again with the counts reset |

A graph node has up to four predecessors. The first version of the graph executor runs
a whole graph inside this kernel; a graph is not spread over clusters.

- **CPU tasks** are functions registered in the image: sampling, choosing a variant,
  picking a core count. A graph can run one between two records.
- **Command sources.** The host ring, the rings from clusters 1 to 3, and the engines
  themselves. Host-fed and X280-decided execution are the same kernel with different
  sources, and both are first-class.

## Rules

- One writer per word. There are no atomics between clusters, and none from the grid
  into the X280.
- The X280 never reads the grid. Everything is pushed to it.
- Every loop polls the stop word and has a deadline.
- The image follows the X280 runtime's contract: no `.bss`, floating point and vector
  enabled first, machine mode.

## Work clusters (X280 clusters 1 to 3)

- The X280 runtime of tt-metal gives one image per cluster, so each cluster runs one image.
- A user's X280 thread is a C program linked against `kmx.h`: `main(hartid)`,
  arguments from a block the host writes, a ring to the dispatch kernel (send a
  command, receive a completion), a stop word and a result.
- Inside a cluster the four cores share memory. Each cluster reaches one DRAM bank
  through its cache (cluster 1 bank 6, clusters 2 and 3 bank 7); between clusters,
  data is pushed through NoC windows, and a wake and its answer take 0.66 µs.
- A user who writes only Tensix kernels needs no X280 toolchain: the dispatch image is
  meant to ship prebuilt with a release.
