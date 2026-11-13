---
title: "Every primitive"
part: 6
summary: "a look-up list of every call this tutorial used, in the three places a program runs, each with the lesson that explains it"
---

# Every primitive

This page lists, in the order of the three places a Komorebi program runs in, every call that
the tutorial's programs use, with one line of what it does and the lesson that explains it.
It is for looking up, not for reading in order. It relies on all of Parts 1 to 5.

A program is Python on the host, kernels on the Tensix cores, and, when it has one, a thread on the X280 cores. Commands go to the Tensix cores from Python and from a thread. The three tables below follow that order.

## Python, on the host

| Call | What it does | Lesson |
|---|---|---|
| `km.Program(name, kernels=folder)` | describes what the cores will run | [[hello-core]] |
| `km.Session()` | the open card | [[hello-core]] |
| `session.load(program)` | builds, loads and starts the kernels; returns the run | [[hello-core]] |
| `run.submit(kind, group=(first, count))` | sends a command, returns a handle | [[hello-core]] |
| `run.wait(handle, timeout=s)` | waits for the event of a command; a list of handles waits for all | [[hello-core]], [[many-commands]] |
| `run.stop()` | ends the kernels | [[hello-core]] |
| `event.kind`, `event.microseconds` | how a command ended, and the card's time for it | [[commands]] |
| `args=[...]` | up to ten numbers in a command | [[arguments]] |
| `result=True`, `event.result` | asks for a result word, and reads it | [[a-result]] |
| `handlers=header` | names for the handler numbers | [[handlers]] |
| `session.load(program, deadline=s)` | the card's time limit for a command | [[when-it-goes-wrong]] |
| `km.WaitTimeout`, `km.CommandTimeout` | the host gave up, the card gave up | [[when-it-goes-wrong]] |
| `session.tensor(array, dtype, "tile")` | puts an array in DRAM as tiles | [[tiles-and-tensors]] |
| `tensor.read()` | reads a tensor back as an array, after `stop` | [[tiles-and-tensors]] |
| `session.read(tensor)` | the raw pages of a tensor | [[writing-to-dram]] |
| `tensor.write(array)` | writes a tensor, before `load` | [[a-whole-map]] |
| `tensor.accessor_args()`, `compile_args=` | lets a kernel address a tensor | [[reading-from-dram]] |
| `km.Buffer(index, format, page_bytes, pages)` | a circular buffer in the L1 of every core | [[circular-buffers]] |
| `compute={...}` | arithmetic settings of the compute kernel | [[the-matrix-unit]] |
| `exclusive=True` | a command waits for earlier ones that share cores | [[port]] |
| `Program(...).dynamic(...)` | the handlers that receive messages | [[routed-message]], [[message-handlers]], [[full-buffers]] |
| `Graph`, `run_hostfed` | a chain of commands that the card runs by itself | [[streams-and-graphs]] |
| `program.thread(...)`, `program.ports(...)` | a thread on an X280 cluster, and its ports | [[hello-thread]], [[core-asks-x280]] |
| `run.thread_result(timeout)`, `run.thread.read(name)` | the thread's result, and a region after the run | [[hello-thread]], [[sharing-memory]] |
| `program.overlay(name, folder)`, `program.slots(count=n)` | an overlay's parts, and the slots that hold them | [[overlays]] |

## Kernels, on the Tensix cores

| Call | What it does | Lesson |
|---|---|---|
| `km::Reader`, `km::Compute`, `km::Writer`, `.run(lambda)` | the loop of a kernel, one pass for each command | [[hello-core]] |
| `km::handler(command)`, `command.a0` to `a9` | which handler, and the arguments | [[handlers]], [[arguments]] |
| `writer.result(value)` | the result word | [[a-result]] |
| `km::first`, `km::count`, `me()` | where a core is in the command's group | [[core-group]] |
| `km::partition(n, rank, count)` | a core's share of `n` items | [[partitions]] |
| `km::group_of(command, me)` | `.rank` and `.size` of a core in the group | [[the-leader]] |
| `reader.reserve`, `writer.wait_front`, `cb_push_back`, `cb_pop_front` | pass tiles through a circular buffer | [[circular-buffers]] |
| `noc_async_read_page`, `noc_async_write_page` and their barriers | copy a tile from or to DRAM | [[reading-from-dram]], [[writing-to-dram]] |
| `add_tiles`, `tile_regs_acquire` to `tile_regs_release`, `pack_tile` | add tiles in the matrix unit's registers | [[computing-on-a-tile]] |
| `matmul_init`, `matmul_tiles` | multiply tiles | [[the-matrix-unit]] |
| `km::l1(address)` | a core's L1 as a tensor Python can read | [[why-cores-talk]] |
| `km::Arrival`, `km::emit`, `km::arrive` | send a tile to another core, and wait for it | [[port]] |
| `km::reduce`, `reduce_arrive`, `reduce_emit`, `reduce_result` | add the tiles of many cores into one | [[reduce]] |
| `km::Broadcast` | one core's tile, a word, or both, to many cores | [[broadcast]], [[casting-a-word]] |
| `km::Messages`: `send`, `next`, `forward`, `consume`, `done`, `quota` | messages to any core | [[routed-message]], [[forward-and-consume]], [[full-buffers]] |
| `km::Messages`: `handler`, `forward` and `send` with a handler | a message names its handler | [[message-handlers]] |
| `km::X280Port`: `call`, `tell` | ask and tell an X280 core | [[core-asks-x280]] |
| `km::Overlay`: `load`, `call`, `held`; `km_overlay(kernel, argument)` | load an overlay into a slot and call it; a part's function | [[overlays]] |

## Threads, on the X280 cores

| Call | What it does | Lesson |
|---|---|---|
| `kmx_main(core)`, `kmx_args_get()`, `kmx_result(code, words, count)` | the entry, the arguments, the result | [[hello-thread]] |
| `kmx_submit`, `kmx_wait`, `kmx_time` | send a command to the grid and wait for its event | [[thread-submits]] |
| `kmx_parallel(function, argument)` | a function on the cluster's four cores | [[four-cores]] |
| `kmx_region`, `kmx_invalidate`, `kmx_flush` | a region as memory, and keeping the cache in step | [[sharing-memory]] |
| `kmx_cluster`, `kmx_lead`, `kmx_serve`, `kmx_parallel_all` | a function on all twelve cores | [[twelve-cores]] |
| `kmx_port_owner`, `kmx_port_asked`, `kmx_port_answer`, `kmx_port_told`, `kmx_port_open`, `kmx_port_run` | serve a Tensix core's asks and tells | [[core-asks-x280]] |

## What next

The reference pages of this site list every call with all its options, and the examples
combine them into larger programs. [[a-whole-program]] and [[your-own-kernel]] are the way
in: start from a program that works, and change one thing.

Previous: [[overlays]].
