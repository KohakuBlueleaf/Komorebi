# komorebi_native

The native layer: everything that runs on the card or builds against tt-metal. Written
so far: the Tensix worker headers, the X280 dispatcher, the library of X280 user
threads, the native Python module and, in `x280/engines/`, the graph executor in plain
C (also built for the host). `include/komorebi/x280/`, `include/komorebi/host/`,
and `kernels/` are empty.

| Directory | What it holds |
|---|---|
| `include/komorebi/device/` | Headers for kernels that run on the Tensix cores: the persistent reader, compute and writer (`km_worker.hpp`) at the top; `core/` (the 64-byte command `km_command.hpp`, state words and timed waits `km_core.hpp`, group completion `km_completion.hpp`, the per-core trace `km_trace.hpp`, islands, groups, the program's parts), `tensors/` (where a tensor's pages live and how a kernel reads them), `movement/` (movers, broadcasts, ports, trees, the chain), `links/` (pages and links across islands), `messages/` (dynamic runs' messages and deliveries) and `overlays/` (loaded code). Static breakpoints (`emit`, `wait`, `reduce`) and the inbox ring of dynamic breakpoints are not written yet. |
| `include/komorebi/x280/` | Reserved; the header a user's X280 thread includes is `x280/kmx/kmx.h`. |
| `include/komorebi/host/` | The C++ host API, for programs that do not use Python. |
| `kernels/` | The runtime's own kernels, shipped as source and compiled by tt-metal when a program runs: `program/` (the reader, compute and writer that run a program's parts, and the service overlay's reader), `link/` (the link cores' services), `messages/` (the movers and compute of a dynamic run's messages); `host_transfer.cpp`, a transfer without a run, at the top. |
| `x280/kernel/` | The dispatcher for X280 cluster 0: the main loop, the services (grid port, completion table, rings, timers, memory) and the code that runs on the cluster's other cores. |
| `x280/engines/` | The engines the dispatcher runs from uploaded data: the micro-op interpreter, the job queue and the graph executor. |
| `x280/kmx/` | The library an X280 thread on a work cluster is linked with: `kmx.h` (its entry point, regions, cache flush / invalidate with the grid, four-core calls, result and progress, the command ring to the dispatcher), the start-up, `kmx_ring.h` (the thread's side of the ring, the host ring's protocol, with its placement as data) and `kmx_layout.h`, the layout of its argument, ring and result blocks shared with the host. |
| `csrc/host/` | The native Python module: the runtime (open the card, load X280 images, start, stop) and plans (upload a stream, a job list or a graph, run, read back). |
| `cmake/` | `km_host.cmake`: the native Python module built inside a configured tt-metal tree with that tree's targets (how `packaging/build_wheel.py --host-module` prebuilds it). |

C and C++ sources are formatted with `python -m komorebi.tools.cxx_format`.
