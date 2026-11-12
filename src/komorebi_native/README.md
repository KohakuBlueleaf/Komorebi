# komorebi_native

The native layer: everything that runs on the card or builds against tt-metal. The
directories exist; their contents are being written and are not yet complete.

| Directory | What it holds |
|---|---|
| `include/komorebi/device/` | Headers for kernels that run on the Tensix cores: the 64-byte record, the resident worker, teams, static breakpoints (`emit`, `wait`, `reduce`), the inbox ring of dynamic breakpoints, and the per-core trace. |
| `include/komorebi/x280/` | The header a user's X280 thread includes (`kmx.h`): its entry point, arguments, and the ring to the dispatch kernel. |
| `include/komorebi/host/` | The C++ host API, for programs that do not use Python. |
| `kernels/` | The generic reader and writer kernels that own inbox and outbox traffic, shipped as source and compiled by tt-metal when a program runs. |
| `x280/kernel/` | The dispatch kernel for X280 cluster 0: the main loop, the services (grid port, completion table, rings, timers, memory) and the code that runs on the cluster's other cores. |
| `x280/engines/` | The engines the dispatch kernel runs from uploaded data: the micro-op interpreter, the job queue and the graph executor. The runtime for user images on the work clusters will sit in `x280/` beside them. |
| `csrc/host/` | The native Python module: the runtime (open the card, load X280 images, start, stop) and plans (upload a stream, a job list or a graph, run, read back). |
| `cmake/` | CMake helpers to build a C++ host and an X280 image against a tt-metal tree. |

C and C++ sources are formatted with `python -m komorebi.tools.cxx_format`.
