# Card tests

The tests here need a Blackhole card and run only with `KMR_CARD=1`. Each file is a
small set of known behaviours on the card; most also need the tt-metal tree with its
X280 runtime (see [install](../../docs/install.md)). The files are in folders by what
they test: `engines/` (streams and graphs), `host/` (the session, transfers, give-up),
`feeder/` (the kernels fed without the X280), `thread/` (X280 threads and ports),
`movement/` (movers, broadcasts, page calls), `programs/` (programs and overlays),
`messages/` (a dynamic run's messages), `workloads/` (the projects' calls) and `mesh/`
(cards 0 and 1 as one session's islands).

## One process, one device open

    KMR_CARD=1 python -m pytest tests/hardware                       # the folder
    KMR_CARD=1 python -m pytest tests/hardware/thread/clusters/test_thread_card.py  # one file

The folder runs in one pytest process on one open device. Opening the card takes about
13 s (the driver's own start-up), so every test shares the open:

- A test file that drives a session takes the `session` fixture (`conftest.py`): the
  one open `km.Session`. A run a file leaves loaded is aborted when the file ends.
- A test that runs an example or a case script calls `shared_card.run_json(script,
  *args)` (or `run_text`). The script runs in this process as `python script args...`
  would, with `km.Session` replaced for its duration by a session on the shared device:
  its `with km.Session(...)` loads, runs and stops its own run and leaves the device
  open. Its stdout is passed on to the log and returned.
- A session whose kernel buffer differs (`km.Session(kernel_bytes=...)`) closes the
  shared device and opens another; keep such tests together at the end of the folder.
  A script whose session names no buffer and whose kernels do not fit the open one
  (`km.KernelSizeError`, nothing loaded) is run once more on a device opened at the
  size they need, as a session that opens its own card follows its kernels.

A card test never starts a child process that opens the card: the chip stays locked by
the process that opened it until that process exits, so a child would wait for it.
`Session.open` raises `km.CardBusy` naming the holder when another live process has the
card.

Kernels are compiled once by the runtime's cache; the first run in a new checkout or
after a kernel change pays the build.

## Two cards: a process of their own

    KMR_CARD=1 KMR_MESH=1 python -m pytest tests/hardware/mesh/test_two_cards.py

A process that opened a card alone may not open cards as a mesh, nor the reverse
([hazard](../../docs/reference/hazards.md#a-card-opened-alone-in-a-process-that-opened-a-mesh)),
so the `mesh` tests (marked `pytest.mark.mesh`) never share the folder's process: they
run only with `KMR_MESH=1` and in a run with no other test, and are skipped by name when
cards 0 and 1 are not both visible. A split card opens its one card alone and runs
with the folder (`host/test_split_card.py`).

2026-11-18, warm kernel cache, each file alone in its own process: `mesh/test_two_cards.py`
4.1 s with the open of cards 0 and 1; `host/test_split_card.py` 11.9 s,
`programs/test_class_loop_card.py` 13.1 s and `programs/test_chain_card.py` 11.7 s,
each with the open of card 0.

## Expected time

Seconds of each file in one run of the folder in a new checkout (a cold kernel cache),
read from the timestamps of its log, to a few seconds; the first file's include the
device open (about 13 s). Two files with no line of their own in the log are counted
together. The kernel and overlay builds of a cold cache are about a third of the total
(68 s and 142 s of the first 600 s); a warm cache does not pay them.

| File | s | | File | s |
| --- | ---: | --- | --- | ---: |
| `workloads/test_attention_host.py` (the open) | 24 | | `movement/test_movement_card.py` | 8 |
| `thread/test_bank_fetch.py` | 6 | | `programs/test_overlays.py` | 15 |
| `movement/test_broadcast_host.py` | 17 | | `movement/test_page_calls.py` | 227 |
| `workloads/test_dense_host.py` | 30 | | `programs/test_programs_card.py` | 40 |
| `engines/test_engines_card.py` | 7 | | `workloads/test_raster_card.py` | 27 |
| `messages/test_forward_host.py` | 10 | | `workloads/test_raytrace_card.py` | 20 |
| `host/test_give_up.py` | 11 | | `workloads/test_swiglu_host.py` | 12 |
| `engines/test_graph_host.py` | 14 | | `thread/clusters/test_thread_card.py` | 19 |
| `host/test_host.py` | 26 | | `thread/commands/test_thread_commands.py` | 10 |
| `host/test_host_transfer.py` | 1 | | `thread/lifecycle/test_thread_lifecycle.py` | 10 |
| `host/test_kernel_buffer.py` | 2 | | `thread/commands/test_thread_split_card.py` | 31 |
| `movement/test_leader.py` | 9 | | `engines/test_uop_stream.py` | 17 |
| `programs/test_leader_overlay.py`, `feeder/test_mapcollect_fed.py` | 27 | | `thread/test_x280_ports_card.py` | 10 |
| `messages/test_message_handlers_card.py` | 11 | | **the folder** | **about 640** |

The folder no longer fits one ten-minute job: run it in parts (a list of files a
process), each part sized from this table.

A file run alone also pays the open. `workloads/test_raytrace_card.py`'s kernels need a
larger kernel buffer than the default (about 85 KiB) and `host/test_kernel_buffer.py`
asks for 128 KiB, so the device is closed and opened again for each, and again for the
next file after it;
the 128 KiB case with its run took 1.2 s, so a reopen is cheap after the first open.

## Run once before a release

Every file of the folder on one card, and the two dispatcher scripts that pytest does
not collect: `dispatcher/dispatch_card.py` and `dispatcher/graph_card.py` (build their
test images first: `python build.py` in `dispatcher/`). From the repository root, with
`KMR_CARD=1`, in parts that each fit one job; the seconds are one run on a warm kernel
cache (2026-11-17), the part's tests without the open (about 13 s a process).

| Part | Command | s |
|---|---|---:|
| dispatcher | `python tests/hardware/dispatcher/dispatch_card.py set` (22 cases), then `python tests/hardware/dispatcher/graph_card.py set` (2) | 17 + 12 |
| engines | `python -m pytest tests/hardware/engines` | 28 |
| host | `python -m pytest tests/hardware/host` | 31 |
| messages, feeder | `python -m pytest tests/hardware/messages tests/hardware/feeder` | 23 |
| movement | `python -m pytest tests/hardware/movement --ignore=tests/hardware/movement/test_page_calls.py` | 27 |
| page calls | `python -m pytest tests/hardware/movement/test_page_calls.py`, in parts by test (`-k`): the six example tests about 87 s, the fifteen lessons about 85 s | 171 |
| programs | `python -m pytest tests/hardware/programs` | 72 |
| thread | `python -m pytest tests/hardware/thread` | 60 |
| workloads | `python -m pytest tests/hardware/workloads` | 66 |
| mesh (cards 0 and 1) | `KMR_MESH=1 python -m pytest tests/hardware/mesh/test_two_cards.py`, a process of its own | 4 |

Off the card, from `experiments/checks/stream_checks` and `experiments/checks/graph_checks`
with the package on the path: `python stream_reference.py --check` (0 of 91 streams
differ), `python mutants.py` (every mutant killed) and `python graph_reference.py
--check` (no difference).
