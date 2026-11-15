# Card tests

The tests here need a Blackhole card and run only with `KMR_CARD=1`. Each file is a
small set of known behaviours on the card; most also need the tt-metal tree with its
X280 runtime (see [install](../../docs/install.md)).

## One process, one device open

    KMR_CARD=1 python -m pytest tests/hardware                       # the folder
    KMR_CARD=1 python -m pytest tests/hardware/test_thread.py        # one file

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

A card test never starts a child process that opens the card: the chip stays locked by
the process that opened it until that process exits, so a child would wait for it.
`Session.open` raises `km.CardBusy` naming the holder when another live process has the
card.

Kernels are compiled once by the runtime's cache; the first run in a new checkout or
after a kernel change pays the build.

## Expected time

Seconds of each file in one run of the folder in a new checkout (a cold kernel cache),
read from the timestamps of its log, to a few seconds; the first file's include the
device open (about 13 s). Two files with no line of their own in the log are counted
together. The kernel and overlay builds of a cold cache are about a third of the total
(68 s and 142 s of the first 600 s); a warm cache does not pay them.

| File | s | | File | s |
| --- | ---: | --- | --- | ---: |
| `test_attention_host.py` (the open) | 24 | | `test_movement.py` | 8 |
| `test_bank_fetch.py` | 6 | | `test_overlays.py` | 15 |
| `test_broadcast_host.py` | 17 | | `test_page_calls.py` | 227 |
| `test_dense_host.py` | 30 | | `test_programs.py` | 40 |
| `test_engines.py` | 7 | | `test_raster_card.py` | 27 |
| `test_forward_host.py` | 10 | | `test_raytrace_card.py` | 20 |
| `test_give_up.py` | 11 | | `test_swiglu_host.py` | 12 |
| `test_graph_host.py` | 14 | | `test_thread.py` | 19 |
| `test_host.py` | 26 | | `test_thread_commands.py` | 10 |
| `test_host_transfer.py` | 1 | | `test_thread_lifecycle.py` | 10 |
| `test_kernel_buffer.py` | 2 | | `test_thread_split.py` | 31 |
| `test_leader.py` | 9 | | `test_uop_stream.py` | 17 |
| `test_leader_overlay.py`, `test_mapcollect_fed.py` | 27 | | `test_x280_ports.py` | 10 |
| `test_message_handlers.py` | 11 | | **the folder** | **about 640** |

The folder no longer fits one ten-minute job: run it in parts (a list of files a
process), each part sized from this table.

A file run alone also pays the open. `test_raytrace_card.py` asks for a smaller kernel
buffer than the others and `test_kernel_buffer.py` asks for 128 KiB, so the device is
closed and opened again for each, and again for the next file after it; the 128 KiB case
with its run took 1.2 s, so a reopen is cheap after the first open.
