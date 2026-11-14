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

Seconds of the tests themselves, after the one device open (about 13 s, once), with the
kernel cache warm. Read from one run of each file before the folder was shared, minus
the kernel builds that run paid; the cases removed or shortened since are taken off.
They are estimates until the folder has run once as it is now.

| File | s | | File | s |
| --- | ---: | --- | --- | ---: |
| `test_host.py` | 19 | | `test_movement.py` | 4 |
| `test_engines.py` | 2 | | `test_overlays.py` | 5 |
| `test_mapcollect_fed.py` | 11 | | `test_programs.py` | 10 |
| `test_host_transfer.py` | 2 | | `test_swiglu_host.py` | 4 |
| `test_raster_card.py` | 10 | | `test_thread_commands.py` | 6 |
| `test_attention_host.py` | 3 | | `test_thread.py` | 10 |
| `test_bank_fetch.py` | 3 | | `test_thread_lifecycle.py` | 11 |
| `test_broadcast_host.py` | 10 | | `test_x280_ports.py` | 6 |
| `test_dense_host.py` | 11 | | `test_thread_split.py` | 6 |
| `test_forward_host.py` | 5 | | `test_uop_stream.py` | 3 |
| `test_kernel_buffer.py` | 2 | | `test_raytrace_card.py` | 6 |
| `test_give_up.py` | 7 | | | |
| `test_graph_host.py` | 11 | | | |
| `test_leader.py` | 8 | | | |
| `test_leader_overlay.py` | 3 | | | |
| `test_message_handlers.py` | 6 | | **the folder** | **about 184 + 13** |

A file run alone also pays the open. `test_raytrace_card.py` asks for a smaller kernel
buffer than the others and `test_kernel_buffer.py` asks for 128 KiB, so the device is
closed and opened again for each, and again for the next file after it; the 128 KiB case
with its run took 1.2 s, so a reopen is cheap after the first open.
