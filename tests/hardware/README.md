# Card tests

The tests here need a Blackhole card and run only with `KMR_CARD=1`. They are the known
behaviours on the card that `python -m komorebi.tools.card_check` does not check: an
example's exactness is card_check's (its list is `src/komorebi/tools/card_checks.txt`),
and these files keep what no example shows. Most also need the tt-metal tree with its
X280 runtime (see [install](../../docs/install.md)). By folder: `engines/` (the graph
engines), `host/` (the session, transfers, the kernel buffer, a deadline's give-up),
`feeder/` (the kernels fed without the X280), `programs/` (a class product's operand
formats) and `thread/` (X280 threads: commands, lifecycle, a bank fetch). The case
scripts `thread/clusters/shared_case.py` and `thread/commands/open_points_case.py` are
on card_check's list.

## One process, one device open

    KMR_CARD=1 python -m pytest tests/hardware                       # the folder
    KMR_CARD=1 python -m pytest tests/hardware/host/test_host.py     # one file

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

## In the regression

`packaging/ci/regression.sh` runs this folder on card 1 while card_check's card 0
process runs, then card_check's cards 0,1 process: `KMR_CARD=1 TT_VISIBLE_DEVICES=1
python -m pytest tests/hardware`.

Off the card, from `experiments/checks/stream_checks` and `experiments/checks/graph_checks`
with the package on the path: `python stream_reference.py --check` (0 of 91 streams
differ), `python mutants.py` (every mutant killed) and `python graph_reference.py
--check` (no difference).
