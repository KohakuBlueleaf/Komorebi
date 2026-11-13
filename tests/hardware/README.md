# Card tests

The tests here need a Blackhole card and run only with `KMR_CARD=1`. Each file is a
small set of known behaviours on the card; most also need the tt-metal tree with its
X280 runtime (see [install](../../docs/install.md)).

## One file per pytest process

Run every file in a pytest process of its own:

    KMR_CARD=1 python -m pytest tests/hardware/test_host.py
    python tests/hardware/run_cards.py                    # every file, one by one
    python tests/hardware/run_cards.py test_thread.py --timeout 300

Why: a process that opens the card keeps the chip's lock until it exits, even after its
session closed the device (the driver takes the lock at the first open and gives it
back only at the process's end). A file that opens the card in the pytest process
(`CARD_IN_PROCESS = True`: `test_host.py`, `test_engines.py`, `test_mapcollect_fed.py`)
followed in the same process by a file that runs a card script as a child would leave
that child waiting for the card. `Session.open` now refuses at once in that case
(`km.CardBusy`, naming the holder), and `conftest.py` refuses such a mixed run before
any card access; one file per process avoids both.

`run_cards.py` runs the files in turn, each with a time bound (SIGTERM, then SIGKILL
30 s later). `--wrap` puts a command in front of each pytest, with `{name}` replaced by
the file's stem, so that each file can run under its own watcher job. The time bound's
SIGTERM and SIGKILL then reach the wrapper only: a wrapper must pass SIGTERM on to its
command, or bound the command itself. A watcher that starts its job in a session of its
own and has no SIGTERM handler ends at the SIGTERM and leaves pytest running; its own
time limit is then the only one that reaches the test.

## Child scripts

A test that runs a script as a child uses `child.run_json(script, *args, timeout=)`:
it returns the script's last JSON line and fails the test on a non-zero exit. A child
still running at the timeout gets SIGTERM first, so that its session closes the X280
runtime (a process killed with the runtime open leaves the card needing a reset), and
SIGKILL only after 30 s.
