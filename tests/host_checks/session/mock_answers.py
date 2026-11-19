"""The mock run refuses what the card's run refuses (a wait of a handle never submitted
or returned, result() and trace() before the stop), keeps the queue's counts and the
finish's keys, refuses a closed session's read and bad depths by the card's words, and
the native module's build prints one line. No card, no build (TT_METAL_HOME set)."""

import contextlib
import io
import os
import sys
import tempfile
import time
import types
from pathlib import Path

from komorebi import abi
from komorebi.device import mock, native
from komorebi.device import tensor as tensor_module
from komorebi.errors import KomorebiError
from komorebi.session import sizes

STARTED = time.monotonic()
checks = {}


def refused(call, kind, words):
    """Whether `call()` raises `kind` with `words` in its message."""
    try:
        call()
    except kind as error:
        return words in str(error)
    return False


run = mock.Run(counted=False)
first = run.submit("h", (0, 1))
checks["wait"] = run.wait(first).kind == "done"
checks["wait twice"] = refused(lambda: run.wait(first), ValueError, "was returned")
checks["never submitted"] = refused(lambda: run.wait(99), ValueError, "never submitted")
second, third = run.submit("h", (0, 1)), run.submit("h", (0, 1))
checks["wait_any"] = run.wait_any([third, second]).handle == second
checks["wait_any again"] = refused(
    lambda: run.wait_any([second]), ValueError, "returned or forgotten"
)
checks["result before stop"] = refused(run.result, KomorebiError, "run.stop()")
checks["trace before stop"] = refused(run.trace, KomorebiError, "run.stop()")
run._queue_command(7, (0, 3), [1])
run._counts[-1] = 0  # what a session's own command does (dispatcher_watch's stall)
checks["counts"] = list(run._counts) == [0]
many = run.submit_many("h", 0, 4, [[1, 2]] * 5)  # one group, a command a row
checks["submit_many rows"] = len(many) == 5 and len(run.wait(list(many))) == 5
block = run.stop()
checks["result after stop"] = block["state_name"] == "done"
words = abi.headers()["KMD_H_WORDS"]
checks["result's finish keys"] = block["cleaned"] and len(block["x280_cores"]) == words

checks["depths good"] = sizes.check_depths(16, 64, 128) is None
for bad in ((0, 64, 128), (65, 64, 128), (16, 48, 128), (16, 64, 256)):
    checks[f"depths {bad}"] = refused(
        lambda bad=bad: sizes.check_depths(*bad), ValueError, "power of two"
    )

closed = types.SimpleNamespace(device=None)
tensor = types.SimpleNamespace(loaded=types.MethodType(lambda self: False, closed))
checks["closed read"] = refused(
    lambda: tensor_module.refuse_closed(tensor), RuntimeError, "the device is closed"
)

built = []
with tempfile.TemporaryDirectory() as folder:
    native.cache_directory = lambda: Path(folder)
    native.build_whole = lambda out, build: built.append(out)
    said = io.StringIO()
    with contextlib.redirect_stderr(said):
        native.module_path(os.environ["TT_METAL_HOME"])
    checks["build line"] = (
        len(built) == 1 and said.getvalue().count("building the native host") == 1
    )

failed = [name for name, ok in checks.items() if not ok]
print(
    f"{len(checks) - len(failed)} of {len(checks)} ok, failed {failed}, "
    f"{time.monotonic() - STARTED:.1f} s"
)
sys.exit(1 if failed else 0)
