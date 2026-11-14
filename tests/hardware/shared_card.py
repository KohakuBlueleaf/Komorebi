"""One device open for the whole card test run (opening the card takes about 13 s).
`host()` is the open session; `run_json` and `run_text` run a script in this process
with `km.Session` replaced by a session on that device. A session with another kernel
buffer closes the device and opens another."""

import contextlib
import gc
import io
import json
import runpy
import sys
from pathlib import Path

import pytest

import komorebi as km

_OPEN = {}  # "session": the Session that owns the device; "key": how it was opened
_SESSION_CLASS = km.Session  # the class the scripts get when no card test runs
CHANNEL = km.session.channel_module.REGION_DEFAULT


def host(kernel_bytes=None, channel=CHANNEL):
    """The open session (opened here the first time), on a device opened with these
    options; a session open with other options is closed first."""
    key = (kernel_bytes, channel)
    if _OPEN.get("key") != key:
        close_host()
        session = _SESSION_CLASS(
            kernel_bytes=kernel_bytes,
            channel=channel,
            log=lambda m: print(m, flush=True),
        )
        session.open()
        _OPEN.update(session=session, key=key)
    return _OPEN["session"]


def release():
    """Aborts a run a test left loaded, so that the next one can load."""
    session = _OPEN.get("session")
    if session is not None and session.run is not None and not session.run.stopped:
        session.run.abort()
    gc.collect()


def close_host():
    """Closes the shared device, if one is open."""
    session = _OPEN.pop("session", None)
    _OPEN.pop("key", None)
    if session is not None:
        session.close()


class Shared(_SESSION_CLASS):
    """A session on the shared device: it opens and closes nothing but its own run."""

    def __init__(self, *args, channel=CHANNEL, **kwargs):
        super().__init__(*args, channel=channel, **kwargs)
        self.channel_option = channel

    def open(self):
        shared = host(self.kernel_bytes, self.channel_option)
        self.device, self._module, self._own_device = (
            shared.device,
            shared._module,
            False,
        )
        self.channel, self._region = shared.channel, shared._region
        self._channel_said = shared._channel_said


class Tee(io.StringIO):
    """Collects what is written and passes it on, so the watched log keeps growing."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def write(self, text):
        self.target.write(text)
        return super().write(text)

    def flush(self):
        self.target.flush()


def run_text(script, *args):
    """Runs `script args...` in this process as `python script args...` would, on the
    shared device; returns its stdout. A non-zero exit fails the test. Nothing here
    bounds the run: the sessions' own waits and the card's watcher do."""
    script = Path(script)
    before = set(sys.modules)
    saved = sys.argv, list(sys.path), km.Session
    sys.argv = [str(script), *map(str, args)]
    sys.path.insert(0, str(script.parent))
    km.Session = Shared
    out = Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            pytest.fail(f"{script.name} exited {error.code}\n{out.getvalue()[-2000:]}")
    finally:
        sys.argv, sys.path[:], km.Session = saved[0], saved[1], saved[2]
        for name in set(sys.modules) - before:
            file = getattr(sys.modules[name], "__file__", None) or ""
            if Path(file).parent == script.parent:
                del sys.modules[name]
        release()
    return out.getvalue()


def run_json(script, *args):
    """run_text, then the last stdout line that starts with "{", parsed."""
    lines = [
        line for line in run_text(script, *args).splitlines() if line.startswith("{")
    ]
    return json.loads(lines[-1])
