"""tt-metal issue 64717: nothing the framework adds to the environment is added after
the native module loads (Tracy's listener reads it then), and the load returns once
Tracy listens (the native load needs a tt-metal tree: the tree-tests lane)."""

import os
import subprocess
import sys

import pytest

from komorebi.device import mock
from komorebi.session import session
from komorebi.session.parts import open as open_part

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Loaded(Exception):
    """Stops the open at the native load."""


def test_an_open_sets_its_names_before_the_native_load(monkeypatch):
    monkeypatch.setenv(mock.ACTIVE, "1")
    monkeypatch.delenv(mock.MOCK, raising=False)
    monkeypatch.delenv("TT_METAL_SLOW_DISPATCH_MODE", raising=False)
    monkeypatch.delenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", raising=False)
    seen = {}

    def load(tree=None):
        seen.update(os.environ)
        raise Loaded

    monkeypatch.setattr(open_part.native, "load_module", load)
    monkeypatch.setattr(open_part.process, "chip_holder", lambda card: None)
    monkeypatch.setattr(open_part.process, "refuse_reopen", lambda kind, cards: None)
    made = session.Session(card=None, log=lambda line: None)
    with pytest.raises(Loaded):
        made.open()
    for name in (mock.MOCK, "TT_METAL_SLOW_DISPATCH_MODE"):
        assert seen.get(name), name
    assert "TT_METAL_OPERATION_TIMEOUT_SECONDS" in seen
    names = set(os.environ)
    with mock.cluster_visible():  # the open's own step after the load adds nothing
        assert set(os.environ) <= names | {mock.VISIBLE}


@pytest.mark.tree
def test_the_native_load_returns_once_tracy_listens(tmp_path):
    code = (
        "from komorebi.device import native\n"
        "before = native.listening_sockets()\n"
        "native.load_module()\n"
        "print(native.tracy_mapped(), native.LISTEN_WAITED.get('listening'),\n"
        "      len(native.listening_sockets() - before) > 0)\n"
    )
    paths = [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))
    env.pop("TRACY_PORT", None)
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    mapped, listening, new = done.stdout.split()[-3:]
    if mapped != "True":
        pytest.skip("the runtime's Tracy is not loaded with the native module")
    assert listening == "True" and new == "True"
