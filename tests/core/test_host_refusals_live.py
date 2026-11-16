"""A tensor written while its session has a run loaded is refused at once, naming
session.write; a prebuild's script stopped by an error exits non-zero with the line
it stopped at, its own exit and its end still 0 (no card)."""

import sys

import numpy as np
import pytest

from komorebi import prebuild
from komorebi.errors import KomorebiError
from komorebi.tensor import Tensor


class Buffer:
    page_size, num_pages, size = 4096, 1, 4096

    def __init__(self):
        self.written = None

    def write(self, raw):
        self.written = raw


def test_a_write_while_a_run_is_loaded_is_refused_by_name():
    tensor = Tensor(Buffer(), (1, 1024), "float32", "row")
    tensor.loaded = lambda: True
    with pytest.raises(KomorebiError, match=r"run.stop\(\), or live with session.wri"):
        tensor.write(np.zeros((1, 1024), np.float32))
    assert tensor.buffer.written is None
    tensor.loaded = lambda: False
    tensor.write(np.zeros((1, 1024), np.float32))
    assert tensor.buffer.written is not None


@pytest.mark.parametrize(
    ("body", "code", "said"),
    [
        ("x = 1\nraise RuntimeError('zeros')\n", prebuild.STOPPED, "stopped at line 2"),
        ("import sys\nsys.exit(1)\n", 0, "the script exited 1"),
        ("x = 1\n", 0, "0 load(s) built"),
    ],
)
def test_a_prebuild_s_script_stopped_by_an_error_exits_non_zero(
    monkeypatch, tmp_path, capsys, body, code, said
):
    script = tmp_path / "script.py"
    script.write_text(body)
    monkeypatch.setenv(prebuild.ACTIVE, "1")
    monkeypatch.delenv(prebuild.MEMORY, raising=False)
    monkeypatch.setattr(sys, "argv", ["child", str(script)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(prebuild, "LOADS", [0])  # no other test's loads
    assert prebuild.main() == code
    assert said in capsys.readouterr().err
