"""komorebi.tools.kernel_sizes opens the stand-in card, never a real one (no card)."""

import os
import sys

from komorebi.device import mock
from komorebi.session import process
from komorebi.session import session as km_session
from komorebi.tools import kernel_sizes


def test_the_tool_s_script_runs_on_the_stand_in(tmp_path, monkeypatch):
    monkeypatch.delenv(mock.MOCK, raising=False)
    monkeypatch.setenv(mock.VISIBLE, "0,1")
    monkeypatch.setattr(process, "chip_holder", process.chip_holder)
    monkeypatch.setattr(km_session.Session, "load", km_session.Session.load)
    script = tmp_path / "seen.py"
    script.write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(tmp_path / 'seen')!r}).write_text(\n"
        f"    os.environ[{mock.MOCK!r}] + ' ' + os.environ[{mock.VISIBLE!r}])\n"
    )
    monkeypatch.setattr(sys, "argv", ["kernel_sizes", str(script)])
    kernel_sizes.main()
    assert (tmp_path / "seen").read_text() == f"{mock.descriptor([0])} 0"


def test_a_mock_named_is_kept(monkeypatch):
    monkeypatch.setenv(mock.MOCK, "/a/mock.yaml")
    monkeypatch.setenv(mock.VISIBLE, "1")
    kernel_sizes.stand_in()
    assert (os.environ[mock.MOCK], os.environ[mock.VISIBLE]) == (
        "/a/mock.yaml",
        "1",
    )
