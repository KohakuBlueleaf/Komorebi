"""A tensor read while its session has a run loaded is refused at once, naming what
reads it live (session.read) and run.stop(); in a prebuild's mock run too, so a
script that reads so fails off the card; read again once the run is stopped (no card).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi.device import mock
from komorebi.device.tensor import Tensor
from komorebi.errors import KomorebiError
from komorebi.session.session import Session


class Buffer:
    page_size, num_pages, size = 4096, 1, 4096

    def read(self):
        return np.zeros(self.size, np.uint8)


def tensor(loaded, shard_grid=None):
    made = Tensor(Buffer(), (1, 1024), "float32", "row", shard_grid)
    made.loaded = loaded
    return made


def test_a_read_while_a_run_is_loaded_is_refused_by_name(monkeypatch):
    with pytest.raises(KomorebiError, match=r"run.stop\(\), or session.read\(tensor\)"):
        tensor(lambda: True).read()
    monkeypatch.setenv(mock.ACTIVE, "1")  # a prebuild's mock run meets it too
    with pytest.raises(KomorebiError, match="by DRAM banks or the host channel"):
        tensor(lambda: True).read()
    with pytest.raises(KomorebiError, match="L1 tensor live only through the host"):
        tensor(lambda: True, (2, 1)).read_bytes()


def test_a_read_after_the_stop_or_with_no_run_reads():
    assert tensor(lambda: False).read().shape == (1, 1024)
    assert tensor(None).read().shape == (1, 1024)
    run = SimpleNamespace(stopped=False)
    session = SimpleNamespace(run=run)
    assert Session._run_loaded(session)
    run.stopped = True
    assert not Session._run_loaded(session)
    assert not Session._run_loaded(SimpleNamespace(run=None))
