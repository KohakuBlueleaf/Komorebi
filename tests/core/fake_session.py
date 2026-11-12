"""Stand-ins for ttnn, the native module and the session's card backend, so that
Session.load and its failure paths run on the CPU over the fake kernel."""

import importlib
import sys
import types

from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.program import Program

K = abi.headers().constants


class FakeDispatch(FakePort):
    """The native Dispatch over the fake kernel: prepare, start, wait_ready, close."""

    grid_x, grid_y = 12, 10
    lim_base = 0x10000
    boot_error = None  # a KMD_ERR_* the kernel ends with before READY

    def __init__(self, log, device, pll_mhz):
        super().__init__()
        self.log = log
        self.is_open = True
        self.settle_us = 1.0
        self.ready_state = K["KMD_STATE_READY"]
        self.idle = True
        self.fail = {}  # call name -> exception it raises

    def _call(self, name):
        self.log.append(name)
        if name in self.fail:
            raise self.fail[name]

    def prepare(self, *args, **kwargs):
        self._call("prepare")
        return 5

    def start(self, image):
        self._call("start")

    def wait_ready(self, timeout_s):
        self._call("wait_ready")
        if self.boot_error is not None:
            self._finish(self.boot_error)
            return self.state
        return self.ready_state

    def close(self):
        self._call("close")
        self.is_open = False
        return True


class SessionBackend(FakeBackend):
    """The session's backend over a FakeDispatch, logging what it is asked to do."""

    def __init__(self, session, dispatch, mhz, trace):
        super().__init__(dispatch, trace_cap=trace)
        self.session = session
        self.log = dispatch.log
        self.io = []

    def describe(self, *args):
        self.log.append("describe")
        return "descriptor"

    def finish(self, timeout):
        self.log.append("finish")
        try:
            return super().finish(timeout)
        finally:
            self.port.close()

    def abort(self):
        self.log.append("abort")
        return self.port.close()

    def read_back(self, stop_grid=False):
        self.log.append("read_back, grid stopped" if stop_grid else "read_back")
        assert not self.port.is_open, "read back with the X280 runtime open"
        return super().read_back()


def load_session(monkeypatch, tmp_path):
    """komorebi.session imported over a stub ttnn; returns (module, log, ttnn stub).
    The fake native module's Dispatch is the last one made: `module.dispatch`."""
    log = []
    ttnn = types.ModuleType("ttnn")
    ttnn.generic_op = lambda io, descriptor: log.append("generic_op")
    ttnn.synchronize_device = lambda device: log.append("synchronize")
    ttnn.close_mesh_device = lambda device: log.append("close_device")
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delitem(sys.modules, "komorebi.session", raising=False)
    session = importlib.import_module("komorebi.session")
    monkeypatch.delitem(sys.modules, "komorebi.session")  # undone: removed again

    module = types.SimpleNamespace(dispatch=None, grid_returns=True)
    module.write_l1 = lambda *args: log.append(("write_l1", *args))

    def wait_grid(device, gx, gy, timeout_s):
        log.append("wait_grid")
        return module.grid_returns

    module.wait_grid = wait_grid

    def make(device, pll_mhz):
        module.dispatch = FakeDispatch(log, device, pll_mhz)
        return module.dispatch

    module.Dispatch = make
    image = tmp_path / "km-dispatch.bin"
    image.write_bytes(b"\0" * 64)
    monkeypatch.setattr(session.native, "load_module", lambda tree=None: module)
    monkeypatch.setattr(session.native, "image_path", lambda tree=None: image)
    monkeypatch.setattr(session, "aiclk_mhz", lambda card=None: 1350)
    monkeypatch.setattr(session, "CHIP_LOCK", str(tmp_path / "no-chip-lock-{card}"))
    state = types.SimpleNamespace(buffer_address=lambda: 0x40000)
    monkeypatch.setattr(session, "_state_area", lambda device: state)
    module.card_backend = session._Backend  # the real one, for its own tests
    monkeypatch.setattr(session, "_Backend", SessionBackend)
    return session, log, ttnn, module


def program(tmp_path):
    for kernel in ("reader", "compute", "writer"):
        (tmp_path / f"{kernel}.cpp").write_text("")
    return Program("p", kernels=tmp_path)
