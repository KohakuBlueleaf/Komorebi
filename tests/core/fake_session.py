"""Stand-ins for the native module (device, launch, dispatch) and the session's card
backend, so that Session.load and its failure paths run on the CPU over the fake
kernel."""

import importlib
import sys
import types

from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.program.description import Program
from komorebi.session import process

CONSTANTS = abi.headers().constants


class FakeDispatch(FakePort):
    """The native Dispatch over the fake kernel: prepare, start, wait_ready, close."""

    grid_x, grid_y = 12, 10
    lim_base = 0x10000
    boot_error = None  # a KMD_ERROR_* the kernel ends with before READY

    def __init__(self, log, device, pll_mhz):
        super().__init__()
        self.log = log
        self.is_open = True
        self.settle_microseconds = 1.0
        self.ready_state = CONSTANTS["KMD_STATE_READY"]
        self.idle = True
        self.fail = {}  # call name -> exception it raises
        self.cleaned = True  # what close() answers
        self.close_failed = ""
        self.markers = (7, 7)  # core 0's pass marker before and after a probe's gap

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
        if not self.cleaned:
            self.close_failed = "cluster 0: core 0 phase=0x2 stage=4"
        return self.cleaned

    def aborted(self):
        x280_cores = [0] * 16
        x280_cores[CONSTANTS["KMD_H_ABANDONED"]] = 2
        return {"words": [0] * CONSTANTS["KMD_R_WORDS"], "x280_cores": x280_cores}

    def probe(self, gap_s):
        words = [0] * CONSTANTS["KMD_R_WORDS"]
        words[CONSTANTS["KMD_R_STATE"]] = CONSTANTS["KMD_STATE_READY"]
        words[CONSTANTS["KMD_R_WHERE"]] = CONSTANTS["KMD_WHERE_LOOP"]
        before, after = self.markers
        states = "core 0 phase=0x2 trap(mcause=0x7)"
        out = {"running": True, "failed": "", "x280_states": states, "words": words}
        out["x280_cores"] = [0] * 16
        return out | {"marker_before": before, "marker_after": after}


class SessionBackend(FakeBackend):
    """The session's backend over a FakeDispatch, logging what it is asked to do."""

    def __init__(self, session, dispatch, mhz, trace):
        super().__init__(dispatch, trace_cap=trace)
        self.session = session
        self.log = dispatch.log
        self.io = []

    def describe(self, *args):
        self.log.append("describe")
        return FakeLaunch(self.session._module)

    def load_overlays(self, program, descriptor):
        """Every run's service overlay: nothing to place without a card."""

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


class FakeDevice:
    """An open device: its close and its wait for the launched cores, logged."""

    def __init__(self, log):
        self.log = log

    def close(self):
        self.log.append("close_device")

    def finish(self):
        self.log.append("synchronize")

    def host_channel(self):
        return 0  # the driver mapped no host memory


class FakeLaunch:
    """The native Launch: build() and run() call the fake module's kernel_build and
    run, which tests replace."""

    def __init__(self, module):
        self.module = module

    def build(self):
        return self.module.kernel_build(self)

    def run(self):
        self.module.run(self)


def built(kernels, total=4096, limit=70656, error=""):
    """What the native kernel_build returns: kernels, config bytes, the buffer, the
    text per Tensix thread, the runtime's message."""
    return kernels, total, limit, [1024, 512, 256, 256, 768], error


def load_session(monkeypatch, tmp_path):
    """komorebi.session.session over a fake native module; returns (session module, log,
    native module). Its Dispatch is the last one made: `module.dispatch`; the sizes
    each Device was opened with are in `module.opened`."""
    log = []
    monkeypatch.delitem(sys.modules, "komorebi.session.session", raising=False)
    session = importlib.import_module("komorebi.session.session")
    monkeypatch.delitem(
        sys.modules, "komorebi.session.session"
    )  # undone: removed again

    module = types.SimpleNamespace(dispatch=None, grid_returns=True, opened=[])
    module.run = lambda launch: log.append("launch")

    def device(worker_l1_size=None):
        module.opened.append(worker_l1_size)
        return FakeDevice(log)

    module.Device = device
    module.write_l1 = lambda *args: log.append(("write_l1", *args))

    def wait_grid(device, grid_x, grid_y, timeout_s):
        log.append("wait_grid")
        return module.grid_returns

    module.wait_grid = wait_grid

    def make(device, pll_mhz):
        module.dispatch = FakeDispatch(log, device, pll_mhz)
        return module.dispatch

    module.Dispatch = make
    module.kernel_build = lambda launch: built([])  # nothing to check
    image = tmp_path / "km-dispatch.bin"
    image.write_bytes(b"\0" * 64)
    monkeypatch.setattr(session.native, "load_module", lambda tree=None: module)
    monkeypatch.setattr(session.native, "image_path", lambda tree=None: image)
    monkeypatch.setattr(session.native, "metal_tree", lambda tree=None: tmp_path)
    monkeypatch.setattr(process, "aiclk_mhz", lambda card=None: 1350)
    monkeypatch.delenv(process.OP_TIMEOUT, raising=False)  # restored after the test
    monkeypatch.setattr(process, "CHIP_LOCK", str(tmp_path / "no-chip-lock-{card}"))
    state = types.SimpleNamespace(address=0x40000)
    monkeypatch.setattr(session, "_state_area", lambda host_session: state)
    module.card_backend = session._Backend  # the real one, for its own tests
    monkeypatch.setattr(session, "_Backend", SessionBackend)
    return session, log, module


def program(tmp_path):
    for kernel in ("reader", "compute", "writer"):
        (tmp_path / f"{kernel}.cpp").write_text("")
    return Program("p", kernels=tmp_path)
