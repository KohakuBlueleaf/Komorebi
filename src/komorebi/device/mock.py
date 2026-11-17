"""The mock device a prebuild's script runs on: whether this process is one, the
remembered sizes' copy it keeps, and the run, trace and X280 dispatch that stand in
for the card's (docs/reference/host.md#building-with-no-card)."""

import os
from pathlib import Path

import numpy as np

from komorebi import abi
from komorebi.device.events import Event, IslandEvent
from komorebi.session import state

COMMAND = "prebuild"  # the command's and the call's one spelling: kmr prebuild
ACTIVE = "KMR_PREBUILD"  # set in the script's process: its loads build, then mock
MOCK = "TT_METAL_MOCK_CLUSTER_DESC_PATH"
MEMORY = "KMR_PREBUILD_MEMORY"  # the remembered sizes a prebuild reads and keeps
MHZ = 1000  # a mock has no clock: the cycles of a limit, never compiled in
LOADS = [0]  # the loads this process built


class Nothing(float):
    """What a mock run's call of no known shape gives: any field, call or item is
    itself, 0.0 as a number (JSON too), it iterates over nothing and is true."""

    def __new__(cls):
        return super().__new__(cls, 0.0)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def __int__(self):
        return 0

    __index__ = __int__

    def __format__(self, spec):
        try:
            return format(0.0, spec)
        except ValueError:
            return format(0, spec)

    def __arith__(self, other):
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __arith__
    __truediv__ = __rtruediv__ = __floordiv__ = __mod__ = __pow__ = __arith__


NOTHING = Nothing()


TICKS = 50  # a mock command's time: 1 us of the X280's 50 MHz timer


class EmptyTrace:
    """A mock run's trace: no core's words for any command; phases and leads None."""

    capacity = 0

    def command(self, handle):
        return np.zeros((0, 4), np.int64)

    def phases(self, handle):
        return None

    lead = island = lambda self, *args: None


class Run:
    """The run a prebuild's load returns in place of the card's: what a run in which
    every command succeeds answers (done events, a clean result block, empty state and
    trace); what has no known shape, Nothing. `parts`: an island run's runs; `host`:
    the session's Run (an IslandRun) over the mock's backend, which builds chains."""

    def __init__(self, counted=True, parts=(), host=None):
        self.stopped, self.parts, self._next = False, list(parts), 1
        self.host = host
        self._results = set()  # handles submitted with result=True
        LOADS[0] += int(counted)

    def stop(self, timeout=None):
        """Ends it, its parts and its host run (the L1 its backend took for the
        launch given back, as a card run's finish does); the result block."""
        self.stopped = True
        for part in self.parts:
            part.stop()
        hosts = getattr(self.host, "runs", (self.host,))  # an IslandRun: each card's
        for host in hosts:
            release = getattr(getattr(host, "backend", None), "release_l1", None)
            if release is not None:
                release()
        return self.result()

    abort = stop

    def alive(self):
        return not self.stopped

    def submit(self, *args, **kwargs):
        if kwargs.get("result"):
            self._results.add(self._next)
        self._next += 1
        return self._next - 1

    load = submit  # an added program's ticket

    def chain(self, *args, **kwargs):
        """The host's run.chain: its descriptions, proofs, refusals and records, as on
        the card (the records in the mock's DRAM); its submit is this run's."""
        if self.host is None:
            return NOTHING
        return self.host.chain(*args, **kwargs)

    def submit_many(self, handler, firsts, *args, **kwargs):
        count = len(np.atleast_1d(firsts))
        self._next += count
        return range(self._next - count, self._next)

    def event(self, handle):
        """A done event of 1 us (a result word asked for: 0, as reads give zeros); an
        island run's, one part an island."""
        result = 0 if int(handle) in self._results else None
        done = Event(int(handle), "done", 0, TICKS, result)
        return IslandEvent([done] * len(self.parts)) if self.parts else done

    def wait(self, handles, timeout=None):
        if isinstance(handles, int | np.integer):
            return self.event(handles)
        return [self.event(handle) for handle in handles]

    def wait_any(self, handles, timeout=None):
        return self.event(min(handles))

    def transfer(self, *args, **kwargs):
        return self.event(0)

    wait_engine = transfer

    def drain(self, timeout=None):
        return 0

    def given_up(self, handle):
        return None

    def holds(self):
        return None

    def result(self):
        """The result block of a run that ended clean: state done, error none."""
        h = abi.headers()
        words = np.zeros(h["KMD_R_WORDS"], np.uint32)
        words[h["KMD_R_STATE"]] = h["KMD_STATE_DONE"]
        block = state.result_block(words, h)
        return [dict(block) for _ in self.parts] if self.parts else block

    def state(self):
        return []

    def inspect(self, *args, **kwargs):
        return []

    def trace(self):
        """An empty trace: a command's rows are none (its [0, 4] words)."""
        return EmptyTrace()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return NOTHING


def active():
    """Whether this process is a prebuild's script."""
    return bool(os.environ.get(ACTIVE))


def memory(real):
    """The remembered sizes' file a prebuild reads and writes in place of `real` (a
    copy of it, the prebuild's own: the card's file is never written), else None."""
    if not active() or not os.environ.get(MEMORY):
        return None
    own = Path(os.environ[MEMORY])
    if not own.exists() and real.is_file():
        own.write_bytes(real.read_bytes())
    return own


def descriptor(cards):
    """The mock cluster of `cards` (one or two P150 cards), shipped in the package."""
    if len(cards) not in (1, 2):
        raise ValueError(f"cards {list(cards)}: one card or two")
    name = "p150.yaml" if len(cards) == 1 else "p150x2.yaml"
    return Path(__file__).resolve().parents[2] / "komorebi_native" / "mock" / name


class Dispatch:
    """The X280 dispatch a mock lacks: what a load asks of it before the launch (the
    grid, the core table, the dispatcher's tile and LIM base: runtime arguments)."""

    idle, close_failed, lim_base, l2_x, l2_y = True, "", 0x20000000, 8, 3
    settle_microseconds, upload_bytes = 0.0, 0

    def __init__(self, device):
        self.device = device
        self.grid_x, self.grid_y = device.grid()

    def prepare(self, *args, **kwargs):
        return 1

    def core_table(self):
        cores = []
        for index in range(self.grid_x * self.grid_y):
            x, y = self.device.worker_core(index % self.grid_x, index // self.grid_x)
            cores.append(x | y << 16)
        return np.array(cores + [0] * (128 - len(cores)), dtype=np.uint64)

    def work_info(self, cluster):
        return {"virtual_x": self.l2_x, "virtual_y": self.l2_y, "lim_base": 0}

    def close(self):
        return True
