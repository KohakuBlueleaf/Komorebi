"""The mock device a prebuild's script runs on: whether this process is one, the
remembered sizes' copy it keeps, and the run, trace and X280 dispatch that stand in
for the card's (docs/reference/host.md#building-with-no-card)."""

import contextlib
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi.chain import calls as chain_module
from komorebi.device.events import Event, IslandEvent
from komorebi.errors import KomorebiError
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
    __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = __arith__  # an address's
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = __arith__

    def __invert__(self):
        return self


NOTHING = Nothing()


def _named(error, island):
    """`error` named by its island as an island run names one ("island i: ...")."""
    if getattr(error, "island", None) is None:
        error.island = island
        if error.args:
            error.args = (f"island {island}: {error.args[0]}", *error.args[1:])
    return error


class EngineEnd(NamedTuple):
    """An engine's end as the card's (engines.control.EngineEvent, kind "end"): its
    registers zeros, as reads give zeros."""

    handle: object
    kind: str = "end"
    value: int = 0
    registers: tuple = (0, 0)
    reason: int | None = None
    reason_name: str | None = None
    pc: int | None = None
    node: int | None = None


TICKS = 50  # a mock command's time: 1 us of the X280's 50 MHz timer


class EmptyTrace:
    """A mock run's trace: no core's words for any command; phases and leads None."""

    capacity = 0
    mhz = MHZ  # the mock's clock, where a card's trace has its AICLK

    def command(self, handle):
        return np.zeros((0, 4), np.int64)

    def phases(self, handle):
        return None

    lead = island = lambda self, *args: None


class Thread:
    """An X280 thread of a mock run: its regions read as zeros of their sizes (the
    card's are read after the run); anything else, Nothing."""

    def __init__(self, thread):
        self.thread = thread

    def read(self, name):
        return np.zeros(self.thread.region_bytes(name), np.uint8)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return NOTHING


class Run:
    """The run a prebuild's load returns in place of the card's: what a run in which
    every command succeeds answers (done events, a clean result, zero state words, an
    empty trace); what has no known shape, Nothing. `parts`: an island run's runs;
    `host`: the session's Run (an IslandRun) over the mock's backend (chains)."""

    def __init__(self, counted=True, parts=(), host=None, threads=None, graph=None):
        self.stopped, self.parts, self._next = False, list(parts), 1
        self.host, self.graph = host, graph  # graph: an island run's, its splits
        self._spans = {}  # an island run's handle: the parts its group has
        if threads:  # the program's X280 threads by work cluster, as a card run's
            self.threads = {c: Thread(t) for c, t in sorted(threads.items())}
            self.thread = self.threads[min(threads)]
        self.diverged = False  # a mocked addon: the host run is no longer the card's
        self._results = set()  # handles submitted with result=True
        LOADS[0] += int(counted)

    @property
    def program(self):
        """The loaded program, as the card's run gives it: its host run's (an island
        run's: island 0's), Nothing without one."""
        return getattr(self.host, "program", NOTHING)

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
        self._checked(*args, **kwargs)
        self._spanned(args[1] if len(args) > 1 else kwargs.get("group"))
        if kwargs.get("result"):
            self._results.add(self._next)
        self._next += 1
        return self._next - 1

    def load(self, program=None, group=None, *args, **kwargs):
        """run.load: the group's cores hold `program` on the host run too (checked as
        the card's run checks it), so later submits are checked against them."""
        if self.host is not None and not self.parts and not self.diverged:
            field, first, count, _ = self.host._load_words(program, group)
            self.host._held(field, first, count)
        self._next += 1
        return self._next - 1

    def addon(self, *args, **kwargs):
        """run.addon builds nothing here, and the host run no longer holds what the
        card's would: its checks of later commands stop (Nothing, as before)."""
        self.diverged = True
        return NOTHING

    def _many_checked(
        self, handler, firsts, counts, args=None, exclusive=False, part=False
    ):
        """submit_many's checks as the host's run makes them, when this run has one;
        the port rows they take are given back (none of them is in flight). An island
        run: each island's rows checked by its part (_islands_many)."""
        if self.parts and not self.diverged:
            self._islands_many(handler, firsts, counts, args, exclusive)
            return
        if self.host is None or self.parts or self.diverged:
            return
        self.host.check_programs = self.__dict__.get("check_programs", True)
        ports = dict(self.host._ports)
        try:
            self.host._many_words(handler, firsts, counts, args, exclusive)
        finally:
            self.host._ports = ports

    def _checked(self, handler=None, group=None, args=(), exclusive=False, **given):
        """A command's checks as the host's run.submit makes them (its words, group,
        holds, rectangles, page bounds), when this run has one: refused as on the
        card. A chain was checked when built; a thread's command has no X280 here."""
        if self.parts and not self.diverged:
            self._islands_checked(handler, group, args, exclusive, given)
            return
        if self.host is None or self.parts or self.diverged:
            return
        if given.get("cluster") is not None:
            return
        self.host.check_programs = self.__dict__.get("check_programs", True)
        if isinstance(handler, chain_module.Chain):
            return
        given.pop("timeout", None)
        _, _, _, _, _, _, kept = self.host._submit_words(
            handler,
            group,
            args,
            exclusive,
            given.get("result", False),
            given.get("queued", False),
            None,
            given.get("block"),
            given.get("part", False),
        )
        if kept is not None:
            kept.free()  # a mock command's block: never read

    def _islands_hosted(self):
        """Whether an island run's checks can be made: its graph, and each part with
        the island's own run (a part without one checks nothing, as on one card)."""
        return self.graph is not None and all(p.host is not None for p in self.parts)

    def _islands_checked(self, handler, group, args, exclusive, given):
        """An island run's command checked as IslandRun.submit sends it: the flat
        group (default every rank) split by island, each island's part checked by its
        own run with the group unchanged (`part=True`), an error named by its island;
        a queued one whole on the first island with the count's ranks."""
        cluster = given.get("cluster") is not None
        chain = isinstance(handler, chain_module.Chain | chain_module.IslandChain)
        if not self._islands_hosted() or cluster or chain:
            return  # a chain was checked when built; a thread's: no X280 here
        given = {**given, "part": not given.get("queued")}
        if given.get("queued"):  # IslandRun._submit_queued's refusals, then its part
            if getattr(getattr(self.host, "session", None), "split", 1) > 1:
                raise KomorebiError(
                    "submit(..., queued=True) on a card split into islands: not "
                    "proven on a split card yet; run.island(i).submit(..., "
                    "queued=True) queues it on island i's ranks"
                )
            if exclusive:
                raise ValueError("a queued command cannot be exclusive")
            ranks = [self.graph.count(i) for i in range(len(self.parts))]
            count = max(ranks) if group is None else int(group)
            fits = [i for i, held in enumerate(ranks) if count <= held]
            if count < 1 or not fits:
                raise ValueError(
                    f"queued count {count}: a queued command runs on one island's "
                    f"ranks (the islands have {ranks})"
                )
            parts, word = fits[:1], count
        else:
            word = (0, self.graph.ranks) if group is None else tuple(group)
            parts = [island for island, _first, _count in self.graph.split(word)]
        for island in parts:
            part = self.parts[island]
            part.check_programs = self.__dict__.get("check_programs", True)
            try:
                part._checked(handler, word, args, exclusive, **given)
            except KomorebiError as error:
                raise _named(error, island) from None

    def _islands_many(self, handler, firsts, counts, args, exclusive):
        """An island run's submit_many checked as IslandRun.submit_many sends it: its
        groups refused past the ranks, each island's rows checked by its part."""
        if not self._islands_hosted():
            return
        first, count = np.broadcast_arrays(
            np.atleast_1d(np.asarray(firsts, np.int64)),
            np.atleast_1d(np.asarray(counts, np.int64)),
        )
        if args is not None and len(args) != len(first) and len(first) == 1:
            first, count = (np.repeat(column, len(args)) for column in (first, count))
        bad = (count < 1) | (first < 0) | (first + count > self.graph.ranks)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"group ({int(first[i])}, {int(count[i])}) of {self.graph.ranks} ranks"
            )
        host = self.parts[0].host
        if host is not None:  # a class's handler: built for its rows' constants
            rows = () if args is None else args
            handler = host._words_built_for(handler, rows, "submit_many")
        scalar = isinstance(handler, str) or np.ndim(handler) == 0
        if not scalar:
            handler = np.broadcast_to(np.asarray(handler), first.shape)
        args = None if args is None else np.asarray(args)  # as IslandRun's rows
        for island, part in enumerate(self.parts):
            low = self.graph.first(island)
            high = low + self.graph.count(island)
            rows = np.flatnonzero(
                (np.minimum(first + count, high)) > np.maximum(first, low)
            )
            if not rows.size:
                continue
            part.check_programs = self.__dict__.get("check_programs", True)
            try:
                part._many_checked(
                    handler if scalar else handler[rows],
                    first[rows],
                    count[rows],
                    None if args is None else args[rows],
                    exclusive,
                )
            except KomorebiError as error:
                raise _named(error, island) from None

    def _spanned(self, group, handle=None):
        """Keeps how many islands the group of `handle` (the next) spans (an island
        run's event has a part each), when the host's graph splits it."""
        graph = self.graph or getattr(self.host, "graph", None)
        if not self.parts or graph is None or not isinstance(group, tuple | list):
            return
        with contextlib.suppress(TypeError, ValueError):  # not one (first, count)
            spans = len(list(graph.split(tuple(group))))
            self._spans[self._next if handle is None else handle] = spans

    def island(self, index):
        """An island run's part `index` (the card's run of that island)."""
        return self.parts[index] if self.parts else NOTHING

    def chain(self, *args, **kwargs):
        """The host's run.chain: its descriptions, proofs, refusals and records, as on
        the card (the records in the mock's DRAM); its submit is this run's."""
        if self.host is None:
            return NOTHING
        return self.host.chain(*args, **kwargs)

    def submit_many(self, handler, firsts, *args, **kwargs):
        self._many_checked(handler, firsts, *args, **kwargs)
        count = len(np.atleast_1d(firsts))
        counts = args[0] if args else kwargs.get("counts")
        if counts is not None:  # a command each (first, count): its islands' parts
            pairs = zip(np.atleast_1d(firsts), np.atleast_1d(counts))
            for handle, (first, size) in enumerate(pairs, self._next):
                self._spanned((int(first), int(size)), handle)
        self._next += count
        return range(self._next - count, self._next)

    def event(self, handle):
        """A done event of 1 us (a result word asked for: 0, as reads give zeros); an
        island run's, one part an island."""
        result = 0 if int(handle) in self._results else None
        done = Event(int(handle), "done", 0, TICKS, result)
        parts = self._spans.get(int(handle), len(self.parts))
        return IslandEvent([done] * parts) if self.parts else done

    def event_parts(self, handle):
        """An island run's parts' events (event's, one a part)."""
        return tuple(self.event(handle))

    def wait(self, handles, timeout=None):
        if isinstance(handles, int | np.integer):
            return self.event(handles)
        return [self.event(handle) for handle in handles]

    def wait_any(self, handles, timeout=None):
        return self.event(min(handles))

    def transfer(self, *args, **kwargs):
        return self.event(0)

    def wait_engine(self, handle, timeout=None):
        """An engine's end (EngineEnd), as the card's wait_engine gives it."""
        return EngineEnd(handle)

    def _arrived(self, handles):
        """Every handle's event has arrived (an island session's waits read it)."""
        return np.ones(np.shape(handles), bool)

    def all_gather(self, *args, **kwargs):
        """An island run's collective (all_gather, all_reduce, reduce_to,
        broadcast_from): its event, a done part an island, as the card's; None on
        one island, as the card returns at once."""
        if len(self.parts) < 2:
            return None
        self._next += 1
        return self.event(self._next - 1)

    all_reduce = reduce_to = broadcast_from = all_gather

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
        if self.parts:  # an island run's: one block, each island's in .islands
            return state.island_result([dict(block) for _ in self.parts])
        return block

    def state(self):
        """Every core's state as the card's run gives it after the stop, its words
        zeros (a prebuild's reads give zeros); an island run's every rank's, in flat
        rank order with its rank and island."""
        if not self.stopped:
            raise KomorebiError("valid only after run.stop()")
        if self.parts and self.graph is not None:
            states = [part.state() for part in self.parts]
            return [
                {**states[island][core], "rank": self.graph.first(island) + i}
                | {"island": island}
                for island, cores in enumerate(self.graph.rank_cores)
                for i, core in enumerate(cores)
            ]
        cores = getattr(self.host, "cores", None)
        if cores is None:  # no host run: no cores known
            return []
        h = abi.headers()
        blocks = np.zeros((cores, h["KM_CONTROL_BYTES"] // 4), np.uint32)
        words = state.state_words(blocks, 0, h)
        return [
            state.core_state(words, core, h)
            | {
                "ports": state.ports(blocks[core], h),
                "broadcasts": state.broadcasts(blocks[core], h),
                "dynamic": state.dynamic(blocks[core], h),
                "overlay_epoch": None,
            }
            for core in range(cores)
        ]

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
