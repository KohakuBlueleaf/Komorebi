"""The host landing's checks (no card): each flag forced by a stand-in, its value or
message printed. Run with PYTHONPATH at the source to check (the base or the fix)."""

import sys
import traceback
from types import SimpleNamespace

import numpy as np

from komorebi import formats
from komorebi.errors import WaitTimeout
from komorebi.islands import graph as islands
from komorebi.islands import run as island_run
from komorebi.islands.parts.io import IoMixin
from komorebi.islands.parts.table import TableMixin
from komorebi.islands.session import IslandSession
from komorebi.session.parts.memory import MemoryMixin
from komorebi.session.parts.transfer import TransferMixin


def said(name, call):
    try:
        print(f"{name}: {call()}")
    except Exception as error:  # noqa: BLE001 (printed)
        print(f"{name}: raised {type(error).__name__}: {str(error)[:160]}")


def turned_tensor():
    rng = np.random.default_rng(1)
    data = formats.from_bytes(
        formats.to_bytes(rng.standard_normal((32, 64)), "bfloat16", "tile")[0],
        (32, 64), "bfloat16", "tile",
    )  # fmt: skip
    raw, _ = formats.to_bytes(data, "bfloat16", "tile", True)
    tensor = SimpleNamespace(
        layout="tile", dtype="bfloat16", shape=(32, 64), transposed_tiles=True,
        page_tiles=1, pages=2, page_bytes=2048,
    )  # fmt: skip
    return data, tensor, raw.reshape(2, 2048)


def t1():
    data, tensor, rows = turned_tensor()
    card = SimpleNamespace(read=lambda *t, **k: [rows] * len(t))
    whole = TransferMixin.read_bf16(card, tensor)[0]
    tiles = TransferMixin.read_tiles_bf16(card, tensor, 0, 2)
    want = data.reshape(1, 32, 2, 32).swapaxes(1, 2).reshape(2, 32, 32)
    islands = IoMixin.read_bf16(card, tensor)[0]
    return (
        f"read_bf16 exact {np.array_equal(whole, data)}, read_tiles_bf16 exact "
        f"{np.array_equal(tiles, want)}, islands read_bf16 exact "
        f"{np.array_equal(islands, data)}"
    )


def writer(tensor):
    """A card session stand-in whose write goes through the channel (its rows kept)."""
    got = {}
    card = SimpleNamespace(
        _module=object(), tree=None, _region=SimpleNamespace(data=1 << 20), run=None,
        channel=np.zeros(1 << 20, np.uint8), _mock_mapped=lambda: False,
        _live_pieces=lambda t: None, _transfer_way=lambda *a: "way",
        _transfer=lambda t, first, pages, way, push: got.setdefault(
            "rows", card.channel[: pages * t.aligned_page_bytes].copy()
        ),
    )  # fmt: skip
    return card, got


def page_tensor(**more):
    given = dict(
        pages=2, page_bytes=64, aligned_page_bytes=64, kept=None, sent=None,
        written=False, in_dram=True, shape=(2, 64), address=0x40,
    )  # fmt: skip
    return SimpleNamespace(**{**given, **more})


def t2():
    tensor = page_tensor(sent="call 1 (a) of a chain")
    card, _ = writer(tensor)
    card._push = lambda t, first, pages, way: np.full(pages * 64, 7, np.uint8)
    chunks = list(TransferMixin.read_chunks(card, tensor))
    return f"read_chunks gave {len(chunks)} chunk(s) of the old bytes"


def t3():
    tensor = page_tensor(sent="call 1 (a) of a chain")
    card, _ = writer(tensor)
    TransferMixin.write(card, tensor, np.zeros((2, 64), np.uint8))
    return f"after session.write, tensor.sent = {tensor.sent!r}"


def t6():
    tensor = page_tensor()
    card, got = writer(tensor)
    TransferMixin.write(card, tensor, np.full((2, 64), 300.5, np.float32))
    return f"a [2, 64] float32 page of 300.5 written as byte {int(got['rows'][0])}"


def t7():
    tensor = page_tensor(kept=[])
    card, _ = writer(tensor)
    try:
        TransferMixin.write(card, tensor, np.zeros((2, 32), np.uint8))
    except ValueError as error:
        return f"refused ({error}); kept {len(tensor.kept)} write(s) for a reopen"
    return "not refused"


def t9():
    done = []
    module = SimpleNamespace(
        Buffer=lambda *a: SimpleNamespace(
            write=lambda raw: done.append("buffer.write (waits for the run's cores)"),
            address=0x1000,
            size=a[1],
            page_size=a[2],
            num_pages=a[1] // a[2],
        )
    )
    card = SimpleNamespace(
        _module=module, tree=None, device=None,
        _note_page=lambda t: None, _transfer_way=lambda *a: None,
        _run_loaded=lambda: True,
        write=lambda t, rows: done.append("session.write (pieces or refused)"),
        _kept=SimpleNamespace(keep=lambda *a, **k: None), _dram=set(),
    )  # fmt: skip
    MemoryMixin.tensor(card, np.zeros((1, 64), np.uint32), "uint32", shard_grid=(1, 1))
    return f"a tensor made with a run loaded: {done}"


class StandRun:
    """A Run's waits: its events kept until forgotten; `late` times out once."""

    def __init__(self, late=False):
        self.late, self.gone = late, set()

    def _wait_once(self, handles):
        if any(h in self.gone for h in handles):
            raise ValueError(
                f"handle {handles[0]}: its event was returned or forgotten"
            )
        if self.late:
            self.late = False
            raise WaitTimeout("island's part not done", list(handles))

    def wait(self, handles, timeout=None):
        self._wait_once(handles)
        events = [self.event(h) for h in handles]
        self.forget(handles)
        return events

    def _wait_kept(self, handles, timeout=None):
        self._wait_once(handles)

    def event(self, handle):
        return None if handle in self.gone else SimpleNamespace(kind="done", ticks=5)

    def forget(self, handles):
        self.gone.update(handles)

    def _arrived(self, handles):
        return np.ones(len(handles), bool)

    def abort(self):
        self.aborted = True
        if getattr(self, "fails", False):
            raise RuntimeError("cluster 0 not restored")


def islands_run(runs):
    fake = SimpleNamespace(
        runs=runs, _next=2, _parts=[np.array([0, 1]), np.array([0, 1])],
        send=lambda: None, _store=lambda w, d: None,
        _tickets=lambda island, pending: pending,
        _error=lambda error, island: error, _raised=lambda e, i, d: e,
        _crossed=lambda *a: None,
    )  # fmt: skip
    return fake


def h1():
    fake = islands_run([StandRun(), StandRun(late=True)])
    first = "no timeout"
    try:
        island_run.IslandRun._wait_tickets(fake, 1, 1.0)
    except WaitTimeout:
        first = "WaitTimeout"
    again = island_run.IslandRun._wait_tickets(fake, 1, 1.0)
    return f"first wait {first}; waited again: {again.kind if again else again}"


def h2():
    queued = []

    class Part:
        def __init__(self, refuses):
            self.refuses = refuses

        def _submit_words(self, *a):
            if self.refuses:
                raise ValueError("group does not hold program")
            return a

        def submit(self, *a, **k):
            self._submit_words()
            queued.append(self)
            return 1

    fake = SimpleNamespace(
        runs=[Part(False), Part(True)], _next=1,
        _parts=[np.zeros(4, np.int64), np.zeros(4, np.int64)],
        graph=SimpleNamespace(split=lambda g: [(0, 0, 4), (1, 0, 4)]),
        _check_half_added=lambda h: None, _x280_holder=lambda *a: None,
        _dynamic=lambda h: False, _first_use=lambda h, g: None,
        _grow=lambda n: None, _error=lambda e, i: e, _check_local=lambda h, g: None,
    )  # fmt: skip
    fake._part_checked = lambda *a: island_run.IslandRun._part_checked(fake, *a)
    try:
        island_run.IslandRun._submit(fake, 1, (0, 8), (), False, False, None, None)
    except ValueError as error:
        return f"refused ({error}); parts queued before it: {len(queued)}"
    return "not refused"


def h2_many():
    queued = []

    class Part:
        def __init__(self, refuses):
            self.refuses = refuses

        def _many_words(self, *a):
            if self.refuses:
                raise ValueError("handler 3: not NOP and not in 1..KM_HANDLER_USER_MAX")

        def submit_many(self, *a, **k):
            self._many_words()
            queued.append(self)
            return [1]

    fake = SimpleNamespace(
        runs=[Part(False), Part(True)], _next=1,
        _parts=[np.zeros(4, np.int64), np.zeros(4, np.int64)],
        graph=SimpleNamespace(first=lambda i: 4 * i, count=lambda i: 4),
        _grow=lambda n: None, _error=lambda e, i: e,
    )  # fmt: skip
    try:
        island_run.IslandRun._queue_many(
            fake, 3, True, None, np.array([0]), np.array([8]), None, False
        )
    except ValueError as error:
        return f"refused ({error}); islands' rows sent before it: {len(queued)}"
    return "not refused"


def h3():
    runs = [StandRun(), StandRun()]
    runs[0].fails = True
    fake = SimpleNamespace(runs=runs)
    try:
        island_run.IslandRun.abort(fake)
        raised = "nothing"
    except RuntimeError as error:
        raised = str(error)
    aborted = [getattr(run, "aborted", False) for run in runs]
    return f"islands aborted {aborted}, raised: {raised}"


def is1():
    tensor = SimpleNamespace(shape=(32, 32), address=0x40, sent=None, pages=1)
    sub = SimpleNamespace(submit=lambda *a, **k: 1)
    one = SimpleNamespace(
        pieces=1, handler=1, group=(0, 4), args=lambda w, p: (),
        direct=[(tensor, "call 0 (a) of a chain")], written=[],
    )  # fmt: skip
    fake = SimpleNamespace(
        runs=[sub, sub], _next=1, _chain_last=0, _parts=[np.zeros(4, np.int64)] * 2,
        graph=SimpleNamespace(split=lambda g: [(0, 0, 2), (1, 0, 2)]),
        _chain_after=lambda t: None, _grow=lambda n: None, _error=lambda e, i: e,
        wait=lambda t: None,
    )  # fmt: skip
    fake._part_checked = lambda *a: None
    chain = SimpleNamespace(chains=[one, one], group=(0, 4))
    island_run.IslandRun._submit_chain(fake, chain, {})
    session = SimpleNamespace(
        card_dram=True, sessions=[SimpleNamespace(read=lambda *p, **k: ["old bytes"])]
    )
    tensor.parts = [tensor]
    got = IoMixin.read(session, tensor)
    return f"after the chain tensor.sent = {tensor.sent!r}; its read gave {got}"


def e3():
    made = IslandSession([0], islands=-3)
    return f"IslandSession([0], islands=-3) made, split {made.split}"


def d2():
    islands.CORES_MAX = 64  # a header that changed
    try:
        fake = SimpleNamespace(
            sessions=[1, 2], _tables=[], _drams=[], _table_size=0,
            _table_entries=lambda: (_ for _ in ()).throw(RuntimeError("checked past")),
        )  # fmt: skip
        TableMixin._tables_for_load(fake)
        return "not refused"
    finally:
        islands.CORES_MAX = 128


for name, check in (
    ("T1", t1), ("T2", t2), ("T3", t3), ("T6", t6), ("T7", t7), ("T9", t9),
    ("H1", h1), ("H2", h2), ("H2 many", h2_many), ("H3", h3), ("IS1", is1),
    ("E3", e3), ("D2", d2),
):  # fmt: skip
    try:
        said(name, check)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stdout)
