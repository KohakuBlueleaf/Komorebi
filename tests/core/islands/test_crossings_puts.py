"""crossings(puts=True): the COUNTS command asks a 128-byte page, and each row also
holds its link core's put words by name; the default 64-byte page gives none."""

from types import SimpleNamespace

import numpy as np

from komorebi import abi
from komorebi.islands.parts.link_plan import CROSS_PAGE
from komorebi.islands.parts.load import LoadMixin, put_words
from komorebi.islands.records import Crossings
from komorebi.islands.run import IslandRun

H = abi.headers()
PUT_FIELDS = (
    "put_tail",
    "put_taken",
    "put_done",
    "put_confirmed",
    "put_bad",
    "queue_pending",
    "queue_last",
)


def page_row(page):
    """One CPU's row of `page` bytes, as words: the crossed counts, then (128-byte
    page) the put block from KM_PUT_TAIL and the queue's first words."""
    row = np.zeros(page // 4, "<u4")
    row[H["KM_CROSS_MESSAGES"]] = 7
    if page >= H["KM_COUNTS_PUT_PAGE"]:
        put, queue = H["KM_COUNTS_PUT"] // 4, H["KM_COUNTS_QUEUE"] // 4
        row[put] = 40  # the tail: 40 numbers handed out
        for j in range(H["KM_PAGE_CPUS"]):
            row[put + (H["KM_PUT_TAKEN"] - H["KM_PUT_TAIL"]) // 4 + j] = 32 + j
        row[put + (H["KM_PUT_DONE"] - H["KM_PUT_TAIL"]) // 4] = 30
        row[put + (H["KM_PUT_CONFIRMED"] - H["KM_PUT_TAIL"]) // 4] = 29
        row[put + (H["KM_PUT_BAD"] - H["KM_PUT_TAIL"]) // 4] = 1
        row[queue + H["KM_QUEUE_PENDING_COUNT"]] = 3
        row[queue + H["KM_QUEUE_LAST"]] = 28
    return row


class Load(LoadMixin):
    """Two islands, one link core each, one link between them: the COUNTS commands
    recorded, each island's counts tensor read back as rows of its page."""

    def __init__(self):
        self.commands = []
        self.graph = SimpleNamespace(
            islands=[0, 1],
            link_group=lambda island: (0, 1),
            between=lambda a, b: SimpleNamespace(count=1),
            per=2,  # links a link core (Blackhole's)
        )
        self.link_cores = ((34,), (70,))
        self._counts = [SimpleNamespace(address=0x1000, page=CROSS_PAGE)] * 2
        page = H["KM_COUNTS_PUT_PAGE"]
        self._counts_puts = [SimpleNamespace(address=0x2000, page=page)] * 2
        part = SimpleNamespace(own_commands={}, _queue_command=self.queue)
        self.run = SimpleNamespace(island=lambda island: part, send=self.send)
        session = SimpleNamespace(read=self.read)
        self.sessions = (session, session)

    def queue(self, handler, group, words, budget=None):
        self.commands.append(list(words))
        return len(self.commands)

    def read(self, tensor):
        rows = np.stack([page_row(tensor.page)] * 2)
        return (rows.view(np.uint8),)

    def pair(self, island, k):
        return 1 - island, 0, None

    def send(self):
        self.commands.append("send")

    def wait_together(self, waits, timeout=10.0):
        self.commands.append("wait")


def test_the_default_page_gives_no_put_words():
    load = Load()
    rows = load._crossings().rows
    asked = load.commands[:2]
    assert [w[H["KM_PAGES_PAGE"]] for w in asked] == [CROSS_PAGE] * 2
    assert set(rows) == {(0, 0, 1), (0, 1, 0)}
    for row in rows.values():
        assert row["messages"] == 7
        assert not set(PUT_FIELDS) & set(row)


def test_puts_asks_the_larger_page_and_names_the_put_words():
    load = Load()
    rows = load._crossings(puts=True).rows
    page = H["KM_COUNTS_PUT_PAGE"]
    asked = load.commands[:2]
    assert page == 128 and [w[H["KM_PAGES_PAGE"]] for w in asked] == [page] * 2
    assert [w[H["KM_PAGES_TO"]] for w in asked] == [0x2000] * 2
    for row in rows.values():
        assert row["messages"] == 7
        assert {f: row[f] for f in PUT_FIELDS} == {
            "put_tail": 40,
            "put_taken": (32, 33, 34, 35),
            "put_done": 30,
            "put_confirmed": 29,
            "put_bad": 1,
            "queue_pending": 3,
            "queue_last": 28,
        }


def test_counts_of_one_island_store_every_island_s_queued_commands_first():
    load = Load()
    load._crossings(islands=(0,))
    assert len(load.commands) == 3 and load.commands[1:] == ["send", "wait"]


def test_put_words_reads_the_offsets_of_the_abi():
    words = put_words(page_row(H["KM_COUNTS_PUT_PAGE"]), H)
    assert tuple(words) == PUT_FIELDS
    assert len(words["put_taken"]) == H["KM_PAGE_CPUS"]


def test_the_run_passes_puts_only_when_asked():
    run = IslandRun.__new__(IslandRun)
    asked = []
    run._counts = lambda **options: asked.append(options) or Crossings({})
    run.crossings()
    run.crossings(puts=True)
    assert asked == [{}, {"puts": True}]
