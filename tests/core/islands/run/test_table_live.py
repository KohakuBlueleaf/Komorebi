"""A tensor made while an island run lives: the table holds `tensors=` entries from the
load, and the row of a later tensor is written by each island's ranks (ENTRY with
KM_ENTRY_TABLE), from a DRAM page at the row's offset mod 64 (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import abi
from komorebi.errors import KomorebiError
from komorebi.islands import graph as islands
from komorebi.islands.session import TENSORS_MAX, IslandSession
from komorebi.session.session import Session

H = abi.headers()


def test_the_table_is_the_knob_s_size_and_its_bound_is_named():
    assert len(IslandSession([0, 1])._entries) == TENSORS_MAX
    assert IslandSession([0, 1], tensors=8)._table_entries() == 8
    with pytest.raises(ValueError, match=r"tensors=65537: 1 to 65536, .* L1"):
        IslandSession([0, 1], tensors=65537)


def two_islands(**options):
    made = IslandSession([0, 1], **options)
    made.sessions = (None, None)  # entries of 48 B
    return made


def test_the_table_is_cached_only_where_that_takes_less_l1_or_as_the_knob_says():
    k = H["KM_TABLE_CACHE_ENTRIES"]
    cache = 64 + 2 * (320 + 32 * 64)  # head; a RISC: 32 tags, 32 uses, 2 words; rows
    assert two_islands(tensors=300)._cache_bytes() == cache == 4800
    cases = [
        ({"tensors": 8}, {}),
        ({"tensors": 32}, {}),
        ({"tensors": 100}, {"KM_TENSORS_MAX": 100}),  # 4,800 B flat: not smaller
        ({"tensors": 101}, {"KM_TENSORS_MAX": 101, "KM_TABLE_CACHE": k}),
        ({"tensors": 300, "tensor_cache": False}, {"KM_TENSORS_MAX": 300}),
        ({"tensors": 8, "tensor_cache": 16}, {"KM_TABLE_CACHE": 16}),
    ]
    for options, bound in cases:
        assert two_islands(**options)._table_bound() == bound, options
    held = [two_islands(tensors=n).held_bytes(apart=True) for n in (300, 1000)]
    flat = two_islands(tensors=100).held_bytes(apart=True)
    assert held == [held[0]] * 2 and held[0] - flat == cache - 100 * 48
    for bad in (True, 2, 6, 2048):
        with pytest.raises(ValueError, match="multiple of 4 from 4 to 1024"):
            IslandSession([0, 1], tensor_cache=bad)


def test_misses_are_read_from_each_core_s_reader_and_writer_part():
    made = two_islands(tensors=300)
    words = np.zeros((4, 4800 // 4), np.uint32)
    reader = 64 // 4 + 2 * 32 + 1  # after the head: tags, uses, the clock
    words[:, reader], words[:, reader + (320 + 32 * 64) // 4] = [1, 2, 3, 4], 7
    made.sessions = [SimpleNamespace(read=lambda t: (words.view(np.uint8),))] * 2
    made._tables = [SimpleNamespace()] * 2
    assert made.table_misses() is None  # flat
    made._drams = [SimpleNamespace()] * 2
    one, _ = made.table_misses()
    assert one.tolist() == [[1, 7], [2, 7], [3, 7], [4, 7]]
    made.run = SimpleNamespace(stopped=False)
    with pytest.raises(KomorebiError, match=r"after run.stop\(\)"):
        made.table_misses()


class Scratch:
    def __init__(self, data, freed):
        self.data, self.address, self.freed = data, 0x40000, freed

    def free(self):
        self.freed.append(self.address)


class IslandRunPart:
    def __init__(self, first):
        self.island_first, self.island_ranks, self.cores = first, 118, 120
        self._flag_part, self.queued, self.handed = 1 << 3, [], 0

    def _queue_command(self, handler, group, words, exclusive, flags):
        self.queued.append((handler, group, words, flags))
        return len(self.queued) - 1

    def _hand(self):
        self.handed += 1

    def wait(self, handle, timeout):
        assert self.handed == 1  # every island's command handed before a wait
        return SimpleNamespace(kind="done")


def live_session(table_address):
    made = IslandSession([0, 1], tensors=8)
    freed, written = [], []

    def tensor(data, dtype):
        written.append(np.asarray(data).view(np.uint8).reshape(-1))
        return Scratch(data, freed)

    made.sessions = (SimpleNamespace(tensor=tensor, dram_banks=8),) * 2
    made._tables = [SimpleNamespace(address=table_address)] * 2
    made.run = SimpleNamespace(
        stopped=False, runs=[IslandRunPart(0), IslandRunPart(118)]
    )
    row = bytes(range(islands.tensor_entry_bytes(2)))
    made._entries[3] = SimpleNamespace(entry=lambda banks: row)
    return made, row, freed, written


def test_a_live_row_is_one_entry_command_an_island_at_its_offset():
    made, row, freed, written = live_session(0x1000)
    made._table_row(3)
    offset = (0x1000 + 3 * len(row)) % 64  # 144 % 64
    assert offset == 16
    for island, run in enumerate(made.run.runs):
        assert run.queued == [
            (
                H["KM_HANDLER_ENTRY"],
                (118 * island, 118),
                (H["KM_ENTRY_TABLE"], 3, 0x40000 + offset),
                1 << 3,
            )
        ]
        assert bytes(written[island][offset : offset + len(row)]) == row
    assert len(freed) == 2  # the scratch pages freed after every island is done


def test_a_row_made_with_no_live_run_is_written_by_the_host():
    made, _, _, _ = live_session(0x1000)
    made.run.stopped = True
    wrote = []
    made._write_table = lambda: wrote.append(True)
    made._table_row(3)
    assert wrote == [True] and made.run.runs[0].queued == []


def test_a_cached_table_s_live_row_is_its_dram_page_and_the_ranks_forget_it():
    made, row, freed, written = live_session(0x1000)
    pages = []
    made.sessions = [
        SimpleNamespace(
            write=lambda tensor, page, first, i=island: pages.append((i, first, page)),
            dram_banks=8,
        )
        for island in range(2)
    ]
    made._drams = [SimpleNamespace(address=0x9000)] * 2
    made._table_row(3)
    for island, run in enumerate(made.run.runs):
        assert run.queued[0][2] == (H["KM_ENTRY_TABLE"], 3, 0)
        _, first, page = pages[island]
        assert first == 3 and page.shape == (1, 64) and bytes(page[0, :48]) == row
    assert freed == [] and written == []  # no scratch page


def test_a_cached_table_s_cores_hold_its_head_and_empty_caches():
    made = two_islands(tensors=300)
    made._drams = [SimpleNamespace(address=0x9000), SimpleNamespace(address=0xA000)]
    head = made._l1_row(1)
    assert (
        head.nbytes == made._cache_bytes() and head[0] == 0xA000 and not head[1:].any()
    )
    made._table_size = 300
    made._entries[299] = SimpleNamespace(entry=lambda banks: bytes(range(48)))
    made.sessions = (SimpleNamespace(dram_banks=8),) * 2
    rows = made._dram_rows()
    assert rows.shape == (300, 64) and bytes(rows[299, :48]) == bytes(range(48))


def test_before_the_first_load_the_held_rows_are_those_the_load_keeps():
    """A class made before the first load gets the room a later one does: the rows
    the load makes (own words, island table, tensor table) are counted already, and
    a card session counts no page bounds list (the load's table bounds the pages)."""
    made = two_islands(tensors=32)
    made.graph, made._regions = SimpleNamespace(islands=[0, 1]), []
    before = made._held_l1(0)
    sizes = {
        "own words": H["KM_OWN_ALL_BYTES"],
        "island table": 64,
        "tensor table, flat (32 entries)": 32 * 48,
    }
    at = " (made at the first load)"
    assert before == [(f"session: {name}{at}", size) for name, size in sizes.items()]

    def held(size):
        return [SimpleNamespace(aligned_page_bytes=size, freed=False)]

    made._held = [{"own words": held(sizes["own words"]), "island table": held(64)}]
    made._tables, made._table_size = held(32 * 48), 32
    after = made._held_l1(0)  # after the load: the same rows, made
    assert after == [(name.replace(at, ""), size) for name, size in before]
    card = SimpleNamespace(page_bounds=1024, l1_held=lambda: before)
    assert Session.page_bounds_list(card, [SimpleNamespace()]) is None


def test_a_split_card_counts_before_its_first_load_what_that_load_makes():
    """A split card's one session: before the first load the rows it makes, its link
    area too, and a bounds list still counted (its load makes one); after the load
    none, as l1_taken counts the tensors made."""
    made = IslandSession([0], islands=3, tensors=32)
    made.sessions = (None, None, None)  # entries of 64 B
    made.graph, made._regions = SimpleNamespace(islands=[0, 1, 2]), []
    at = " (made at the first load)"
    sizes = {
        "own words": H["KM_OWN_ALL_BYTES"],
        "island table": 128,
        "split card's tensor table, flat (32 entries)": 32 * 64,
        "link area": H["KM_LINK_AREA_BYTES"],
    }
    before = made._split_held()
    assert before == [(f"session: {name}{at}", size) for name, size in sizes.items()]
    card = SimpleNamespace(
        page_bounds=1024, l1_held=lambda: before, _dram=Live(), dram_banks=8
    )
    assert Session.page_bounds_list(card, [SimpleNamespace()]) is not None
    made._held = [{}]
    assert made._split_held() == []


class Live:
    """A session's DRAM tensors: none."""

    def live(self):
        return []
