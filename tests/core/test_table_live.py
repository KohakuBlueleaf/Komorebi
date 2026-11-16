"""A tensor made while an island run lives: the table holds `tensors=` entries from the
load, and the row of a later tensor is written by each island's ranks (ENTRY with
KM_ENTRY_TABLE), from a DRAM page at the row's offset mod 64 (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import abi, islands
from komorebi.island_session import TENSORS_MAX, IslandSession

H = abi.headers()


def test_the_table_is_the_knob_s_size_and_its_bound_is_named():
    assert len(IslandSession([0, 1])._entries) == TENSORS_MAX
    assert IslandSession([0, 1], tensors=8)._table_entries() == 8
    with pytest.raises(ValueError, match=r"tensors=65537: 1 to 65536, .* L1"):
        IslandSession([0, 1], tensors=65537)


def test_a_table_over_32_entries_is_the_kernels_bound_and_cached_only_then():
    cached = {"KM_TENSORS_MAX": 300, "KM_TABLE_CACHE": H["KM_TABLE_CACHE_ENTRIES"]}
    for entries, bound in ((8, {}), (32, {}), (300, cached)):
        made = IslandSession([0, 1], tensors=entries)
        assert made._table_bound() == bound
    held = []
    for entries in (100, 300):  # a cached table's L1: the same whatever its entries
        made = IslandSession([0, 1], tensors=entries)
        made.sessions = (None, None)
        held.append(made.held_bytes(apart=True))
    two = IslandSession([0, 1])
    two.sessions = (None, None)
    # head 64; each RISC: 32 tags and 16 ways (192 B), 32 rows of 64 B
    assert held == [held[0]] * 2 and held[0] - two.held_bytes(apart=True) == (
        64 + 2 * (192 + 32 * 64) - 32 * islands.tensor_entry_bytes(2)
    )


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

    made.sessions = (SimpleNamespace(tensor=tensor),) * 2
    made._tables = [SimpleNamespace(address=table_address)] * 2
    made.run = SimpleNamespace(
        stopped=False, runs=[IslandRunPart(0), IslandRunPart(118)]
    )
    row = bytes(range(islands.tensor_entry_bytes(2)))
    made._entries[3] = SimpleNamespace(entry=lambda: row)
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
            write=lambda tensor, page, first, i=island: pages.append((i, first, page))
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
    made = IslandSession([0, 1], tensors=300)
    made.sessions = (None, None)
    made._drams = [SimpleNamespace(address=0x9000), SimpleNamespace(address=0xA000)]
    head = made._l1_row(1)
    assert (
        head.nbytes == made._cache_bytes() and head[0] == 0xA000 and not head[1:].any()
    )
    made._table_size = 300
    made._entries[299] = SimpleNamespace(entry=lambda: bytes(range(48)))
    rows = made._dram_rows()
    assert rows.shape == (300, 64) and bytes(rows[299, :48]) == bytes(range(48))
