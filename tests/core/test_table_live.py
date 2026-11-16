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
    with pytest.raises(ValueError, match=r"tensors=33: 1 to 32, .* bank address"):
        IslandSession([0, 1], tensors=33)


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
