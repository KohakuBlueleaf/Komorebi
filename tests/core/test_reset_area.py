"""A kept reset list writes the overlay area as the load wrote it (no card)."""

from array import array
from types import SimpleNamespace

import numpy as np

from komorebi import abi, overlay
from komorebi.run import NO_LIST, Run, with_area

H = abi.headers()
WORDS = overlay.constants()["kOverlayAreaBytes"] // 4
START = H["KM_CONTROL_BYTES"] // 4


def reset_list():
    """ASSERT, one WRITE of a word, END: whole 64-byte lines, as reset_list gives."""
    words = [H["KMD_RESET_ASSERT"], 0, 0, 0, H["KMD_RESET_WRITE"], 64, 4, 0]
    words += [7, 0, 0, 0, H["KMD_RESET_END"], 0, 0, 0]
    return array("I", words).tobytes()


def entries(blob):
    """The list's entries as (kind, address, bytes, data words), as reset.h reads it."""
    words, at, out = np.frombuffer(blob, "<u4"), 0, []
    while at * 4 + 16 <= len(blob):
        kind, address, size = (int(word) for word in words[at : at + 3])
        at += 4
        data = []
        if kind == H["KMD_RESET_WRITE"]:
            data = words[at : at + size // 4].tolist()
            at += -(-size // 16) * 4
        out.append((kind, address, size, data))
        if kind == H["KMD_RESET_END"]:
            break
    return out


def run_of(rows):
    run = Run.__new__(Run)
    run.h = H
    run.backend = SimpleNamespace(
        control_rows=rows, control_offset=0, control_addr=4096
    )
    return run


def test_area_written_after_assert():
    area = np.arange(WORDS, dtype=np.uint32)
    blob = with_area(reset_list(), 4096 + H["KM_CONTROL_BYTES"], area, H)
    assert len(blob) % 64 == 0
    kinds = entries(blob)
    assert [kind for kind, *_ in kinds] == [
        H["KMD_RESET_ASSERT"],
        H["KMD_RESET_WRITE"],
        H["KMD_RESET_WRITE"],
        H["KMD_RESET_END"],
    ]
    _, address, size, data = kinds[1]
    assert (address, size, data) == (
        4096 + H["KM_CONTROL_BYTES"],
        4 * WORDS,
        list(area),
    )
    assert kinds[2][1:] == (64, 4, [7])


def test_lists_take_their_cores_area():
    rows = np.zeros((3, START + WORDS), np.uint32)
    rows[:, START + 5] = 9
    core_list = bytearray([0, 0, NO_LIST] + [NO_LIST] * 5)
    lists = [reset_list()]
    assert run_of(rows)._area_resets(lists, core_list) is None
    assert entries(lists[0])[1][3][5] == 9


def test_differing_areas_refused():
    rows = np.zeros((2, START + WORDS), np.uint32)
    rows[1, START] = 1
    core_list = bytearray([0, 0])
    message = run_of(rows)._area_resets([reset_list()], core_list)
    assert message == "the overlay areas of reset list 0's cores differ"


def test_no_area_list_unchanged():
    rows = np.zeros((2, START), np.uint32)
    lists = [reset_list()]
    assert run_of(rows)._area_resets(lists, bytearray([0, 0])) is None
    assert lists == [reset_list()]
