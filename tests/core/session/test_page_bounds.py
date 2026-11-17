"""KM_GUARD's page bounds on the host: a class's ranges checked at submit, the list a
card with no tensor table reads, and the pages a table entry carries."""

import gc
import struct
from types import SimpleNamespace

import numpy as np

import komorebi as km
from komorebi import abi
from komorebi.device import extents
from komorebi.islands import graph as islands
from komorebi.session import page_bounds, state
from komorebi.session.session import Session

K = 2


class Once(km.Program):
    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(self.rank * K, K)):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            out.pages(0, 1).write(whole)


class Loop(km.Program):
    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor, turns: int):
        for i in range(turns):
            total = km.tile.zeros(km.fp32)
            for tile in km.Stream(x.pages((self.rank * turns + i) * K, K)):
                total += tile
            whole = km.reduce(total, form="tree")
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


class Guarded(km.Program):
    @km.handler
    def h(self, x: km.Tensor, n: int):
        if self.rank < n:  # under an if: the card's check
            x.pages(self.rank, 1).write(km.tile.zeros(km.fp32))


def tensor(pages):
    return SimpleNamespace(pages=pages, shape=(pages * 32, 32))


def test_a_class_range_past_its_tensor_is_named_with_its_line_rank_and_turn():
    assert page_bounds.problem(Once.h, (tensor(16), tensor(1)), (0, 8), Once) is None
    said = page_bounds.problem(Once.h, (tensor(15), tensor(1)), (0, 8), Once)
    assert "pages 14 to 15 of x (its pages: 0 to 14)" in said and "rank 7" in said
    assert page_bounds.problem(Loop.h, (tensor(64), tensor(4), 4), (0, 8), Loop) is None
    said = page_bounds.problem(Loop.h, (tensor(64), tensor(3), 4), (0, 8), Loop)
    assert "pages 3 to 3 of out" in said and "rank 0 at i 3" in said
    said = page_bounds.problem(Loop.h, (tensor(63), tensor(4), 4), (0, 8), Loop)
    assert "pages 62 to 63 of x" in said and "rank 7 at i 3" in said


def test_a_range_under_an_if_is_left_to_the_card():
    assert page_bounds.problem(Guarded.h, (tensor(1), 8), (0, 8), Guarded) is None


def test_a_checked_command_is_refused_once_and_remembered_when_right():
    checked = page_bounds.Checked()
    checked.check(Once.h, (tensor(16), tensor(1)), [1, 2], (0, 8), Once)
    try:
        checked.check(Once.h, (tensor(15), tensor(1)), [1, 3], (0, 8), Once)
    except ValueError as error:
        assert "refused before it runs" in str(error)
    else:
        raise AssertionError("not refused")


def dram(address, pages, page_bytes=2048):
    return SimpleNamespace(
        address=address,
        pages=pages,
        aligned_page_bytes=page_bytes,
        in_dram=True,
        shape=(pages * 32, 32),
        dtype="float32",
    )


def test_the_list_is_sorted_by_address_in_lines_and_holds_at_most_its_count():
    words, kept, left = extents.listed(
        [dram(0x3000, 4), dram(0x1000, 10), dram(0x2000, 1, 96)], 8
    )
    assert words.tolist() == [
        2,
        0x1000,
        10 * 2048 // 64,
        0x3000,
        4 * 2048 // 64,
        0,
        0,
        0,
    ]
    assert len(kept) == 2 and left == []  # 96 bytes: no whole lines, unchecked
    words, kept, left = extents.listed([dram(0x1000 * k, 1) for k in range(1, 6)], 3)
    assert words[0] == 3 and len(left) == 2 and words.nbytes % 16 == 0
    assert "2 DRAM tensors past the list's 3" in extents.unchecked_says(left, 3)


def test_a_freed_tensor_leaves_the_session_s_list():
    class Held:  # a weak reference's target, as a session's tensor is
        in_dram, freed = True, False

    kept, freed = Held(), Held()
    freed.freed = True  # Tensor.free(): its buffer would raise at .address
    registry = extents.Registry()
    for tensor in (kept, freed):
        registry.add(tensor)
    assert registry.live() == [kept]


class Made:  # a session's DRAM tensor, weakly held by the registry
    in_dram, freed, pages, aligned_page_bytes = True, False, 4, 2048

    def __init__(self, address):
        self.address = address


def test_a_tensor_only_a_cycle_holds_is_not_listed_before_python_collects():
    """A class's room and the load's table count the session's list: a tensor that
    only a reference cycle holds is left out whenever Python would collect it."""
    registry, kept = extents.Registry(), Made(0x1000)
    registry.add(kept)
    holder = SimpleNamespace(page_bounds=extents.MOST, l1_held=list, _dram=registry)
    gc.disable()
    try:
        for address in (0x2000, 0x3000):
            dropped = Made(address)
            dropped.cycle = dropped  # unreachable once dropped, not yet collected
            registry.add(dropped)
        del dropped
        assert len(registry._tensors) == 3
        pending = Session.page_bounds_list(holder, [SimpleNamespace()])
        gc.collect()
        collected = Session.page_bounds_list(holder, [SimpleNamespace()])
    finally:
        gc.enable()
    assert pending[1] == collected[1] == [kept]
    assert pending[0].tolist() == collected[0].tolist()


def test_a_table_entry_carries_its_pages_beside_the_copy_bit():
    placement = islands.Placement(parts=((0, 2), (2, 2)), steps=4, inner=3, outer=5)
    entry = islands.tensor_entry(placement, 2048, [0x1000, 0x2000])
    head = struct.unpack("<4I", entry[:16])
    assert head[3] & 3 == 0 and head[3] >> 2 == placement.pages == 60
    copy = islands.Placement(parts=((0, 1), (0, 1)), copy=True, steps=1, inner=7)
    head = np.frombuffer(islands.tensor_entry(copy, 2048, [1, 2])[:16], np.uint32)
    assert head[3] & 3 == 1 and head[3] >> 2 == 7


def test_a_give_up_past_a_tensor_reads_as_its_page_in_the_live_words():
    headers = abi.headers()
    words = [0] * (headers["KM_LIVE_READ"] // 4)
    words[headers["kLvLine"]] = 1 << 31 | 7
    out = state.live(words, headers)
    assert out["page"] == 7 and out["line"] is None
    words[headers["kLvLine"]] = 1 << 16 | 42  # core.wait's give-up: kernel, line
    assert state.live(words, headers)["line"] == {"kernel": state.ROLES[1], "line": 42}


def test_a_give_up_past_a_tensor_is_named_in_the_gave_up_words():
    headers = abi.headers()
    word = headers["kGaveBadItem"] | headers["kWherePageRange"] << 8 | 3 << 16
    out = state.gave_up(word, 9, headers)
    assert (out["code"], out["where"], out["slot"]) == ("bad_item", "pagerange", 3)
