"""A product of a stream's tile with a broadcast's block (no card): waited for before
the loop, freed after it, K row k x block + t of its part (a window, transposed where
the cast turned it) by tile t; refused: two parts named whole, a window past the part,
a width not the sums', a stream on the cast's mover after a cast outside a loop."""

from pathlib import Path

import pytest
from broadcast_product_fixture import K, N, project, tensors
from chain_fixture import quiet

import komorebi as km
from komorebi.errors import ProgramError


class Window(km.Program):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:8].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws[:, 3 : 3 + N])
        y.pages(self.rank * N, N).write(acc)


class Parts(km.Program):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            kv = column.cast(wt.tiles[0:N, 0:K].T.read(), w.tiles[0:K, 0:N].read())
        else:
            kv = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, kv[PART])
        y.pages(self.rank * N, N).write(acc)


class Whole(Parts):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            kv = column.cast(wt.tiles[0:N, 0:K].T.read(), w.tiles[0:K, 0:N].read())
        else:
            kv = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, kv)
        y.pages(self.rank * N, N).write(acc)


class Past(Window):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:8].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws[:, 6 : 6 + N])
        y.pages(self.rank * N, N).write(acc)


class Wide(Window):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:8].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)


class Waiting(Window):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:N].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)


PART = 0


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


def test_a_window_is_read_from_its_first_column_between_receive_and_done():
    text = compute_of(Window)
    received = text.index(
        "km::broadcast_receive(compute, kBroadcast0, kBroadcast0Pages)"
    )
    init = text.index("matmul_block_init(kRead0.mover_a, kBroadcast0, false, 4, 1, ")
    call = text.index(
        "matmul_block(km_half0, kBroadcast0, km_tile0, 0 + (km_block0 * kRead0Block + "
        "km_tile0) * 8 + 3, 0, false, 4, 1, kRead0Block);"
    )
    done = text.index("km::broadcast_done(kBroadcast0, kBroadcast0Pages);")
    assert received < init < call < done


def test_a_transposed_part_is_multiplied_transposed_and_a_later_part_from_its_page():
    global PART
    text = compute_of(Parts)
    assert "matmul_block_init(kRead0.mover_a, kBroadcast0, true, 4, 1, " in text
    assert "* kRead0Block + km_tile0) * 4 + 0, 0, true, 4, 1, kRead0Block);" in text
    PART = 1
    try:
        text = compute_of(Parts)
    finally:
        PART = 0
    assert "kBroadcast0, km_tile0, 32 + (km_block0 * kRead0Block + km_tile0) * 4" in (
        text
    )
    assert "matmul_block_init(kRead0.mover_a, kBroadcast0, false, 4, 1, " in text


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Whole, r"kv is a cast of 2 parts: a product reads one; write kv\[i\]"),
        (Past, r"columns 6 \.\. 9 of a part of 8"),
        (Wide, r"the block's part is 8 columns and acc holds 4"),
        (Waiting, r"mover B waits at the cast .*pattern=\"A\""),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `acc \+= project.*{refusal}"):
        compute_of(cls)
