"""A product with a broadcast's block (no card): waited for before the loop, freed
after it; K row k x block + t of its part (a window, turned where cast so) by tile t, or
a row of K on the left by the stream's K rows; the finish after a stream on its mover;
refused: two parts named whole, a window past the part, a width not the sums'."""

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


PART = 0


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Whole, r"kv is a cast of 2 parts: a product reads one; write kv\[i\]"),
        (Past, r"columns 6 \.\. 9 of a part of 8"),
        (Wide, r"the block's part is 8 columns and acc holds 4"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `acc \+= project.*{refusal}"):
        compute_of(cls)
