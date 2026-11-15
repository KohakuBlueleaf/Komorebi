"""A class's broadcasts a mover (no card): a core holds one broadcast's arguments a
mover, read by every handler, so two handlers' broadcasts on one mover are refused at
the make, naming both handlers, the mover and the lines; on two movers each is its
own (two buffers, two numbers: two handlers' arguments are never one set)."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

K, N = 8, 4


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, K * 32), 0x100000),
        "w": made((K * 32, 1024), 0x1000000),
        "y": made((128, 512), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


class OneMover(km.Program):
    @km.handler
    def first(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:N].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)

    @km.handler
    def second(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, N : N + N].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)


class TwoMovers(km.Program):
    @km.handler
    def first(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:N].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)

    @km.handler
    def second(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w.tiles[0:K, N : N + N].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="B"):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)


def made(cls):
    return cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())


def test_two_handlers_broadcasts_on_one_mover_are_refused_by_name():
    with pytest.raises(
        ProgramError,
        match=(
            r"second, line \d+, `column = km\.Broadcast.*`: first runs a broadcast "
            r"on mover B \(line \d+, `column = km\.Broadcast.*`\); .*so both would "
            r"run with the last one's; give one of them mover=\"A\""
        ),
    ):
        made(OneMover)


def test_on_two_movers_each_broadcast_is_its_own():
    program = made(TwoMovers)
    assert set(program.broadcast_lines) == {"first", "second"}
    header = (Path(program.built.folder) / "twomovers.h").read_text()
    assert "kBroadcast0 = 8;" in header and "kBroadcast1 = 9;" in header
