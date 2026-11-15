"""A class's broadcasts (no card): two handlers' on one mover each read their own
arguments, the second's after the first's (kArgBroadcast + kBroadcastArgs), each a
buffer and number; on two movers each its mover's first; a third broadcast, past a
core's KM_BROADCASTS areas, refused at the make naming the class's broadcasts."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import abi, formats, layout
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


class Three(OneMover):
    @km.handler
    def third(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        row = km.Broadcast(self, along="row", window=1)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            ws = row.cast(w.tiles[0:K, 0:N].read())
        else:
            ws = row.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="B"):
            acc += project(xb, ws)
        y.pages(self.rank * N, N).write(acc)


def made(cls):
    return cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())


def nocs():
    """A 2 x 4 grid's NoC words (made up: x + 1, y + 2), for the arguments' layout."""
    order = layout.core_order((2, 4))
    words = [(x + 1) | (y + 2) << 16 for x, y in order]
    return layout.nocs((2, 4), words, words)


def test_two_handlers_broadcasts_on_one_mover_read_their_own_arguments():
    program = made(OneMover)
    movement = (Path(program.built.folder) / "movement.cpp").read_text()
    assert "km::Broadcast column(movement, command, kArgBroadcast);" in movement
    assert (
        "km::Broadcast column(movement, command, kArgBroadcast + km::kBroadcastArgs "
        "* 1);"
    ) in movement
    program.rectangle["nocs"] = nocs()  # what Session.load hands over
    fields = len(abi.struct_fields(abi.headers().device[-1], "BroadcastArgs"))
    mover_a, mover_b, common = program.args(0)
    assert len(mover_a) == len(common) and len(mover_b) == len(common) + 2 * fields
    first, second = mover_b[len(common) :][:fields], mover_b[len(common) + fields :]
    assert first[0] == 0 and second[0] == 1  # each its own number (control area)
    assert first[8] != second[8]  # and its own buffer


def test_a_third_broadcast_is_refused_naming_the_others():
    with pytest.raises(
        ProgramError,
        match=(
            r"third, line \d+, `row = km\.Broadcast.*`: the class runs 2 broadcasts "
            r"already \(first, line \d+, .*; second, line \d+, .*\); a core has 2 "
            r"broadcast areas"
        ),
    ):
        made(Three)


def test_on_two_movers_each_broadcast_is_its_own():
    program = made(TwoMovers)
    assert set(program.broadcast_lines) == {"first", "second"}
    header = (Path(program.built.folder) / "twomovers.h").read_text()
    assert "kBroadcast0 = 8;" in header and "kBroadcast1 = 9;" in header
