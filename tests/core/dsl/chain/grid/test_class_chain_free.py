"""A call that writes its output in tiles of no block form (core (x, y) a tile row y
and columns from x: a grid's tiling) in a chain (no card): taken, its tiles kept;
a call that reads that tensor, or writes any of the same tiles again, refused."""

import pytest
from chain_fixture import quiet, run_of
from class_movers_fixture import nocs
from class_movers_fixture import tensors as mover_tensors

import komorebi as km
from komorebi.chain import calls as kc


@km.tile_function
def project(x, w):
    return x @ w


class Tiled(km.Program):
    """Core (x, y) of 2 x 4 writes Y's tile row y, columns call 8 + x 4 on."""

    @km.handler
    def tiled(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, call: int):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w.tiles[0:8, 0:4].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, 4)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:8], block=2, pattern="A"):
            acc += project(xb, ws)
        left = call * 8 + self.x * 4
        y.tiles[self.y : self.y + 1, left : left + 4].write(acc)

    @km.handler
    def copied(self, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(self.rank, 1), write=True)
        for tile in km.Stream(y.pages(self.rank, 1)):
            results.push(tile)

    @km.chain
    def steps(self, x, w, y):
        self.tiled(x, w, y, 0)


def made():
    program = Tiled(
        l1=1 << 20,
        log=quiet,
        rect=(0, 0, 2, 4),
        chain=True,
        out=mover_tensors()["y"],
        **mover_tensors(),
    )
    program.rectangle["nocs"] = nocs()
    program.args(0)
    return program


def test_calls_writing_other_tiles_of_no_block_form_are_chained():
    program, shared = made(), mover_tensors()
    x, w, y = shared["x"], shared["w"], shared["y"]
    calls = [Tiled.tiled(x, w, y, c) for c in range(2)]
    chain = run_of(program).chain(calls, cores=(0, 8))
    assert [call.output for call in chain.calls] == [None, None]


def test_a_call_writing_the_same_tiles_again_is_refused():
    program, shared = made(), mover_tensors()
    x, w, y = shared["x"], shared["w"], shared["y"]
    with pytest.raises(
        kc.ChainRefused,
        match=r"call 1 \(Tiled\.tiled\): writes tile \(0, 0\) of y .*again, and call 0 "
        r"\(Tiled\.tiled\) writes y .*in tiles of no block form",
    ):
        run_of(program).chain([Tiled.tiled(x, w, y, 0)] * 2, cores=(0, 8))


def test_a_call_reading_tiles_of_no_block_form_is_refused():
    program, shared = made(), mover_tensors()
    x, w, y = shared["x"], shared["w"], shared["y"]
    out = mover_tensors()["y"]
    with pytest.raises(
        kc.ChainRefused,
        match=r"call 1 \(Tiled\.copied\): reads y .*, and call 0 \(Tiled\.tiled\) "
        r"writes y .*in tiles of no block form .*: no mark waits for them",
    ):
        run_of(program).chain(
            [Tiled.tiled(x, w, y, 0), Tiled.copied(y, out)], cores=(0, 8)
        )
