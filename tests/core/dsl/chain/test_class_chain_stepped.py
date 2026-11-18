"""A read of a rectangle with a step (t.tiles[r0:r1:step, c0:c1], a group's rows of a
conv layout): each core's bounding span, from its first row's c0 to its last row's
c1; of an outside tensor nothing to wait for, of one written in the chain the span's
blocks; a stepped write and a span past a range word are refused (no card)."""

import pytest
from chain_fixture import Pages, quiet, run_of

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.device.tensor import Tensor

H, CT, J = 4, 2, 2  # group g is tile rows y CT + g of J tiles


def tiles(rows, columns, address):
    return Tensor(Pages(4096, rows * columns, address), (rows * 32, columns * 32),
                  "float32", "tile")  # fmt: skip


class Conv(km.Program):
    @km.handler
    def rows(self, x: km.Tensor, sq: km.Tensor):
        r = self.rank
        results = km.Stream(sq.tiles[r : r + 1, 0:J], write=True, block=J)
        for xb in km.Stream(x.tiles[r : r + 1, 0:J], block=J):
            results.push(xb)

    @km.handler
    def sums(self, sq: km.Tensor, x: km.Tensor, t: km.Tensor):
        g = self.rank
        acc = km.tile.zeros(km.fp32, J)
        for qb in km.Stream(sq.tiles[g : g + H * CT : CT, 0:J], block=J):
            acc += qb
        for xb in km.Stream(x.tiles[g : g + H * CT : CT, 0:J], block=J):
            acc += xb
        t.tiles[g : g + 1, 0:J].write(acc)

    @km.handler
    def spread(self, sq: km.Tensor, y: km.Tensor):
        g = self.rank
        results = km.Stream(y.tiles[g : g + H * CT : CT, 0:J], write=True, block=J)
        for qb in km.Stream(sq.tiles[g : g + H * CT : CT, 0:J], block=J):
            results.push(qb)

    @km.handler
    def wide(self, t: km.Tensor, w: km.Tensor):
        g = self.rank
        acc = km.tile.zeros(km.fp32, 1)
        for wb in km.Stream(w.tiles[g : g + 200 : CT, 0:1]):
            acc += wb
        t.tiles[g : g + 1, 0:1].write(acc)


X, SQ, Y = (tiles(H * CT, J, 0x10000 * (i + 1)) for i in range(3))
T, W = tiles(CT, J, 0x80000), tiles(200, 16, 0x100000)


def chain_of(*calls, cores):
    program = Conv(l1=1 << 20, log=quiet, x=X, sq=SQ, y=Y, t=T, w=W, chain=True)
    return run_of(program).chain(list(calls), cores=cores)


def test_a_stepped_read_of_a_tensor_the_chain_writes_waits_for_its_span():
    chain = chain_of(
        Conv.rows(X, SQ), Conv.sums(SQ, X, T, cores=(0, CT)), cores=(0, H * CT)
    )
    sums = chain.calls[1]
    span = (H - 1) * CT * J + J  # row g's first tile to row g + (H - 1) CT's last
    assert sums.input == chain.calls[0].output
    assert sums.ranges == (kc.Range(0, J, span),)
    assert kc.prove(chain.calls)


def test_a_stepped_read_of_an_outside_tensor_has_nothing_to_wait_for():
    chain = chain_of(
        Conv.rows(X, SQ), Conv.sums(SQ, X, T, cores=(0, CT)), cores=(0, H * CT)
    )
    (x,) = chain.calls[1].reads
    assert x.earlier is False


def test_a_stepped_write_is_refused_by_name():
    with pytest.raises(kc.ChainRefused, match="a write of a rectangle with a step"):
        chain_of(
            Conv.rows(X, SQ),
            Conv.spread(SQ, Y, cores=(0, CT)),
            cores=(0, H * CT),
        )


def test_a_stepped_span_past_a_range_word_is_refused_by_name():
    with pytest.raises(kc.ChainRefused, match="w's rows with a step span tiles 0 to"):
        chain_of(Conv.rows(X, SQ), Conv.wide(T, W, cores=(0, CT)), cores=(0, H * CT))
