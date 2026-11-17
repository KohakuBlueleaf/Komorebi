"""A call that writes block g of its output at tiles base + g x pitch (a padded
tensor's interior): its record counts blocks of its width, and each read of the
tensor is mapped to those blocks (a padding tile: none), one range when it goes up
by a step, else the covering range; a direct input of it is refused (no card)."""

import pytest
from chain_fixture import Pages, quiet, run_of

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.device.tensor import Tensor

G, J = 8, 2  # groups of one; J interior tile columns, J + 2 in the padded tensor


def tiles(rows, columns, address):
    return Tensor(Pages(4096, rows * columns, address), (rows * 32, columns * 32),
                  "float32", "tile")  # fmt: skip


class Pad(km.Program):
    @km.handler
    def inner(self, x: km.Tensor, y: km.Tensor):
        r = self.rank
        k = r + 1
        for j in range(1, J + 1):
            i = j - 1  # a name of the loop's value (rectangle_of follows it)
            acc = km.tile.zeros(km.fp32, 1)
            for tile in km.Stream(x.tiles[r : r + 1, i : i + 1]):
                acc += tile
            y.tiles[k : k + 1, j : j + 1].write(acc)

    @km.handler
    def halo(self, y: km.Tensor, z: km.Tensor):
        r = self.rank
        acc = km.tile.zeros(km.fp32, 1)
        for tile in km.Stream(y.tiles[r : r + 3, 1:2]):
            acc += tile
        z.tiles[r : r + 1, 0:1].write(acc)

    @km.handler
    def ahead(self, y: km.Tensor, z: km.Tensor):
        r = self.rank
        k = r + 1
        acc = km.tile.zeros(km.fp32, 1)
        for tile in km.Stream(y.tiles[k : k + 2, 1:2]):
            acc += tile
        z.tiles[r : r + 1, 0:1].write(acc)

    @km.handler(direct=("y",))
    def taken(self, y: km.Tensor, z: km.Tensor):
        r = self.rank
        acc = km.tile.zeros(km.fp32, 1)
        for tile in km.Stream(y.tiles[r : r + 1, 1:2]):
            acc += tile
        z.tiles[r : r + 1, 0:1].write(acc)


X, Y, Z = tiles(G, J, 0x10000), tiles(G + 2, J + 2, 0x80000), tiles(G, 1, 0x100000)


def chain_of(*calls):
    program = Pad(l1=1 << 20, log=quiet, x=X, y=Y, z=Z, chain=True)
    return run_of(program).chain(list(calls), cores=(0, G))


def test_an_interior_s_blocks_and_a_halo_s_covering_range():
    chain = chain_of(Pad.inner(X, Y), Pad.halo(Y, Z))
    inner, halo = chain.calls
    assert (inner.output, inner.width) == ("y (0x80000)", J)  # block coordinates
    assert halo.input == inner.output
    assert halo.ranges == (kc.Range(0, 0, G * J - 1),)  # every block: r - 1 is below
    assert kc.prove(chain.calls)


def test_a_read_of_the_blocks_ahead_is_one_range():
    chain = chain_of(Pad.inner(X, Y), Pad.ahead(Y, Z))
    assert chain.calls[1].ranges == (kc.Range(0, J, J + 1),)  # tiles 2r and 2r + 2
    assert kc.prove(chain.calls)


def test_a_direct_input_of_an_interior_is_refused():
    with pytest.raises(kc.ChainRefused, match="at a pitch .* not built"):
        chain_of(Pad.inner(X, Y), Pad.taken(km.direct(Y), Z))
