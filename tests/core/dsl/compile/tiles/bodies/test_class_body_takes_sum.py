"""A body at a group's root takes the reduce's sum of a field: the field's buffer and
first tile bound, waited for before the body and freed after it, as the hand's
operation on the sum; taken elsewhere, on another CPU or twice, refused (no card)."""

import pytest
from chain_fixture import quiet
from class_reduce_arrival_fixture import tiles

import komorebi as km
from komorebi.errors import ProgramError

ADD = "add_residual(whole_buffer, whole_index, residual_buffer, out_buffer);"


class Added(km.Program):
    """A K split of two cores a group, the root adding a residual to the sum."""

    sums: km.L1[km.fp32, 1, 8, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, r: km.Tensor, y: km.Tensor):
        k0 = self.rank % 2 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            residual = km.Stream(r.tiles[0:1, 0:8], block=8)
            out = km.Stream(y.tiles[0:1, 0:8], write=True, block=8)
            km.cxx(ADD, takes=[whole, residual, out])


def made(cls):
    return cls(
        l1=1 << 20,
        log=quiet,
        x=tiles(1, 4, 0x10000),
        w=tiles(4, 8, 0x20000),
        r=tiles(1, 8, 0x80000),
        y=tiles(1, 8, 0x90000),
    )


class Outside(Added):
    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, r: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        km.cxx(ADD, takes=[whole])


class OnPack(Added):
    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, r: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            km.cxx(ADD, takes=[whole], on="pack")


class Twice(Added):
    @km.handler
    def add(self, x: km.Tensor, w: km.Tensor, r: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            km.cxx(ADD, takes=[whole])
            km.cxx(ADD, takes=[whole])


@pytest.mark.parametrize(
    "cls, says",
    [
        (Outside, "whole is the reduce's sum of self.sums, which only the leader"),
        (OnPack, "a body of compute's three takes it"),
        (Twice, "which only the leader holds, once"),
    ],
)
def test_a_sum_taken_elsewhere_on_one_cpu_or_twice_is_refused(cls, says):
    with pytest.raises(ProgramError) as error:
        made(cls)
    assert says in str(error.value) and "line " in str(error.value)
