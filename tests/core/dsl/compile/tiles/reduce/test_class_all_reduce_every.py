"""km.all_reduce's tile sum written at every rank: every rank packs it into the same
buffer (rank 0 its own, the others the sum cast back), so its write to a float32 tensor
needs no leader's if; km.reduce's sum stays the leader's, and a write in another
format, packed again by compute, is refused by name (no card)."""

import pytest
from test_class_thread_header import Pages, grid, quiet

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError


class Every(km.Program):
    GROUP = km.constant(4)

    @km.handler
    def every(self, s: km.Tensor, t: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            total += tile
        whole = km.all_reduce(total, form="tree", group=self.GROUP)
        t.pages(self.rank, 1).write(whole)


class Leader(km.Program):
    GROUP = km.constant(4)

    @km.handler
    def every(self, s: km.Tensor, t: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            total += tile
        whole = km.reduce(total, form="tree", group=self.GROUP)
        t.pages(self.rank, 1).write(whole)


def out(dtype):
    page = 4096 if dtype == "float32" else 2048
    return Tensor(Pages(page, 16, 0x9000), (32, 16 * 32), dtype, "tile")


def test_every_rank_writes_the_sum_to_a_float32_tensor():
    program = Every(l1=1 << 20, log=quiet, s=grid(16), t=out("float32"))
    movement = (program.built.folder / "movement.cpp").read_text()
    after = movement[movement.index("km::all_reduce_emit(") :]
    write = after.index("km::write_pages(movement, command, T(t), km_rank, 1, ")
    assert "km_rank ==" not in after[:write]  # no leader's if around it
    compute = (program.built.folder / "compute.cpp").read_text()
    assert "km::all_reduce_take(compute, kArrival0" in compute


def test_another_format_at_every_rank_and_km_reduce_s_sum_are_refused_by_name():
    with pytest.raises(ProgramError) as refused:
        Every(l1=1 << 20, log=quiet, s=grid(16), t=out("bfloat16"))
    assert (
        "whole is km.all_reduce's sum, float32 at every rank; packed again in another "
        "format by compute, whose buffer front at a rank but the leader is the part it "
        "sent; write it to a float32 tensor, or under `if self.rank == self.leader:`"
    ) in str(refused.value)
    with pytest.raises(ProgramError, match=r"whole is the reduce's sum, which only"):
        Leader(l1=1 << 20, log=quiet, s=grid(16), t=out("float32"))
