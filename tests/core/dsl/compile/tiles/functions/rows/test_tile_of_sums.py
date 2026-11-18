"""A product and a tile function in one handler (no card): `write(act(acc))` of DST
sums runs act's body on each DST tile in place before the pack; `write(act(self.part))`
of a field of sums; a product in the function written of sums is refused at the
line."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

HERE = Path(__file__).resolve().parent


def tensor(shape, address):
    pages = shape[0] * shape[1] // 1024
    page = formats.tile_bytes("bfloat16")
    return Tensor(Pages(page, pages, address), shape, "bfloat16", "tile")


def tensors():
    return {
        "x": tensor((32, 1024), 0x100000),
        "w": tensor((1024, 1536), 0x1000000),
        "y": tensor((32, 1536), 0x2000000),
    }


@km.tile_function
def act(h):
    return km.silu(h)


@km.tile_function
def again(h):
    return h @ h


class InDst(km.Program):
    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        first = self.rank * 4
        acc = km.tile.zeros(km.fp32, 4)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:32], block=2),
            km.Stream(w.tiles[0:32, first : first + 4], block=8),
        ):
            acc += self.project(xb, wb)
        y.pages(first, 4).write(act(acc))


class Again(InDst):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 4)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:32], block=2),
            km.Stream(w.tiles[0:32, 0:4], block=8),
        ):
            acc += self.project(xb, wb)
        y.pages(0, 4).write(again(acc))


def compute_of(cls):
    cls(l1=1 << 20, log=quiet, **tensors())
    folder = sorted(
        (HERE / "build").glob(f"{cls.__name__.lower()}-*"),
        key=lambda p: p.stat().st_mtime,
    )[-1]
    return (folder / "compute.cpp").read_text()


def test_a_product_of_dst_sums_is_refused_at_the_line():
    with pytest.raises(ProgramError) as e:
        compute_of(Again)
    assert "`y.pages(0, 4).write(again(acc))`" in str(e.value)
