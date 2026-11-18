"""C1 and C3 own check (no card): a reversed or empty rectangle side is refused at
the build when its length is known, and at the submit when its ends are int
parameters; an items range past its tensor is refused at the submit."""

import sys

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size, pages=64):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


PAGE = formats.tile_bytes("bfloat16")


def tensor(pages=64):
    return Tensor(Pages(PAGE, pages), (256, 1024), "bfloat16", "tile")


def tensors():
    return {n: tensor() for n in "wxy"}


class RowsReversed(km.Program):
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[3:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(acc)


class ColumnsReversed(km.Program):
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 5:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(acc)


class NameReversed(km.Program):  # c + 2 : c, c the rank
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor):
        c = self.rank
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, c + 2 : c].write(acc)


class Empty(km.Program):
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(x.tiles[0:1, 1:1], block=1):
            acc += xb
        y.tiles[0:1, 0:1].write(acc)


class Given(km.Program):  # x.tiles[0:1, c0:c1], c0 and c1 parameters
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, c0: int, c1: int):
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(x.tiles[0:1, c0:c1], block=1):
            acc += xb
        y.tiles[0:1, 0:1].write(acc)


class Items(km.Program):  # 64 pages of 2048 B: 2048 items of 64 B
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, first: int, n: int):
        km.Stream(x.items(first, n, size=64)).drain()


class Stepped(km.Program):
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, first: int, n: int):
        km.Stream(x.items(first=first, count=n, size=64, step=4)).drain()


refused = set()
for cls in (RowsReversed, ColumnsReversed, NameReversed, Empty, Given, Items, Stepped):
    try:
        compile_class(cls, {}, tensors())
        kept = getattr(cls.go, "bounds", "none (no such record)")
        print(f"{cls.__name__:16} BUILT; kept for the submit: {kept}")
    except Refused as refusal:
        print(f"{cls.__name__:16} REFUSED: " + str(refusal).split("`: ", 1)[-1][:300])
        refused.add(cls.__name__)

ws = (tensor(), tensor(), tensor())
for cls, calls in (
    (Given, [(0, 2), (3, 1), (2, 2)]),
    (Items, [(0, 2048), (2000, 48), (2000, 49), (2048, 1)]),
    (Stepped, [(0, 512), (3, 512), (0, 513)]),
):
    for values in calls:
        try:
            cls.go.args(*ws, *values)
            print(f"  {cls.__name__}.go{values}: words given")
        except ProgramError as error:
            print(f"  {cls.__name__}.go{values}: ProgramError: {error}"[:330])
            refused.add((cls.__name__, values))

WANTED = {
    "RowsReversed", "ColumnsReversed", "NameReversed", "Empty",
    ("Given", (3, 1)), ("Given", (2, 2)), ("Items", (2000, 49)), ("Items", (2048, 1)),
    ("Stepped", (0, 513)),
}  # fmt: skip
print("PASS" if refused == WANTED else f"FAIL: refused {sorted(map(str, refused))}")
sys.exit(0 if refused == WANTED else 1)
