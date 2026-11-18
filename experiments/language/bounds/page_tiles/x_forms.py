"""Which forms of a decode product with X not re-read from DRAM a class can say: each
built on the host (compile_class), its refusal or its buffers a core (README.md, "X
read once"). python experiments/language/bounds/page_tiles/x_forms.py"""

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class

TILE = {"bfloat16": 2048, "bfp8_b": 1088, "bfp4_b": 576}


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(rows, columns, dtype="bfp4_b", k=1):
    """A tile tensor of rows x columns tiles in pages of k tiles."""
    made = Tensor(
        Pages(k * TILE[dtype], rows * columns // k),
        (rows * 32, columns * 32),
        dtype,
        "tile",
    )
    made.page_tiles = k
    return made


GATE = {
    "x": tensor(1, 168, "bfloat16"),
    "w": tensor(168, 672, "bfp4_b", 2),
    "y": tensor(1, 672, "bfloat16"),
}
GATE_T = {
    "x": tensor(1, 168, "bfloat16"),
    "wt": tensor(672, 168, "bfp4_b", 2),
    "y": tensor(1, 672, "bfloat16"),
}
C = {"KS": 84, "W": 14, "CH": 2, "BW": 28}


class CastLeft(km.Program):
    """(ii) X's share cast down the column once, W streamed against it (the cast block
    left)."""

    KS = km.constant(84)
    W = km.constant(14)
    CH = km.constant(2)
    BW = CH * W
    sums: km.L1[km.fp32, 1, W, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 2 * self.KS
        first = col0 + self.rank // 2 * self.W
        down = km.Broadcast(self, along="column", window=1, mover="B")
        if down.sender():
            xs = down.cast(x.tiles[0:1, k0 : k0 + self.KS].read())
        else:
            xs = down.arrive()
        for wb in km.Stream(
            w.tiles[k0 : k0 + self.KS, first : first + self.W],
            block=self.BW,
            pattern="A",
        ):
            self.sums += self.project(xs, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.W].write(whole)


class CastField(CastLeft):
    """(ii) the cast block written into a field compute writes, then field x stream."""

    h: km.L1[km.bf16, 84, 32, 32]

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 2 * self.KS
        first = col0 + self.rank // 2 * self.W
        down = km.Broadcast(self, along="column", window=1, mover="B")
        if down.sender():
            xs = down.cast(x.tiles[0:1, k0 : k0 + self.KS].read())
        else:
            xs = down.arrive()
        self.h.write(xs)
        acc = km.tile.zeros(km.fp32, self.W)
        for wb in km.Stream(
            w.tiles[k0 : k0 + self.KS, first : first + self.W],
            block=self.BW,
            pattern="A",
        ):
            acc += self.project(self.h, wb)
        whole = km.reduce(acc, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.W].write(whole)


class FilledLeft(CastLeft):
    """(ii) X's share filled into a field from DRAM (no cast), field x stream."""

    xf: km.L1[km.bf16, 1, 84, 32, 32]

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 2 * self.KS
        first = col0 + self.rank // 2 * self.W
        self.xf.fill(km.Stream(x.tiles[0:1, k0 : k0 + self.KS]))
        for wb in km.Stream(
            w.tiles[k0 : k0 + self.KS, first : first + self.W],
            block=self.BW,
            pattern="AB",
        ):
            self.sums += self.project(self.xf, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.W].write(whole)


class TurnedRows(CastLeft):
    """(ii') Y^T = W^T X^T: a row of W^T streamed (left) against X's cast block (right),
    one output tile a row; the core's W rows one stream each."""

    @km.handler
    def product(self, x: km.Tensor, wt: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 2 * self.KS
        first = col0 + self.rank // 2 * self.W
        down = km.Broadcast(self, along="column", window=1, mover="A")
        if down.sender():
            xs = down.cast(x.tiles[0:1, k0 : k0 + self.KS].T.read())
        else:
            xs = down.arrive()
        for r in range(self.W):
            acc = km.tile.zeros(km.fp32, 1)
            for wb in km.Stream(
                wt.tiles[first + r : first + r + 1, k0 : k0 + self.KS],
                block=self.CH,
                pattern="B",
            ):
                acc += self.project(wb, xs)
            whole = km.reduce(acc, form="tree", group=2)
            if self.rank == self.leader:
                y.tiles[0:1, first + r : first + r + 1].write(whole)


def trial(cls, tensors):
    """Builds cls for gate's shape: its refusal, or its buffers."""
    try:
        out = compile_class(cls, {}, tensors, constants=C)
    except Refused as error:  # the refusal is the result
        line = str(error).replace("\n", " ")
        print(f"{cls.__name__}: REFUSED {type(error).__name__}: {line[:700]}")
        return
    print(f"{cls.__name__}: BUILDS")
    for b in out.buffers:
        print(
            "  ",
            {
                k: v
                for k, v in vars(b).items()
                if k in ("name", "page_bytes", "pages", "dtype", "line")
            },
        )


class CastLeftDst(CastLeft):
    """(ii) the cast block left, W streamed right, the sums in DST (7 tiles)."""

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 2 * self.KS
        first = col0 + self.rank // 2 * 7
        down = km.Broadcast(self, along="column", window=1, mover="A")
        if down.sender():
            xs = down.cast(x.tiles[0:1, k0 : k0 + self.KS].read())
        else:
            xs = down.arrive()
        acc = km.tile.zeros(km.fp32, 7)
        for wb in km.Stream(
            w.tiles[k0 : k0 + self.KS, first : first + 7], block=14, pattern="B"
        ):
            acc += self.project(xs, wb)
        whole = km.reduce(acc, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + 7].write(whole)


class HeldLeft(CastLeft):
    """(ii) X's share as a held stream (one block, read once a command), W streamed
    against its row: X still from DRAM, once a core."""

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 2 * self.KS
        first = col0 + self.rank // 2 * 7
        acc = km.tile.zeros(km.fp32, 7)
        for xs in km.Stream(x.tiles[0:1, k0 : k0 + self.KS], block=84):
            for wb in km.Stream(
                w.tiles[k0 : k0 + self.KS, first : first + 7], block=14, pattern="B"
            ):
                acc += self.project(xs, wb)
        whole = km.reduce(acc, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + 7].write(whole)


if __name__ == "__main__":
    trial(CastLeft, GATE)
    trial(CastField, GATE)
    trial(FilledLeft, GATE)
    trial(TurnedRows, GATE_T)
    trial(CastLeftDst, dict(GATE, w=tensor(168, 672, "bfp4_b", 7)))
    trial(HeldLeft, dict(GATE, w=tensor(168, 672, "bfp4_b", 7)))
