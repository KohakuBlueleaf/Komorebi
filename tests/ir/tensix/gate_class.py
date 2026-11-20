"""The corpus's program 4, `Gate.gate` of examples/22_gate/as_class/gate_class.py (its
timed and blocks forms left out), as today's class DSL writes it."""

import komorebi as km
from komorebi.device.tensor import Tensor

CORES, TILES = 12, 64  # cores, tiles a core
TOLERANCE = 1e-6  # the gate's largest relative error against float64 (fp32 values)


@km.tile_function(tolerance=TOLERANCE)
def silu_gate(a, b):
    """The gate of two tiles: what the tile compiler emits for compute."""
    return km.silu(a) * b


class Gate(km.Program):
    """gate: each core's `tiles` pages of a and b from rank x tiles, silu(a) * b into
    the same pages of out, block by block in the command's pattern."""

    BLOCK = km.constant(8)  # tiles a block, as the hand program's kBlock

    @km.handler
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        first = self.rank * tiles
        results = km.Stream(
            out.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for x, g in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=self.BLOCK, pattern=pattern),
        ):
            results.push(silu_gate(x, g))


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: a, b and out of CORES x TILES bfloat16 tiles each."""

    def tensor():
        shape = (32, CORES * TILES * 32)
        return Tensor(Pages(2048, CORES * TILES), shape, "bfloat16", "tile")

    return {"l1": 1 << 20, "a": tensor(), "b": tensor(), "out": tensor()}


if __name__ == "__main__":
    made = Gate(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
