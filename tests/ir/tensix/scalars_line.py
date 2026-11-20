"""The corpus's program 1, `Scalars.line` of experiments/language/tile_lines/scalars.py
(its other handlers left out), as today's class DSL writes it: the reference the IR
path's files are compared with. Made with no card and no compiler, its tensors
stand-ins of a format and a page; it prints the build folder it wrote."""

import komorebi as km
from komorebi.device.tensor import Tensor

CORES, TILES = 12, 64  # cores, tiles a core


class Scalars(km.Program):
    """Each core: TILES tiles of a from rank x TILES, scaled into those of out."""

    @km.handler
    def line(self, a: km.Tensor, out: km.Tensor, scale: float):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(x * scale)


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, CORES * TILES, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: a of bf16 and out of float32, a tile a page."""
    shape = (32, CORES * TILES * 32)
    a = Tensor(Pages(2048), shape, "bfloat16", "tile")
    out = Tensor(Pages(4096), shape, "float32", "tile")
    return {"l1": 1 << 20, "a": a, "out": out}


if __name__ == "__main__":
    made = Scalars(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
