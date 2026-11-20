"""The corpus's program 5, `Movement.read` and `Movement.copy` of
examples/11_movement/as_class/movement_class.py, as today's class DSL writes it: one
stream both movers carry, dropped or copied. Its tensors are in tiles, not the example's
rows, since an IR view is in tiles: today's make writes the same folder from either."""

import komorebi as km
from komorebi.device.tensor import Tensor

CORES, TILES = 12, 64  # cores, pages a core


class Movement(km.Program):
    """Every core reads its `tiles` pages, from page rank x tiles, through one stream
    both movers carry; READ drops them, COPY writes each block to the output."""

    BLOCK = km.constant(16)  # tiles a block; the make gives the run's (--block)

    @km.handler
    def read(
        self, source: km.Tensor, tiles: int, pattern: km.Pattern, repetitions: int
    ):
        pages = source.pages(self.rank * tiles, tiles)
        for _ in km.Stream(
            pages, block=self.BLOCK, pattern=pattern, repeat=repetitions
        ):
            pass

    @km.handler
    def copy(
        self, source: km.Tensor, output: km.Tensor, tiles: int, pattern: km.Pattern
    ):
        first = self.rank * tiles
        out = km.Stream(
            output.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for tile in km.Stream(
            source.pages(first, tiles), block=self.BLOCK, pattern=pattern
        ):
            out.push(tile)


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: source and output of CORES x TILES bfloat16 pages of 2048 B,
    one tile a page (the example's rows of 2048 B make the same files)."""

    def tensor():
        shape = (32, CORES * TILES * 32)
        return Tensor(Pages(2048, CORES * TILES), shape, "bfloat16", "tile")

    return {"source": tensor(), "output": tensor()}


if __name__ == "__main__":
    made = Movement(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
