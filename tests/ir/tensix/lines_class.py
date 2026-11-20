"""A tile function whose calls of `km.exp` sit on a continuation line and in a helper
it calls: the line today's listing names for each (`caller_line`, the call's frame),
for the origin's place of each call (docs/ir/frontends.md)."""

import komorebi as km
from komorebi.device.tensor import Tensor

CORES, WIDTH = 12, 64  # cores, tiles a core


def grown(x):
    """A helper the tile function calls."""
    return km.exp(x) * 2.0


# fmt: off
@km.tile_function
def spread(a, b):
    return (a
            + km.exp(b)
            + km.exp(
                a)
            + grown(a))
# fmt: on


class Lines(km.Program):
    """Each core's `width` tiles from rank x width of a and b, spread into out."""

    @km.handler
    def spread(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first, width))
        ):
            results.push(spread(x, y))


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: a, b and out of CORES x WIDTH float32 tiles each."""

    def tensor():
        shape = (32, CORES * WIDTH * 32)
        return Tensor(Pages(4096, CORES * WIDTH), shape, "float32", "tile")

    return {"l1": 1 << 20, "a": tensor(), "b": tensor(), "out": tensor()}


if __name__ == "__main__":
    made = Lines(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
