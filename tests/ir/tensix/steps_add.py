"""The corpus's program 3, `StepHandlers.add` of
examples/29_chain/as_class/steps_class.py (its other handlers left out), as today's
class DSL writes it."""

import komorebi as km
from komorebi.device.tensor import Tensor

CORES, WIDTH = 12, 64  # cores, tiles a core


@km.tile_function
def plus(a, b):
    return a + b


class StepHandlers(km.Program):
    """Each core's `width` tiles from rank x width of a and b, added into out."""

    @km.handler
    def add(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first, width))
        ):
            results.push(plus(x, y))


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
    made = StepHandlers(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
