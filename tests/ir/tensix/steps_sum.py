"""The corpus's program 2, `StepHandlers.sum` of
examples/29_chain/as_class/steps_class.py (its other handlers left out), as today's
class DSL writes it."""

import komorebi as km
from komorebi.device.tensor import Tensor

CORES, TILES = 12, 64  # cores, tiles a core


class StepHandlers(km.Program):
    """Its share of `tiles` tiles summed up a tree to the group's root."""

    @km.handler
    def sum(self, s: km.Tensor, t: km.Tensor, tiles: int):
        part = self.partition(s.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s[part]):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: s of CORES x TILES float32 tiles, t of one."""
    s = Tensor(Pages(4096, CORES * TILES), (32, CORES * TILES * 32), "float32", "tile")
    t = Tensor(Pages(4096, 1), (32, 32), "float32", "tile")
    return {"l1": 1 << 20, "s": s, "t": t}


if __name__ == "__main__":
    made = StepHandlers(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
