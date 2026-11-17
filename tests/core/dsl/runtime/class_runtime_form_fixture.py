"""What test_class_runtime_form.py and other tests share: Copy, Pages, Started,
StartedUp, doubled, made, quiet, tiles."""

import komorebi as km
from komorebi.device.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tiles(count):
    """A bfloat16 tensor of `count` tiles."""
    return Tensor(Pages(2048, count), (count * 32, 32), "bfloat16", "tile")


def quiet(line):
    pass


def made(cls, **given):
    return cls(source=tiles(64), target=tiles(64), l1=1 << 20, log=quiet, **given)


@km.tile_function
def doubled(x):
    return x + x


class Copy(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor, count: int):
        out = km.Stream(target.pages(self.rank * count, count), write=True)
        for tile in km.Stream(source.pages(self.rank * count, count)):
            out.push(doubled(tile))


class Started(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor):
        km.cxx("prepare();", at="start")  # a body at the kernel's start
        km.Stream(source.pages(0, 1)).drain()


class StartedUp(km.Program, startup=km.Startup("ones", "ones", "ones")):
    ones: km.L1[km.bf16, 1, 32, 32] = km.own(push=None)

    @km.handler
    def copy(self, source: km.Tensor, target: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()
