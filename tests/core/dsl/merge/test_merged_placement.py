"""The placement check at submit through a merged program (no card): a command's W
tensor stored otherwise than its product takes it is refused through a class deriving
two classes (its handler, the merged handler of its class), and through km.merge of
made programs (the merged handler, and the class's own), the merge's own check."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype, turned=False):
    made = Tensor(Pages(formats.tile_bytes(dtype)), (64, 128), dtype, "tile")
    made.transposed_tiles = turned
    return made


def quiet(line):
    pass


def classes():
    """Fresh classes each test: a turned product's, and one other."""

    class Turned(km.Program):
        sums: km.L1[km.fp32, 1, 4, 32, 32]

        @km.tile_function(fidelity="HiFi2")
        def turned(x, w):
            return x @ km.srcb(w)

        @km.handler
        def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:2], block=2),
                km.Stream(w.tiles[0:2, 0:4], block=8),
            ):
                self.sums += self.turned(xb, wb)
            y.tiles[0:1, 0:4].write(self.sums)

    class Other(km.Program):
        @km.tile_function
        def doubled(a):
            return a + a

        @km.handler
        def copy(self, x: km.Tensor, y: km.Tensor):
            out = km.Stream(y.pages(0, 2), write=True, block=2)
            for t in km.Stream(x.pages(0, 2), block=2):
                out.push(self.doubled(t))

    return Turned, Other


TENSORS = {"x": tensor("bfloat16"), "w": tensor("bfp8_b", True)}
TENSORS["y"] = tensor("float32")


def refused(handler):
    with pytest.raises(ProgramError, match="tensor w is not stored with its tiles"):
        handler.args(TENSORS["x"], tensor("bfp8_b"), TENSORS["y"])


def test_a_class_deriving_two_classes_checks_the_weight():
    Turned, Other = classes()

    class Layer(Turned, Other):
        pass

    assert Turned.apply.placed == {}
    layer = Layer(l1=1 << 20, log=quiet, **TENSORS)
    assert Layer.apply.args(*TENSORS.values())
    refused(Layer.apply)
    refused(layer.handler(Turned, "apply"))


def test_a_merge_of_made_programs_checks_the_weight():
    Turned, Other = classes()
    turned = Turned(l1=1 << 20, log=quiet, **TENSORS)
    Turned.apply.placed = {}  # as if the class were never made alone
    other = Other(l1=1 << 20, log=quiet, x=TENSORS["x"], y=TENSORS["y"])
    merged = km.merge(turned, other, log=quiet)
    refused(merged.handler(turned, "apply"))
    refused(Turned.apply)
