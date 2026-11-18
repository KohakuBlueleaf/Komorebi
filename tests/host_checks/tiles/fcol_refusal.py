import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stand_in import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class


@km.tile_function(tolerance=1e-5)
def scaled_rows(row, f):
    return row * f.col()


@km.tile_function
def apply_col(x, f):
    return x * f.col()


class RowForm(km.Program):
    scale: km.L1[km.fp32, 1, 32, 64]

    @km.handler
    def run(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for row in km.Stream(t.tiles[0:1, 0:2], block=2, row=2):
            results.push(scaled_rows(row, self.scale))


class StreamForm(km.Program):
    scale: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, t: km.Tensor, sc: km.Tensor, out: km.Tensor):
        self.scale.fill(km.Stream(sc.pages(0, 1)))
        results = km.Stream(out.tiles[0:1, 0:2], write=True)
        for x in km.Stream(t.tiles[0:1, 0:2]):
            results.push(apply_col(x, self.scale))


def tensors():
    return {
        n: Tensor(Pages(4096), (32, 64), "float32", "tile") for n in ("t", "sc", "out")
    }


for cls in (RowForm, StreamForm):
    try:
        compile_class(cls, {}, tensors())
        print(f"{cls.__name__}: BUILT")
    except Exception as e:  # noqa: BLE001
        text = str(e)
        print(f"{cls.__name__}: {type(e).__name__}: {text}")
