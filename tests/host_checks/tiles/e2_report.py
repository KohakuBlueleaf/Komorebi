"""The build's report says a class of float16 tiles sums in a 16-bit Dst
(`km.tile.zeros(km.fp32)` and `+=` hold float16), and not for bfloat16."""

from stand_in import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_options, summary


class Sum(km.Program):
    @km.handler
    def run(self, x: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.tiles[0:8, 0:1]):
            total += tile
        out.tiles[0:1, 0:1].write(total)


for dtype, sixteen in (("float16", True), ("bfloat16", False)):
    t = {
        "x": Tensor(Pages(2048), (256, 32), dtype, "tile"),
        "out": Tensor(Pages(2048), (32, 32), dtype, "tile"),
    }
    compiled = compile_class(Sum, {}, t)
    said = [x.strip() for x in summary(compiled, "f", {}, None) if "Dst is 16-bit" in x]
    assert compute_options(compiled)["fp32_dest_acc_en"] is not sixteen
    assert bool(said) is sixteen, said
    print(
        f"{dtype}: fp32_dest_acc_en {not sixteen};",
        said[0][:70] if said else "no line",
    )
print("OK")
