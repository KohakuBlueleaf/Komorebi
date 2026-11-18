"""The printed cost of a wide row holds its float32 copies into Dst (126 cycles a tile
a pass) and its us at 1,350 MHz: softmax over 4 tiles, 3 passes (1,512 cycles)."""

from stand_in import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import tile_costs


@km.tile_function(tolerance=1e-5)
def soft(row):
    m = km.reduce_max(row, exact=True)
    e = km.exp(row - m.col())
    return e * km.reciprocal(km.reduce_sum(e, exact=True)).col()


class Soft(km.Program):
    @km.handler
    def run(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:4], write=True, block=4)
        for row in km.Stream(t.tiles[0:1, 0:4], block=4, row=4):
            results.push(soft(row))


t = {n: Tensor(Pages(4096), (32, 128), "float32", "tile") for n in ("t", "out")}
lines = tile_costs(compile_class(Soft, {}, t))
assert "1,350 MHz" in lines[0], lines[0]
line = lines[1]
print(line[line.index(" on ") : line.index("; ")])
assert "its float32 copies into DST 1512.0 of them" in line and " us" in line
cycles = float(line.split(" cycles a row")[0].split(", ")[-1])
assert f"{cycles / 1350:.2f} us" in line, line
print("OK")
