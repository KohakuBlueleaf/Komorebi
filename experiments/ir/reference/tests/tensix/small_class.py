"""The corpus's program 6, `Small.read` of projects/dense/small_class.py, as today's
class DSL writes it: Y's tile (r, n) the sum along K of a row block of X by a column
block of W, both streamed. The class's field and its other handlers (`load`, `held`)
are left out: a field is not written in an IR module yet."""

import komorebi as km
from komorebi.device.tensor import Tensor

CALLS = 32  # calls whose Ys the output holds, one after the other


class Small(km.Program):
    """Y [32 M, 32 N] = X [32 M, 32 K] W [32 K, 32 N]; core r N + n makes Y's tile
    (r, n), written to Y's page c M N + r N + n by call c."""

    M_TILES = km.constant(4)
    K_TILES = km.constant(28)
    N_TILES = km.constant(4)
    CHUNK = km.constant(4)  # K tiles a block

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def read(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, call: int):
        r = self.rank // self.N_TILES
        n = self.rank % self.N_TILES
        acc = km.tile.zeros(km.fp32, 1)
        for xb, wb in zip(
            km.Stream(x.tiles[r : r + 1, 0 : self.K_TILES], block=self.CHUNK),
            km.Stream(w.tiles[0 : self.K_TILES, n : n + 1], block=self.CHUNK),
        ):
            acc += self.project(xb, wb)
        y.pages(call * self.M_TILES * self.N_TILES + self.rank, 1).write(acc)


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: X [128, 896] and W [896, 128] of bfloat16, Y of CALLS Ys
    [128, 128] of float32, one tile a page."""
    x = Tensor(Pages(2048, 4 * 28), (128, 896), "bfloat16", "tile")
    w = Tensor(Pages(2048, 28 * 4), (896, 128), "bfloat16", "tile")
    y = Tensor(Pages(4096, CALLS * 16), (CALLS * 128, 128), "float32", "tile")
    return {"x": x, "w": w, "y": y}


if __name__ == "__main__":
    made = Small(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
