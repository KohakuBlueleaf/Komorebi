"""The corpus's program 7, examples/31_swiglu/swiglu_class.py, as today's class DSL
writes it: `up` (x's row by W1's and W3's columns summed into the fields h and g, their
gate stored to the hidden) and `down` (the hidden's row by W2's columns into out, stored
to y), and its chain `forward`: up on twelve cores, then down on four."""

import komorebi as km
from komorebi.device.tensor import Tensor


@km.tile_function
def gate(h, g):  # the vector unit
    return km.silu(h) * g


class SwiGLU(km.Program):
    """x is [32, D], W1 and W3 are [D, H], W2 is [H, D]; every size in tiles of 32."""

    D = km.constant(32)  # x's and y's width: 1,024
    H = km.constant(96)  # the hidden width: 3,072
    WIDTH = km.constant(8)  # the columns one core makes
    CHUNK = km.constant(2)  # K tiles a block
    W_BLOCK = CHUNK * WIDTH

    h: km.L1[km.fp32, 1, WIDTH, 32, 32]  # x W1, this core's columns: a field in L1
    g: km.L1[km.fp32, 1, WIDTH, 32, 32]  # x W3
    out: km.L1[km.fp32, 1, WIDTH, 32, 32]  # hidden W2

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):  # the matrix unit
        return x @ w

    @km.handler
    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.WIDTH  # this core's columns of the hidden
        for xb, w1b, w3b in zip(
            km.Stream(x.tiles[0:1, 0 : self.D], block=self.CHUNK),
            km.Stream(
                w1.tiles[0 : self.D, first : first + self.WIDTH], block=self.W_BLOCK
            ),
            km.Stream(
                w3.tiles[0 : self.D, first : first + self.WIDTH], block=self.W_BLOCK
            ),
        ):
            self.h += self.project(xb, w1b)
            self.g += self.project(xb, w3b)
        hidden.tiles[0:1, first : first + self.WIDTH].write(gate(self.h, self.g))

    @km.handler
    def down(self, hidden: km.Tensor, w2: km.Tensor, y: km.Tensor):
        first = self.rank * self.WIDTH  # this core's columns of y
        for hb, wb in zip(
            km.Stream(hidden.tiles[0:1, 0 : self.H], block=self.CHUNK),
            km.Stream(
                w2.tiles[0 : self.H, first : first + self.WIDTH], block=self.W_BLOCK
            ),
        ):
            self.out += self.project(hb, wb)
        y.tiles[0:1, first : first + self.WIDTH].write(self.out)

    @km.chain
    def forward(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))
        self.down(hidden, w2, y, cores=(0, 4))


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: x [32, 1024], W1 and W3 [1024, 3072], the hidden [32, 3072]
    and W2 [3072, 1024] of bfloat16, y [32, 1024] of float32, one tile a page."""
    x = Tensor(Pages(2048, 32), (32, 32 * 32), "bfloat16", "tile")
    w1, w3 = (
        Tensor(Pages(2048, 32 * 96), (32 * 32, 96 * 32), "bfloat16", "tile")
        for _ in range(2)
    )
    hidden = Tensor(Pages(2048, 96), (32, 96 * 32), "bfloat16", "tile")
    w2 = Tensor(Pages(2048, 96 * 32), (96 * 32, 32 * 32), "bfloat16", "tile")
    y = Tensor(Pages(4096, 32), (32, 32 * 32), "float32", "tile")
    return {"x": x, "w1": w1, "w3": w3, "hidden": hidden, "w2": w2, "y": y}


if __name__ == "__main__":
    made = SwiGLU(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
