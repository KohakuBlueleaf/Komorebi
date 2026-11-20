"""The corpus's program 7, examples/31_swiglu/swiglu_class.py, as today's class DSL
writes it, its parts in the order the IR takes them: `SwiGLU.down` here, the product of
a row of the hidden by W2's columns summed into the field `out`, stored to y. Its other
handler (`up`), the fields `h`, `g` and the chain `forward` come with their hashes."""

import komorebi as km
from komorebi.device.tensor import Tensor


class SwiGLU(km.Program):
    """x is [32, D], W1 and W3 are [D, H], W2 is [H, D]; every size in tiles of 32."""

    D = km.constant(32)  # x's and y's width: 1,024
    H = km.constant(96)  # the hidden width: 3,072
    WIDTH = km.constant(8)  # the columns one core makes
    CHUNK = km.constant(2)  # K tiles a block
    W_BLOCK = CHUNK * WIDTH

    out: km.L1[km.fp32, 1, WIDTH, 32, 32]  # hidden W2

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):  # the matrix unit
        return x @ w

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


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def made_with():
    """The make's keywords: the hidden [32, 3072] and W2 [3072, 1024] of bfloat16, y
    [32, 1024] of float32, one tile a page."""
    hidden = Tensor(Pages(2048, 96), (32, 96 * 32), "bfloat16", "tile")
    w2 = Tensor(Pages(2048, 96 * 32), (96 * 32, 32 * 32), "bfloat16", "tile")
    y = Tensor(Pages(4096, 32), (32, 32 * 32), "float32", "tile")
    return {"hidden": hidden, "w2": w2, "y": y}


if __name__ == "__main__":
    made = SwiGLU(log=lambda line: None, **made_with())
    print(f"FOLDER {made.built.folder}")
