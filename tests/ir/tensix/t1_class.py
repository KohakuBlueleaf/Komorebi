"""The first lesson (scale) as today's class DSL writes it: the reference the Tensix
backend's files are compared with. Made with no card and no compiler, its tensors
stand-ins of a format and a page; it prints the build folder it wrote."""

import komorebi as km
from komorebi.device.tensor import Tensor


@km.tile_function
def times(a, k):
    return a * k


class Scale(km.Program):
    """Each core scales its part of `tiles` tiles of x from `first` into y."""

    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, first: int, tiles: int, k: float):
        src = self.partition(x.pages(first, tiles))
        out = km.Stream(y[src], write=True, block=1)
        for xb in km.Stream(x[src], block=1):
            out.push(times(xb, k))


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 2048, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor():
    return Tensor(Pages(2048), (2048, 32), "bfloat16", "tile")


if __name__ == "__main__":
    made = Scale(log=lambda line: None, l1=1 << 20, x=tensor(), y=tensor())
    print(f"FOLDER {made.built.folder}")
    # Each kernel the program hands the runtime: its role, source, processor as the
    # backend's describe numbers it, compile arguments, defines and options.
    for number, role in enumerate(("reader", "writer", "compute")):
        options = made.compute if role == "compute" else {}
        print(
            "KERNEL",
            role,
            made.sources()[role].name,
            number,
            "args=" + ",".join(str(word) for word in made.kernel_compile_args(role)),
            "defines=" + ",".join(f"{k}:{v}" for k, v in sorted(made.defines.items())),
            "options="
            + ",".join(
                f"{k}:{str(v).lower() if isinstance(v, bool) else v}"
                for k, v in sorted(options.items())
            ),
        )
