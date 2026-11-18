"""C2 own check (no card): a put of items of a size not a multiple of 16 is refused
at the build unless at= and the source's item are ints that keep the 16-byte phase."""

import sys

from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors(size):
    page = 1920 if size in (12, 20) else formats.tile_bytes("bfloat16")  # 12, 20 divide
    return {n: Tensor(Pages(page), (256, 1024), "bfloat16", "tile") for n in "x"}


import c2_classes  # noqa: E402

refused = []
for cls, size, at, source in c2_classes.CASES:
    try:
        compile_class(cls, {}, tensors(size))
        outcome = "BUILT"
    except Refused as refusal:
        outcome = "REFUSED: " + str(refusal).split("`: ", 1)[-1][:330]
        refused.append((size, at, source))
    print(f"size {size:3} at={at:18} source {source:10} {outcome}")

WANTED = [
    (4, "self.rank", "self.st"), (8, "self.rank", "self.st"),
    (12, "self.rank", "self.st"), (20, "self.rank", "self.st"),
    (4, "3", "self.st"), (12, "5", "self.st"),
]  # fmt: skip
print("PASS" if refused == WANTED else f"FAIL: refused {refused}")
sys.exit(0 if refused == WANTED else 1)
