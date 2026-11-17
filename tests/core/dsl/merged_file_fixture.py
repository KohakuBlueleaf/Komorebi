"""A class in a file of its own, merged with a class of another file
(test_merged_lines.py): an inline body and a field whose size the build solves."""

import komorebi as km

STEP = """
step(words);
"""


class Elsewhere(km.Program):
    B_TILES = km.symbol()
    b_part: km.L1[km.fp32, B_TILES, 32, 32]

    @km.handler
    def stepped(self, words: km.Tensor):
        km.cxx(STEP, takes=[words], on="mover_b")
