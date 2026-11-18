"""Every other compute path the class compiler generates guards only its packs with
pack's reserve: a push of a copy, a tile function's result, a tile expression or a
body's tile; a fed body; a tile or a sum written; a drained stream and its end. A
body's written-page word guards only the packs (no card)."""

import sys
from pathlib import Path

import chain_fixture as cf
import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused

EXAMPLE = Path(__file__).resolve().parents[5] / "examples" / "22_gate" / "as_class"
sys.path.insert(0, str(EXAMPLE))


STEERED = """
if (results != 0) {
    tile_regs_acquire();
    copy_tile(tile_buffer, tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, results_buffer);
    tile_regs_release();
}
"""
LEFT = """
if (results == 0) return;
tile_regs_acquire();
"""


class Steered(km.Program):
    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        results = km.Stream(out.pages(self.rank * width, width), write=True)
        for tile in km.Stream(a.pages(self.rank * width, width)):
            km.cxx(STEERED, takes=[tile, results])


class Left(km.Program):
    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        results = km.Stream(out.pages(self.rank * width, width), write=True)
        for tile in km.Stream(a.pages(self.rank * width, width)):
            km.cxx(LEFT, takes=[tile, results])


def test_a_body_s_page_word_steers_no_section():
    with pytest.raises(Refused, match=r"results is pack's word .* `tile_regs_acquire"):
        cf.made(Steered, cf.tensors())
    with pytest.raises(Refused, match=r"results is pack's word .* `if \(results == 0"):
        cf.made(Left, cf.tensors())
