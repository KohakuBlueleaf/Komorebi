"""Every other compute path the class compiler generates guards only its packs with
pack's reserve: a push of a copy, a tile function's result, a tile expression or a
body's tile; a fed body; a tile or a sum written; a drained stream and its end. A
body's written-page word guards only the packs (no card)."""

import importlib
import sys
from pathlib import Path

import chain_fixture as cf
import pytest
from class_drained_stream_fixture import Ended, lists
from class_pack_room_fixture import check

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

EXAMPLE = Path(__file__).resolve().parents[5] / "examples" / "22_gate" / "as_class"
sys.path.insert(0, str(EXAMPLE))


class Written(km.Program):
    @km.handler
    def sum(self, s: km.Tensor, t: km.Tensor, tiles: int):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(0, tiles)):
            total += tile
        t.pages(self.rank, 1).write(total)


class Expressed(km.Program):
    @km.handler
    def mix(self, a: km.Tensor, out: km.Tensor, width: int):
        results = km.Stream(out.pages(self.rank * width, width), write=True)
        for x in km.Stream(a.pages(self.rank * width, width)):
            results.push(x + x * x)  # a tile expression in the handler's line


GUARDED = """
tile_regs_acquire();
copy_tile(tile_buffer, tile_index, 0);
tile_regs_commit();
tile_regs_wait();
if (results != 0) {
    pack_tile(0, results_buffer);
}
tile_regs_release();
"""
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


class Guarded(km.Program):
    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        results = km.Stream(out.pages(self.rank * width, width), write=True)
        for tile in km.Stream(a.pages(self.rank * width, width)):
            km.cxx(GUARDED, takes=[tile, results])


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


def compute_of(program):
    return (program.built.folder / "compute.cpp").read_text()


def test_each_other_path_guards_only_the_packs():
    given = cf.tensors()
    seen = check(compute_of(cf.made(cf.Handlers, given)))  # copy, function, body tile
    assert "pack_tile<true>(0, km_out0, km_tile0);" in seen
    assert "pack_tile(0, kOut0);" in check(compute_of(cf.made(Written, given)))
    assert check(compute_of(cf.made(Expressed, given)))
    gc = importlib.import_module("gate_class")
    x = given["x"]
    gate = gc.Gate(a=x, b=x, out=given["y"], l1=1 << 20, log=cf.quiet)
    assert check(compute_of(gate))
    body = cf.made(Guarded, given)
    text = compute_of(body)  # the body's pack under its page word, pack's alone
    assert "const uint32_t results = km_room0 ? (" in text
    assert "if (results != 0) {" in text
    compiled = compile_class(Ended, {}, {"lists": lists()})
    drained = "\n".join(render(prune(compiled.compute["setup"]), 0))
    check(drained)
    assert "if (compute.reserve_back(kEndOut, 1)) {" in drained


def test_a_body_s_page_word_steers_no_section():
    with pytest.raises(Refused, match=r"results is pack's word .* `tile_regs_acquire"):
        cf.made(Steered, cf.tensors())
    with pytest.raises(Refused, match=r"results is pack's word .* `if \(results == 0"):
        cf.made(Left, cf.tensors())
