"""A scan's state carried by a tile function in a stream loop, `h = f(x, h)` (no
card): h made before the loop (zeros, or a field's tile) packed into an fp32 carry,
read back as f's operand a tile and packed again after it, pushed or not, back in
DST after the loop; the refusals."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source


@km.tile_function
def gated(a, h, u):
    return a * h + u


@km.tile_function
def both(a, h):
    return a * h, h


class Scan(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def pushed(self, a: km.Tensor, u: km.Tensor, y: km.Tensor, hs: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
            results.push(h)
        hs.pages(0, 1).write(h)

    @km.handler
    def resumed(self, a: km.Tensor, u: km.Tensor, h0: km.Tensor, hs: km.Tensor):
        self.state.fill(km.Stream(h0.pages(0, 1)))
        h = self.state
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
        hs.pages(0, 1).write(h)

    @km.handler
    def left(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
            results.push(h)


def built(cls, dtype="float32"):
    """The class on tiles of `dtype` (a and u; the rest float32)."""
    tensors = {n: Tensor(Pages(4096), (256, 32), "float32", "tile") for n in "yh"}
    tensors |= {
        n: Tensor(Pages(4096 if dtype == "float32" else 2048), (256, 32), dtype, "tile")
        for n in "au"
    }
    tensors["hs"] = tensors["h0"] = tensors["y"]
    tensors["b0"] = Tensor(Pages(2048), (256, 32), "bfloat16", "tile")
    return compile_class(cls, {}, tensors)


def case_of(source, name):
    at = source.index(f"case kHandler{name.capitalize()}")
    end = source.find("case kHandler", at + 1)
    return source[at : end if end > 0 else len(source)]


def test_a_scan_carries_its_state_between_the_tiles():
    source = compute_source(built(Scan))
    case = case_of(source, "pushed")
    loop = case.index("for (uint32_t km_block0")
    carried = "pack_tile(0, kCarryH);"
    assert case.index(carried) < loop  # the start, into the carry before the tiles
    tile = case[case.index(": h = gated(") : case.index("// h, the scan's state, back")]
    assert "km_slot" not in tile  # one call a tile, the next reads its result
    assert "copy_tile(kCarryH, 0, 1);" in tile  # h: gated's second operand
    assert tile.count("MATH((km_tile::gated(0)));") == 1
    out = tile.index("pack_tile<true>(0, km_out0, km_tile0);")
    assert out < tile.index(carried) < tile.index("tile_regs_release();")
    after = case[case.index("// h, the scan's state, back") :]
    assert "copy_tile(kCarryH, 0, 0);" in after  # back in DST 0 for the write
    assert "pack_tile(0, kOut1);" in after


def test_a_scan_started_from_a_field_and_not_pushed():
    case = case_of(compute_source(built(Scan)), "resumed")
    start = case[: case.index("for (uint32_t km_block0")]
    assert "copy_tile(kFieldState, 0, 0);" in start
    assert "cb_pop_front(kFieldState, 1);" in start
    tile = case[case.index(": h = gated(") : case.index("// h, the scan's state, back")]
    assert "pack_tile(0, kCarryH);" in tile  # carried at the tile's end, no push


def test_a_scan_state_left_in_dst_is_released_at_the_end():
    case = case_of(compute_source(built(Scan)), "left")
    after = case[case.index("// h, the scan's state, back") :]
    assert "tile_regs_release();" in after


def test_a_scan_of_bf16_tiles_sets_the_streams_format_again():
    case = case_of(compute_source(built(Scan, "bfloat16")), "pushed")
    tile = case[case.index(": h = gated(") : case.index("// h, the scan's state, back")]
    carry = tile.index("copy_tile(kCarryH, 0, 1);")
    assert tile.index("copy_init(kRead0.mover_a);", carry) < tile.index("km_half1")


class Unset(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)  # noqa: F821
            results.push(h)


class TwoTiles(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32, 2)
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
            results.push(h)


class Twice(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g in km.Stream(a.pages(0, 8)):
            h = gated(g, h, h)
            results.push(h)


class Field(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor, h0: km.Tensor):
        self.state.fill(km.Stream(h0.pages(0, 1)))
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g in km.Stream(a.pages(0, 8)):
            h = gated(g, h, self.state)
            results.push(h)


class Results(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g in km.Stream(a.pages(0, 8)):
            h = both(g, h)
            results.push(h)


class Bf16Field(km.Program):
    half: km.L1[km.bf16, 1, 32, 32]

    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor, b0: km.Tensor):
        self.half.fill(km.Stream(b0.pages(0, 1)))
        h = self.half
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
            results.push(h)


REFUSALS = {
    "unset": Unset,
    "two_tiles": TwoTiles,
    "twice": Twice,
    "field": Field,
    "results": Results,
    "bf16_field": Bf16Field,
}


@pytest.mark.parametrize(
    "handler,why",
    [
        ("unset", "h is read before it is set"),
        ("two_tiles", "one DST tile, passed once, and it is 2 tiles"),
        ("twice", "one DST tile, passed once"),
        ("field", "no field"),
        ("results", "several results"),
        ("bf16_field", "a float32 field of one tile"),
    ],
)
def test_the_refusals(handler, why):
    with pytest.raises(Refused, match=why):
        built(REFUSALS[handler])
