"""A cast of a tensor in DRAM pages of k tiles (page_tiles): its read a page of k
tiles through km::PagesOf, the slot a page a tile as at k = 1, the rest the same
text; refused by name: a transposed cast, a width or first column not whole pages,
parts of two page sizes (no card)."""

import pytest
from page_tiles_fixture import tensor

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Owned(km.Program):
    KC = km.constant(2)
    OW = km.constant(6)

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, n0: int):
        own = km.Broadcast(self, along="row", length=1, window=2)
        left = n0 + self.rank * self.OW
        acc = km.tile.zeros(km.fp32, self.OW)
        for chunk in range(4):
            k0 = chunk * self.KC
            if own.sender():
                ws = own.cast(w.tiles[k0 : k0 + self.KC, left : left + self.OW].read())
            else:
                ws = own.arrive()
            for xb in km.Stream(x.tiles[0:1, k0 : k0 + self.KC], block=2, pattern="B"):
                acc += self.project(xb, ws)
        y.tiles[0:1, left : left + self.OW].write(acc)


class Transposed(Owned):
    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, n0: int):
        own = km.Broadcast(self, along="row", length=1, window=2)
        acc = km.tile.zeros(km.fp32, self.OW)
        if own.sender():  # noqa: SIM108 (the compiler's form)
            ws = own.cast(w.tiles[0:6, 0:2].T.read())
        else:
            ws = own.arrive()
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2, pattern="B"):
            acc += self.project(xb, ws)
        y.tiles[0:1, 0:6].write(acc)


class Two(Owned):
    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, v: km.Tensor, y: km.Tensor, n0: int):
        own = km.Broadcast(self, along="row", length=1, window=2)
        acc = km.tile.zeros(km.fp32, self.OW)
        if own.sender():
            ws = own.cast(w.tiles[0:2, 0:6].read(), v.tiles[0:2, 0:6].read())
        else:
            ws = own.arrive()
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2, pattern="B"):
            acc += self.project(xb, ws[0][:, 0 : 0 + 6])
        y.tiles[0:1, 0:6].write(acc)


def compiled(cls, k, ow=6, more=None):
    tensors = {
        "x": tensor(1, 8, "bfloat16"),
        "w": tensor(8, 240, "bfp8_b", k),
        "y": tensor(1, 240, "bfloat16"),
        **(more or {}),
    }
    return compile_class(cls, {}, tensors, constants={"KC": 2, "OW": ow})


def movement(result):
    return "\n".join(render(prune(result.movement["owned"]), 0))


def test_k_reads_the_cast_a_page_of_k_tiles_and_keeps_the_rest_as_at_one_tile():
    one, two = compiled(Owned, 1), compiled(Owned, 2)
    text1, text2 = movement(one), movement(two)
    assert "km::PagesOf<" not in text1
    assert "km::read_pages(movement, command, km::PagesOf<2, W>{W(w)}, " in text2
    assert text2.replace("km::PagesOf<2, W>{W(w)}", "W(w)") == text1
    assert two.page_tiles[("owned", "w")] == 2
    slots = [(b.name, b.page_bytes, b.pages) for b in one.buffers]
    assert [(b.name, b.page_bytes, b.pages) for b in two.buffers] == slots
    compute = ["\n".join(render(prune(c.compute["owned"]), 0)) for c in (one, two)]
    assert compute[0] == compute[1]
    assert movement(compiled(Owned, 3)).count("km::PagesOf<3, W>{W(w)}") == 1


def test_a_transposed_cast_in_pages_of_k_tiles_is_refused_by_name():
    with pytest.raises(Refused, match="w is in pages of 2 tiles: a transposed cast"):
        compiled(Transposed, 2)
    compiled(Transposed, 1)


def test_a_width_not_whole_pages_is_refused():
    with pytest.raises(Refused, match=r"its width 6 over pages of 4 tiles"):
        compiled(Owned, 4)


def test_a_cast_of_parts_in_pages_of_two_sizes_is_refused():
    with pytest.raises(Refused, match=r"its tensors' pages are of \[1, 2\] tiles"):
        compiled(Two, 2, more={"v": tensor(8, 240, "bfp8_b", 1)})
    compiled(Two, 2, more={"v": tensor(8, 240, "bfp8_b", 2)})
