"""Handlers whose W blocks differ share one read pair (layout.pair, a make that shares
pairs): its halves the largest block's, its rings padded at their limits
(KM_RING_PADDED), each handler its own `<pair>Block`; W in pages of 2 tiles, blocks
of 14 and 8, X's 7 and 4 (no card)."""

from test_page_tiles import tensor

import komorebi as km
from komorebi.dsl.compile.layout import padded_rings
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Two(km.Program):
    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def wide(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:56], block=7),
            km.Stream(w.tiles[0:56, 0:2], block=14),
        ):
            acc += self.project(xb, wb)
        y.tiles[0:1, 0:2].write(acc)

    @km.handler
    def narrow(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:56], block=4),
            km.Stream(w.tiles[0:56, 4:6], block=8),
        ):
            acc += self.project(xb, wb)
        y.tiles[0:1, 0:2].write(acc)


def compiled(share):
    tensors = {
        "x": tensor(1, 56, "bfloat16"),
        "w": tensor(56, 240, "bfp8_b", 2),
        "y": tensor(1, 240, "bfloat16"),
    }
    return compile_class(Two, {}, tensors, share_pairs=share)


def text(result, part, handler):
    return "\n".join(render(prune(getattr(result, part)[handler]), 0))


def test_one_w_pair_padded_each_handler_its_block():
    shared = compiled(True)
    (w,) = [b for b in shared.buffers if b.name.startswith("kRead1")]
    assert (w.name, w.pages, w.block, w.padded) == ("kRead1", (56, 56), 14, True)
    assert w.what.startswith("shared by wide, narrow; ")
    assert w.pages[0] % 2 == 0  # whole pages of 2 tiles: every pad is too
    (x,) = [b for b in shared.buffers if b.name.startswith("kRead0")]
    assert (x.name, x.pages, x.block, x.padded) == ("kRead0", (28, 28), 7, True)
    rings = (x.index_a, x.index_b, w.index_a, w.index_b)
    assert padded_rings(shared) == sum(1 << index for index in rings)
    for part in ("movement", "compute"):
        assert "constexpr uint32_t kRead1Block = 8;" in text(shared, part, "narrow")
        assert "kRead1Block = " not in text(shared, part, "wide")
    assert "km::pages_of<2>(W(w))" in text(shared, "movement", "narrow")


def test_without_sharing_each_block_its_own_pair_and_no_padding():
    apart = compiled(False)
    names = sorted(b.name for b in apart.buffers if b.name.startswith("kRead1"))
    assert names == ["kRead1", "kRead1b"]
    assert padded_rings(apart) == 0 and not any(b.padded for b in apart.buffers)
