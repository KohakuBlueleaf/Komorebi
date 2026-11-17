"""A word made on one mover and taken on the other: the giver's page of the hand-over
buffer, the taker's wait and free; and the refusals (no card)."""

import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.nodes import prune, render


class Counted(km.Program):
    @km.handler
    def count(self, n: int):
        pages = km.cxx("pages = n;", takes=[n], gives=int, on="mover_a")
        cycles = km.cxx("cycles = 7;", gives=int, on="mover_a")
        km.cxx("report(pages, cycles);", takes=[pages, cycles], on="mover_b")


def test_the_giver_pushes_a_page_of_its_words_and_the_taker_reads_it():
    compiled = compile_class(Counted, {}, {})
    movement = "\n".join(render(prune(compiled.movement["count"]), 0))
    # mover A: both words in one page, pushed
    giving = movement.index("if (!movement.reserve(kHandover, 1)) {")
    assert "km::l1(km_handed17)[0] = pages;\nkm::l1(km_handed17)[1] = cycles;" in (
        movement
    )
    # mover B: waits for the page, the body's words from it, the page freed after
    taking = movement.index("if (!movement.wait_front(kHandover, 1)) {")
    assert giving < taking
    assert "const uint32_t cycles = km::l1(km_handed17)[1];" in movement
    assert movement.index("report(pages, cycles);") < movement.index(
        "cb_pop_front(kHandover, 1);"
    )
    assert [b.page_bytes for b in compiled.buffers if b.name == "kHandover"] == [64]
    assert any("from mover A to mover B" in h for h in compiled.handovers)


class Many(km.Program):
    @km.handler
    def count(self, n: int):
        w0 = km.cxx("w0 = 0;", gives=int, on="mover_a")
        w1 = km.cxx("w1 = 1;", gives=int, on="mover_a")
        w2 = km.cxx("w2 = 2;", gives=int, on="mover_a")
        w3 = km.cxx("w3 = 3;", gives=int, on="mover_a")
        w4 = km.cxx("w4 = 4;", gives=int, on="mover_a")
        w5 = km.cxx("w5 = 5;", gives=int, on="mover_a")
        w6 = km.cxx("w6 = 6;", gives=int, on="mover_a")
        w7 = km.cxx("w7 = 7;", gives=int, on="mover_a")
        w8 = km.cxx("w8 = 8;", gives=int, on="mover_a")
        w9 = km.cxx("w9 = 9;", gives=int, on="mover_a")
        wa = km.cxx("wa = 10;", gives=int, on="mover_a")
        wb = km.cxx("wb = 11;", gives=int, on="mover_a")
        wc = km.cxx("wc = 12;", gives=int, on="mover_a")
        wd = km.cxx("wd = 13;", gives=int, on="mover_a")
        we = km.cxx("we = 14;", gives=int, on="mover_a")
        wf = km.cxx("wf = 15;", gives=int, on="mover_a")
        wg = km.cxx("wg = 16;", gives=int, on="mover_a")
        km.cxx(
            "use();",
            takes=[w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, wg],
            on="mover_b",
        )


class Unstated(km.Program):
    @km.handler
    def count(self, n: int):
        pages = km.cxx("pages = n;", takes=[n], gives=int, on="mover_a")
        km.cxx("report(pages);", takes=[pages])


@pytest.mark.parametrize(
    "cls, why",
    [
        (Many, "holds 16 words a body, and this body takes wg after 16"),
        (Unstated, "pages is a word here"),
    ],
)
def test_each_refusal(cls, why):
    with pytest.raises(Refused, match=why):
        compile_class(cls, {}, {})
