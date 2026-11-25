"""Handlers sharing buffers (komorebi.dsl.compile.shared, no card): the three ways in
turn, the report, the change at a handler's entry, an index's one format, refusals, a
merge (one with a hand handler, built in the tree lane), the command function's stack
frame (tree lane), and a constant named as a handler's name in the thread header."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from class_handler_constants_fixture import TENSORS, Gate

import komorebi as km
from komorebi import formats
from komorebi.device import mock, native
from komorebi.device.tensor import Tensor
from komorebi.dsl import compiler
from komorebi.dsl.compile.records import Refused, Unfit
from komorebi.errors import ProgramError
from komorebi.program import several
from komorebi.program.checks import stack

MIX = "(void)ia; (void)ib; (void)ic; (void)total; (void)mixed;"
FOUR = "(void)ia; (void)ib; (void)ic; (void)id; (void)total;"


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 4096, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def words(page=1024, dtype="uint32"):
    """A tensor of rows of `page` B."""
    return Tensor(Pages(page), (4096, page // formats.itemsize(dtype)), dtype, "row")


def tiles(dtype):
    """A tensor of tiles of `dtype`."""
    page = 32 * 32 * formats.itemsize(dtype)
    return Tensor(Pages(page), (4096, 32), dtype, "tile")


def quiet(line):
    pass


class Three(km.Program):
    """Five streams a handler, items of 256, 512 and 1,024 B: 30 indices as one set,
    and no two handlers' streams of one page."""

    @km.handler
    def first(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, s: km.Tensor, x: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        mixed = km.Stream(x.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=0, count=64, size=256)),
            km.Stream(b.items(first=0, count=64, size=256)),
            km.Stream(c.items(first=0, count=64, size=256)),
        ):
            km.cxx(MIX, takes=[ia, ib, ic, total, mixed], on="pack")

    @km.handler
    def second(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, s: km.Tensor, x: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=512), write=True)
        mixed = km.Stream(x.items(first=0, count=64, size=512), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=0, count=64, size=512)),
            km.Stream(b.items(first=0, count=64, size=512)),
            km.Stream(c.items(first=0, count=64, size=512)),
        ):
            km.cxx(MIX, takes=[ia, ib, ic, total, mixed], on="pack")

    @km.handler
    def third(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, s: km.Tensor, x: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=1024), write=True)
        mixed = km.Stream(x.items(first=0, count=64, size=1024), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=0, count=64, size=1024)),
            km.Stream(b.items(first=0, count=64, size=1024)),
            km.Stream(c.items(first=0, count=64, size=1024)),
        ):
            km.cxx(MIX, takes=[ia, ib, ic, total, mixed], on="pack")


def three(**given):
    io = {name: words() for name in "abcsx"}
    return Three(l1=1 << 20, log=quiet, **io, **given)


class Orders(km.Program):
    """Four handlers reading the same four pages in four orders: 16 read pairs by a
    stream's place, four by shape."""

    @km.handler
    def one(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, d: km.Tensor, s: km.Tensor):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic, id in zip(
            km.Stream(a.items(first=0, count=64, size=256)),
            km.Stream(b.items(first=0, count=64, size=512)),
            km.Stream(c.items(first=0, count=64, size=1024)),
            km.Stream(d.items(first=0, count=64, size=2048)),
        ):
            km.cxx(FOUR, takes=[ia, ib, ic, id, total], on="pack")

    @km.handler
    def two(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, d: km.Tensor, s: km.Tensor):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic, id in zip(
            km.Stream(a.items(first=0, count=64, size=512)),
            km.Stream(b.items(first=0, count=64, size=1024)),
            km.Stream(c.items(first=0, count=64, size=2048)),
            km.Stream(d.items(first=0, count=64, size=256)),
        ):
            km.cxx(FOUR, takes=[ia, ib, ic, id, total], on="pack")

    @km.handler
    def three(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, d: km.Tensor, s: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic, id in zip(
            km.Stream(a.items(first=0, count=64, size=1024)),
            km.Stream(b.items(first=0, count=64, size=2048)),
            km.Stream(c.items(first=0, count=64, size=256)),
            km.Stream(d.items(first=0, count=64, size=512)),
        ):
            km.cxx(FOUR, takes=[ia, ib, ic, id, total], on="pack")

    @km.handler
    def four(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, d: km.Tensor, s: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic, id in zip(
            km.Stream(a.items(first=0, count=64, size=2048)),
            km.Stream(b.items(first=0, count=64, size=256)),
            km.Stream(c.items(first=0, count=64, size=512)),
            km.Stream(d.items(first=0, count=64, size=1024)),
        ):
            km.cxx(FOUR, takes=[ia, ib, ic, id, total], on="pack")


ORDERS_IO = {name: words(4096) for name in "abcds"}


class Fits(km.Program):
    """Two handlers of one stream each: one set."""

    @km.handler
    def small(self, a: km.Tensor, s: km.Tensor):
        for ia, total in zip(
            km.Stream(a.items(first=0, count=8, size=256)),
            km.Stream(s.items(first=0, count=8, size=256), write=True),
        ):
            km.cxx("(void)ia; (void)total;", takes=[ia, total], on="pack")

    @km.handler
    def large(self, a: km.Tensor, s: km.Tensor):
        for ia, total in zip(
            km.Stream(a.items(first=0, count=8, size=1024)),
            km.Stream(s.items(first=0, count=8, size=1024), write=True),
        ):
            km.cxx("(void)ia; (void)total;", takes=[ia, total], on="pack")


def report(program):
    return "\n".join(program.built.report)


def test_a_class_that_fits_as_one_set_is_built_as_one_set():
    program = Fits(l1=1 << 20, log=quiet, a=words(), s=words())
    assert "  buffers: as one set for the program (way 1 of 3)" in program.built.report
    assert not program.shared_indices
    files = program.built.files
    assert "km::share" not in files["movement.cpp"] + files["compute.cpp"]
    assert "kShare" not in files["fits.h"]


def test_a_class_that_fits_by_shape_only_shares_pairs_by_shape():
    with pytest.raises(Unfit):  # 16 read pairs by a stream's place
        compiler.compile_class(Orders, {}, ORDERS_IO, 1 << 20)
    program = Orders(l1=1 << 20, log=quiet, **ORDERS_IO)
    lines = program.built.report
    assert "  buffers: shared by shape (way 2 of 3)" in lines
    header = program.built.files["orders.h"]
    assert header.count("constexpr km::BufferPair ") == 5  # four read pages, a write
    for size in (256, 512, 1024, 2048):
        (line,) = [x for x in lines if f", {4 * size} B + {4 * size} B): " in x]
        assert line.endswith(": shared by one, two, three, four")
    assert not program.shared_indices
    files = program.built.files
    assert "km::share" not in files["movement.cpp"] + files["compute.cpp"]


def test_a_chained_class_takes_two_ways_and_is_refused_by_its_streams_pages():
    with pytest.raises(Refused) as raised:
        three(chain=True)
    text = str(raised.value)
    assert text.startswith(
        "Three: a class with a chain shares buffers as one set or by shape only, and "
        "its buffers fit neither: handlers first, second"
    )
    assert "first's kRead0 (uint32, 256 B) and second's kRead0b (uint32, 512 B)" in text
    assert (
        "; a chain's cores go on from one call to the next without waiting; give "
        "these streams the same pages, or run the calls as commands (as one set for "
        "the program and shared by shape: "
    ) in text


def test_the_runtime_form_does_not_take_the_third_way():
    with pytest.raises(Refused) as raised:
        three(form="runtime")
    assert str(raised.value).startswith(
        "Three: its buffers fit neither as one set nor by shape, and "
        "form=\"runtime\": a program's parts in the runtime's form do not set a "
        "handler's pages at its entry (shared indices of other pages); make it in "
        "the kernel form, or give its streams the same pages (as one set for the "
        "program and shared by shape: "
    )


class Overlaid(km.Program):
    """Three's handlers and an overlay part."""

    first, second, third = Three.first, Three.second, Three.third

    @km.handler(overlay=True)
    def part(self, a: km.Tensor, s: km.Tensor):
        for ia, total in zip(
            km.Stream(a.items(first=0, count=8, size=256)),
            km.Stream(s.items(first=0, count=8, size=256), write=True),
        ):
            km.cxx("(void)ia; (void)total;", takes=[ia, total], on="pack")


def test_an_overlay_part_does_not_take_the_third_way():
    io = {name: words() for name in "abcsx"}
    with pytest.raises(Refused) as raised:
        Overlaid(l1=1 << 20, log=quiet, **io)
    assert (
        "and part with overlay=: a handler's overlay parts do not set a handler's "
        "pages at its entry (shared indices of other pages); make it without "
        "overlay=, or give its streams the same pages"
    ) in str(raised.value)


class AnySize(km.Program):
    """Three's first two handlers and a third whose item size is a constant
    parameter."""

    first, second = Three.first, Three.second

    @km.handler
    def third(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        s: km.Tensor,
        x: km.Tensor,
        size: km.constant[int] = 1024,
    ):
        total = km.Stream(s.items(first=0, count=64, size=size), write=True)
        mixed = km.Stream(x.items(first=0, count=64, size=size), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=0, count=64, size=size)),
            km.Stream(b.items(first=0, count=64, size=size)),
            km.Stream(c.items(first=0, count=64, size=size)),
        ):
            km.cxx(MIX, takes=[ia, ib, ic, total, mixed], on="pack")


def test_a_set_compiled_at_its_first_use_does_not_take_the_third_way():
    io = {name: words() for name in "abcsx"}
    with pytest.raises(Refused) as raised:
        AnySize(l1=1 << 20, log=quiet, handlers={"third": km.ANY}, **io)
    assert (
        "and km.ANY: a set compiled at its first use takes the load's buffers and "
        "does not set its pages at its entry; list the sets (handlers=), or give its "
        "streams the same pages"
    ) in str(raised.value)
    Gate(l1=1 << 20, log=quiet, handlers={"gate": km.ANY}, **TENSORS)  # one set


class Met(km.Program):
    """A constant whose name in the thread header is a handler's (MET_GATE_WORDS)."""

    GATE_WORDS = km.constant(3)

    @km.handler
    def gate(self, a: km.Tensor, s: km.Tensor):
        for ia, total in zip(
            km.Stream(a.items(first=0, count=8, size=256)),
            km.Stream(s.items(first=0, count=8, size=256), write=True),
        ):
            km.cxx("(void)ia; (void)total;", takes=[ia, total], on="pack")


def test_a_constant_named_as_a_handler_s_thread_header_name_is_refused():
    with pytest.raises(ProgramError) as raised:
        Met(l1=1 << 20, log=quiet, a=words(), s=words())
    assert str(raised.value) == (
        "Met: constant GATE_WORDS is MET_GATE_WORDS in the thread header "
        "(met_thread.h), and handler gate gives that name too; rename the constant"
    )


def rows(program, number, cpu="#else  // the movers"):
    """Set `number`'s rows of a CPU in the header: (index, pages, page bytes)."""
    header = program.built.files[f"{type(program).__name__.lower()}.h"]
    header = header.split(cpu)[1]
    table = header.split(f"kShare{number}Rows[] = {{\n")[1].split("};")[0]
    return [
        tuple(int(word) for word in line.split("{")[1].split("}")[0].split(", "))
        for line in table.strip().split("\n")
    ]


def test_a_class_past_both_ways_shares_indices_of_other_pages_set_at_entry():
    program = three()  # the first two ways refuse it (the chained class's test)
    lines = program.built.report
    assert (
        lines.count(
            "  buffers: at shared indices of other pages (way 3 of 3); commands of "
            "handlers that give an index other pages do not overlap on a core"
        )
        == 1
    )
    assert program.shared_indices
    for number, page in ((1, 256), (2, 512), (3, 1024)):  # ten indices each
        assert [i for i, _, _ in rows(program, number)] == [*range(6), *range(16, 20)]
        assert {b for _, _, b in rows(program, number)} == {page}
    for cpu, indices in (
        ("defined(TRISC_UNPACK)", range(6)),
        ("(TRISC_PACK)", (16, 17, 18, 19)),
    ):
        assert [i for i, _, _ in rows(program, 1, cpu)] == list(indices)  # its own
    header = program.built.files["three.h"]
    math = header.split("defined(TRISC_MATH)")[1].split("#else")[0]
    assert "kShareTables = nullptr;" in math and "Rows" not in math  # sets none
    assert "#define KM_SHARE_WORD uint16_t" in header  # every field fits 16 bits
    assert '#include "komorebi/device/core/km_shared_indices.hpp"' in header
    assert "constexpr uint32_t kShares = 3;" in header
    for handler in ("first", "second", "third"):
        assert f"namespace km_buffers_{handler} {{" in header


def test_an_index_takes_its_largest_use_and_each_handler_its_own_pages():
    program = three()
    header = program.built.files["three.h"]
    declared = header.split("kShared0 = ")[0].rsplit("\n// ", 1)[1]
    assert "index 0 of first, second, third: 4096 B, third's pages declared" in (
        declared
    )
    assert (
        "    buffer 0: 4096 B; first 4 x 256 B, second 4 x 512 B, third 4 x 1024 B"
        in (program.built.report)
    )
    assert (
        "    buffer 16: 2048 B; first 2 x 256 B, second 2 x 512 B, third 2 x 1024 B"
        in (program.built.report)
    )
    assert "    pages 1 (first): 10 buffers" in program.built.report


def test_the_reader_gates_a_change_and_each_cpu_sets_its_pages_at_entry():
    files = three().built.files
    for kernel in ("movement", "compute"):
        text = files[f"{kernel}.cpp"]
        for handler in ("First", "Second", "Third"):
            assert (
                f"if (kShareOf{handler} != km::share_now) {{\n"
                f"                    km::share_set({kernel}, kShareTables, "
                f"kShareOf{handler});\n"
            ) in text
        assert text.count("using namespace km_buffers_") == 3
    movement = files["movement.cpp"]
    assert movement.index("#define KM_SHARE_GATED\n") < movement.index("#include")
    assert (
        "    const uint32_t pages = km_share_of(km::handler(command));\n"
        "    return pages == 0 || pages == km::share_now ||\n"
        "        km::share_gate(reader, command, kShareTables, pages);\n"
    ) in movement
    assert "share_gate" not in files["compute.cpp"]
    assert "KM_SHARE_GATED" not in files["compute.cpp"]
    assert "case kHandlerThird: return 3;" in files["three.h"]
    plain = Fits(l1=1 << 20, log=quiet, a=words(), s=words()).built.files
    assert "km::share" not in plain["movement.cpp"] + plain["compute.cpp"]
    assert "KM_SHARE_GATED" not in plain["movement.cpp"]


WORK = "(void)it;"


class Kinds(km.Program):
    """Three's handlers and two of tiles, bfloat16 and float32."""

    first, second, third = Three.first, Three.second, Three.third

    @km.handler
    def halves(self, h: km.Tensor):
        for it in km.Stream(h.pages(0, 8)):
            km.cxx(WORK, takes=[it], on="pack")

    @km.handler
    def floats(self, f: km.Tensor):
        for it in km.Stream(f.pages(0, 8)):
            km.cxx(WORK, takes=[it], on="pack")


def pairs(program, handler):
    """The indices of `handler`'s pairs in the header: {"0, 1", ...}."""
    header = program.built.files[f"{type(program).__name__.lower()}.h"]
    own = header.split(f"namespace km_buffers_{handler} {{")[1].split("\n}")[0]
    return {
        line.split("{")[1].split("}")[0]
        for line in own.split("\n")
        if "km::BufferPair kRead" in line
    }


def test_an_index_compute_unpacks_has_one_format_among_its_handlers():
    io = {name: words() for name in "abcsx"}
    program = Kinds(
        l1=1 << 20, log=quiet, h=tiles("bfloat16"), f=tiles("float32"), **io
    )
    (halves,), (floats,) = pairs(program, "halves"), pairs(program, "floats")
    items = pairs(program, "first")  # uint32
    assert len(items) == 3 and halves != floats
    assert halves not in items and floats not in items


class Crowded(km.Program):
    """A handler of 13 read pairs of uint32 items (indices 0 to 25), one of float32
    tiles."""

    @km.handler
    def many(self, a: km.Tensor):
        for ia, ib, ic, id, ie, i_f, ig, ih, ii, ij, ik, il, im in zip(
            km.Stream(a.items(first=0, count=8, size=256)),
            km.Stream(a.items(first=8, count=8, size=256)),
            km.Stream(a.items(first=16, count=8, size=256)),
            km.Stream(a.items(first=24, count=8, size=256)),
            km.Stream(a.items(first=32, count=8, size=256)),
            km.Stream(a.items(first=40, count=8, size=256)),
            km.Stream(a.items(first=48, count=8, size=256)),
            km.Stream(a.items(first=56, count=8, size=256)),
            km.Stream(a.items(first=64, count=8, size=256)),
            km.Stream(a.items(first=72, count=8, size=256)),
            km.Stream(a.items(first=80, count=8, size=256)),
            km.Stream(a.items(first=88, count=8, size=256)),
            km.Stream(a.items(first=96, count=8, size=256)),
        ):
            km.cxx(
                "",
                takes=[ia, ib, ic, id, ie, i_f, ig, ih, ii, ij, ik, il, im],
                on="pack",
            )

    @km.handler
    def floats(self, f: km.Tensor):
        for it in km.Stream(f.pages(0, 8)):
            km.cxx(WORK, takes=[it], on="pack")


def test_past_the_three_ways_the_refusal_names_the_handler_of_most_buffers():
    with pytest.raises(Refused) as raised:
        Crowded(l1=1 << 20, log=quiet, a=words(), f=tiles("float32"))
    text = str(raised.value)
    assert text.startswith(
        "Crowded: its buffers fit neither as one set, nor by shape, nor at shared "
        "indices of other pages; many has the most buffers, 26 (kRead0, kRead1, "
    )
    assert "(as one set for the program and shared by shape: floats's read" in text
    assert (
        "; at shared indices of other pages: Crowded.floats: its kRead0b (float32 (to "
        "Dst as float32)) finds no index from 0 to 25: index 0 is uint32 in another "
        "handler, and an index compute unpacks or packs through has one format among "
        "the handlers; give the buffers one format, or split the class)"
    ) in text


class Spill(km.Program):
    """A handler of six write pairs: indices 16 to 25 hold five."""

    @km.handler
    def spill(self, a: km.Tensor, s: km.Tensor):
        o0 = km.Stream(s.items(first=0, count=8, size=256), write=True)
        o1 = km.Stream(s.items(first=8, count=8, size=256), write=True)
        o2 = km.Stream(s.items(first=16, count=8, size=256), write=True)
        o3 = km.Stream(s.items(first=24, count=8, size=256), write=True)
        o4 = km.Stream(s.items(first=32, count=8, size=256), write=True)
        o5 = km.Stream(s.items(first=40, count=8, size=256), write=True)
        for item in km.Stream(a.items(first=0, count=8, size=256)):
            km.cxx("", takes=[item, o0, o1, o2, o3, o4, o5], on="pack")


def test_a_write_stream_s_refusal_names_its_handler():
    with pytest.raises(Refused) as raised:
        Spill(l1=1 << 20, log=quiet, a=words(), s=words())
    text = str(raised.value)
    assert "spill's write stream needs pair kWrite5" in text
    assert "kWrite0 16-17 (uint32, spill)" in text and "None" not in text


def test_a_switching_run_and_an_added_program_refuse_shared_indices():
    program = three()
    assert several.shared_indices(program)
    assert several.shared_refusal(program, "a switching run's") == (
        "three: its handlers give an index other pages, set at a handler's entry, "
        "which a switching run's program does not do (a switch sets a program's "
        "buffers once); run it in a run of its own, or give its handlers' streams "
        "the same pages so its buffers fit as one set or by shape"
    )
    fits = Fits(l1=1 << 20, log=quiet, a=words(), s=words())
    assert not several.shared_indices(fits)
    assert not several.shared_indices(Orders(l1=1 << 20, log=quiet, **ORDERS_IO))


class FirstAlone(km.Program):
    first = Three.first


class TwoMore(km.Program):
    second, third = Three.second, Three.third


def test_a_merge_takes_the_ways_in_turn_and_shares_indices_when_both_refuse():
    io = {name: words() for name in "abcsx"}
    first = FirstAlone(l1=1 << 20, log=quiet, **io)
    more = TwoMore(l1=1 << 20, log=quiet, **io)
    assert not first.shared_indices and not more.shared_indices  # one set each
    merged = km.merge(first, more, l1=1 << 20, log=quiet)
    assert merged.shared_indices
    header = next(
        text for name, text in merged.built.files.items() if name.endswith(".h")
    )
    assert "constexpr uint32_t kShares = 3;" in header


class Noted(km.Program):
    """A hand handler (km.kernel) on an own field only: pack fills, mover B writes."""

    record: km.L1[km.u32, 1, 64] = km.own(push="pack", raw=True)

    @km.handler(
        km.kernel(
            writer="kernel_note/note_writer.hpp",
            pack="kernel_note/note_pack.hpp",
            entry="note_k::note",
        )
    )
    def note(self, records: km.Tensor, tiles: int):
        """The record page."""


def hand_merge(**given):
    """FirstAlone and TwoMore (shared indices once merged) merged with Noted."""
    io = {name: words() for name in "abcsx"}
    parts = (
        FirstAlone(l1=1 << 20, log=quiet, **io, **given),
        TwoMore(l1=1 << 20, log=quiet, **io, **given),
        Noted(l1=1 << 20, log=quiet, records=words(), **given),
    )
    return km.merge(*parts, l1=1 << 20, log=quiet, **given)


def test_a_hand_handler_s_command_passes_the_gate_and_every_cpu_keeps_its_set():
    merged = hand_merge()
    assert merged.shared_indices
    files = merged.built.files
    header = next(text for name, text in files.items() if name.endswith("noted.h"))
    assert "constexpr uint32_t kShareOfNote = 0;" in header
    assert "case kHandlerNote" not in header  # km_share_of: 0, the set stays
    field = int(header.split("constexpr uint32_t kFieldRecord = ")[1].split(";")[0])
    mask = int(header.split("{kShare1Rows, 10, ")[1].split("}")[0], 16)
    assert not mask >> field & 1  # its field at an index of its own
    movement, compute = files["movement.cpp"], files["compute.cpp"]
    assert movement.startswith("// ") and "#define KM_SHARE_GATED\n" in movement
    for text in (movement, compute):
        note = text[text.index("case kHandlerNote: {") :]
        assert "share_set" not in note.split("case kHandler")[1]


class Same(km.Program):
    """Three with items of 256 B in all three handlers: one set."""

    first = Three.first

    @km.handler
    def second(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, s: km.Tensor, x: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        mixed = km.Stream(x.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=0, count=64, size=256)),
            km.Stream(b.items(first=0, count=64, size=256)),
            km.Stream(c.items(first=0, count=64, size=256)),
        ):
            km.cxx(MIX, takes=[ia, ib, ic, total, mixed], on="pack")

    @km.handler
    def third(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, s: km.Tensor, x: km.Tensor
    ):
        total = km.Stream(s.items(first=0, count=64, size=256), write=True)
        mixed = km.Stream(x.items(first=0, count=64, size=256), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=0, count=64, size=256)),
            km.Stream(b.items(first=0, count=64, size=256)),
            km.Stream(c.items(first=0, count=64, size=256)),
        ):
            km.cxx(MIX, takes=[ia, ib, ic, total, mixed], on="pack")


def test_three_of_one_page_fits_as_one_set():
    io = {name: words() for name in "abcsx"}
    program = Same(l1=1 << 20, log=quiet, **io)
    assert "  buffers: as one set for the program (way 1 of 3)" in program.built.report


# A mock load of Three (shared indices) and Same (one set): each build's ELFs.
LOADS = """
import json, sys
sys.path[:0] = sys.argv[2:]
import komorebi as km
from komorebi.session.parts import load
from test_shared_buffers import Same, Three, hand_merge

seen = {}
checked = load.LoadMixin._check_stack


def kept(self, program, elves):
    seen[type(program).__name__] = [[str(s), [str(e) for e in es]] for s, es in elves]
    return checked(self, program, elves)


load.LoadMixin._check_stack = kept
with km.Session() as session:
    io = {name: session.empty((4096, 256), "uint32") for name in "abcsx"}
    for cls in (Three, Same):
        session.load(cls(session=session, log=lambda line: None, **io)).stop()
    session.load(hand_merge(session=session)).stop()  # builds (a hand handler's too)
with open(sys.argv[1], "w") as out:
    json.dump(seen, out)
"""
COMMAND = "_ZZ11kernel_mainvENKUlRK12km_command_tE_cl"  # the command function


def command_stack(elves, tree):
    """{RISC: the stack bytes of the function the command function is in}: its own
    where it is out of line, else _start's (inlined into it)."""
    taken = {}
    for _, risc, elf in stack.kernel_images(elves):
        found, _ = stack.read(elf, tree)
        own = [f.frame for f in found.values() if f.name.startswith(COMMAND)]
        start = [f.frame for f in found.values() if f.name == "_start"]
        taken[risc] = max(own or start)
    return taken


@pytest.mark.tree
def test_a_command_function_s_frame_is_within_16_b_of_one_set_s_on_each_cpu(tmp_path):
    here = Path(__file__).parent
    env = dict(os.environ, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor((0,), "blackhole"))
    env[mock.MEMORY] = str(tmp_path / "kernel_buffer.json")
    env["TT_VISIBLE_DEVICES"] = "0"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(km.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    out = tmp_path / "elves.json"
    paths = [str(here), str(here / "constants")]
    done = subprocess.run(
        [sys.executable, "-c", LOADS, str(out), *paths],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    seen = json.loads(out.read_text())
    tree = native.metal_tree()
    shared, one = (command_stack(seen[name], tree) for name in ("Three", "Same"))
    assert set(shared) == set(one) == set(stack.THREADS.values())
    for risc in shared:
        assert abs(shared[risc] - one[risc]) <= 16, (risc, shared, one)
    (hand,) = set(seen) - {"Three", "Same"}  # the merge with a hand handler, built
    assert {risc for _, risc, _ in stack.kernel_images(seen[hand])} == set(shared)
