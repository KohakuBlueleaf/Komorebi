"""An own field of raw pages (km.own(raw=True)) and buffer indices past 25: its buffer,
header lines, L1 and solved pages, each refusal, fields at 32 up (never 33), room made
by moving an own field up, a merge, two fields of one header name, "if_present", a
kernel handler's call on a raw field (no card; kernel_hand/'s files only included)."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile import layout
from komorebi.dsl.compiler import compile_class
from komorebi.errors import ProgramError

HAND = "kernel_hand/"  # beside this file
GATE = km.kernel(movement=HAND + "movement.hpp")


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 2048, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def words():
    return Tensor(Pages(256), (2048, 64), "uint32", "row")


class Passes(km.Program):
    MESH_PAGES = km.constant(4)
    mesh: km.L1[km.u32, MESH_PAGES, 704] = km.own(push="mover_a", raw=True)
    hist: km.L1[km.u32, 1, 16384] = km.own(push=None, raw=True)

    @km.handler(GATE)
    def gate(self, n: int):
        pass


def made(cls, **given):
    return cls(log=lambda line: None, l1=1 << 20, **given)


def fields(cls, count, **declared):
    """`cls` with `count` own fields of one bf16 tile (f0, f1, ...) first."""
    body = {"__annotations__": {}}
    for index in range(count):
        body["__annotations__"][f"f{index}"] = km.L1[km.bf16, 1, 32, 32]
        body[f"f{index}"] = km.own()
    body.update(declared)
    return type(f"Fields{count}", (cls,), body)


class Gated(km.Program):
    @km.handler(GATE)
    def gate(self, n: int):
        pass


def indices(compiled):
    return {b.name: b.index_a for b in compiled.buffers if b.role == "field"}


def test_a_raw_field_is_its_pages_of_its_bytes_with_three_header_lines():
    program = made(Passes)
    buffers = {
        b.name: b
        for b in compile_class(Passes, {}, {}, constants={"MESH_PAGES": 4}).buffers
    }
    mesh, hist = buffers["kFieldMesh"], buffers["kFieldHist"]
    assert (mesh.index_a, mesh.page_bytes, mesh.pages) == (8, 2816, (4,))
    assert (hist.index_a, hist.page_bytes, hist.pages) == (9, 65536, (1,))
    header = program.built.files["passes.h"]
    assert (
        "constexpr uint32_t kFieldMesh = 8;\n"
        "constexpr uint32_t kFieldMeshPages = 4;\n"
        "constexpr uint32_t kFieldMeshPageBytes = 2816;\n"
    ) in header
    assert "kFieldHistPageBytes = 65536;" in header
    assert "unpack_to_dest_mode" not in program.compute  # no tiles in DST


def test_the_l1_sum_counts_a_raw_field_as_pages_by_page_bytes():
    report = made(Passes).built.report
    assert "  L1: the buffers take 76800 B of 1048576 B left for buffers" in report


def test_the_solved_symbol_is_a_raw_field_s_pages():
    class Solved(Gated):
        PAGES = km.symbol()
        scratch: km.L1[km.u32, PAGES, 64] = km.own(raw=True)

    compiled = compile_class(Solved, {}, {}, l1=10_000)
    assert compiled.constants["PAGES"] == 39  # 39 pages of 256 B in 10,000 B
    assert [b.pages for b in compiled.buffers] == [(39,)]


def refused(cls, **given):
    with pytest.raises(ProgramError) as raised:
        made(cls, **given)
    return str(raised.value)


def test_each_refusal_of_a_raw_field_names_it():
    class One(Gated):
        words: km.L1[km.u32, 64] = km.own(raw=True)

    class Block(Gated):
        blocks: km.L1[km.bfp8_b, 1, 64] = km.own(raw=True)

    class Odd(Gated):
        odd: km.L1[km.bf16, 1, 3] = km.own(raw=True)

    class Items(Gated):
        ITEMS = km.symbol()
        items: km.L1[km.u32, 2, ITEMS] = km.own(raw=True)

    class Lined(km.Program):
        table: km.L1[km.u32, 1, 64] = km.own(raw=True)

        @km.handler
        def drain(self, out: km.Tensor):
            pages = km.Stream(self.table, mover="B")
            km.cxx("", takes=[pages])

    assert "One.words, line" in refused(One)
    assert "a raw field is pages x items: give both" in refused(One)
    assert "a block format has a tile's bytes only" in refused(Block)
    assert "a raw field's page of 6 B: a page is whole 16 B, 16 B or more" in (
        refused(Odd)
    )
    assert "ITEMS is solved against the L1, and a raw field's solved size is" in (
        refused(Items)
    )
    assert "self.table is a raw field: bodies use it" in refused(Lined, out=words())
    with pytest.raises(ProgramError, match=r"km.own\(raw='yes'\): raw= is True"):
        km.own(raw="yes")


def test_fields_past_25_take_32_and_up_never_33_to_the_chip_s_last():
    assert indices(compile_class(fields(Gated, 19), {}, {}))["kFieldF18"] == 32
    placed = indices(compile_class(fields(Gated, 21), {}, {}))
    assert [placed[f"kFieldF{i}"] for i in (17, 18, 19, 20)] == [25, 32, 34, 35]
    assert "33" not in str(sorted(placed.values()))
    message = refused(fields(Gated, 50))
    assert "more buffers and fields than the indices: 49 buffers take 49 indices" in (
        message
    )
    assert "a class has 57 of the chip's 64" in message
    with pytest.raises(ProgramError, match="a class has 26 of the chip's 32"):
        compile_class(fields(Gated, 19), {}, {}, indices=32)  # no upper range


def test_room_is_made_by_moving_the_highest_own_field_up():
    cls = fields(Gated, 18, out=km.Pair(km.bf16, pages=2, write=True))
    compiled = compile_class(cls, {}, {})
    pair = next(b for b in compiled.buffers if b.name == "kPairOut")
    assert (pair.index_a, pair.index_b) == (16, 17)  # the pair below 26
    assert indices(compiled)["kFieldF17"] == 34  # fields past 25 up
    layout.declared_pair(compiled, "more", "write", "bfloat16", 2048, 2, "a pair")
    more = next(b for b in compiled.buffers if b.name == "kPairMore")
    assert (more.index_a, more.index_b) == (24, 25)  # room made below 26
    moved = indices(compiled)
    assert (moved["kFieldF14"], moved["kFieldF15"]) == (36, 35)  # moved up


def test_a_merge_carries_a_raw_field():
    class First(Gated):
        record: km.L1[km.u32, 1, 64] = km.own(push="pack", raw=True)

    class Second(km.Program):
        record: km.L1[km.u32, 1, 64] = km.own(push="pack", raw=True)

        @km.handler(GATE)
        def other(self, n: int):
            pass

    merged = km.merge(made(First), made(Second), log=lambda line: None)
    record = [b for b in merged.compiled.buffers if b.name == "kFieldRecord"]
    assert len(record) == 1 and (record[0].page_bytes, record[0].raw) == (256, True)
    assert "kFieldRecordPageBytes = 256;" in merged.built.files["first_second.h"]


def merge_refused(first, second):
    """The refusal of a merge of `first` and `second` (made first)."""
    with pytest.raises(ProgramError) as raised:
        km.merge(made(first), made(second), log=lambda line: None)
    return str(raised.value)


def test_a_merge_refusal_of_one_field_says_what_differs():
    class Packed(Gated):
        record: km.L1[km.u32, 1, 64] = km.own(push="pack", raw=True)

    class Moved(km.Program):
        record: km.L1[km.u32, 1, 64] = km.own(push="mover_a", raw=True)

        @km.handler(GATE)
        def other(self, n: int):
            pass

    class Raw(Gated):
        record: km.L1[km.u32, 1, 1024] = km.own(push="pack", raw=True)

    class Tiled(km.Program):
        record: km.L1[km.u32, 1, 1024] = km.own(push="pack")  # a tile of one shape

        @km.handler(GATE)
        def other(self, n: int):
            pass

    one = "a field of one name is one field: declare it alike, or rename one"
    text = merge_refused(Packed, Moved)  # push= alone differs
    assert "push='pack' in Packed and push='mover_a' in Moved" in text
    assert "Field(" not in text and "pages" not in text and text.endswith(one)
    text = merge_refused(Raw, Tiled)  # raw= alone differs
    assert "raw pages in Raw, tiles in Tiled" in text
    assert "Field(" not in text and "push=" not in text and text.endswith(one)


class Both(Gated):
    out: km.L1[km.bf16, 2, 32, 32] = km.own()
    out_pages: km.L1[km.bf16, 1, 32, 32] = km.own()


class OutPages(km.Program):
    out_pages: km.L1[km.bf16, 1, 32, 32] = km.own()

    @km.handler(GATE)
    def other(self, n: int):
        pass


def test_two_fields_that_give_the_header_one_name_are_refused_naming_both():
    shared = "field out and field out_pages both give the header kFieldOutPages"
    text = refused(Both)
    assert text.startswith("Both.out_pages, line ")
    assert text.endswith(
        f"`out_pages: km.L1[km.bf16, 1, 32, 32] = km.own()`: {shared}; rename one"
    )

    class Out(Gated):
        out: km.L1[km.bf16, 2, 32, 32] = km.own()

    assert merge_refused(Out, OutPages).endswith(f"{shared}; rename one")

    class RawBytes(Gated):  # a raw field's third name
        out: km.L1[km.u32, 1, 64] = km.own(push="pack", raw=True)
        out_page_bytes: km.L1[km.bf16, 1, 32, 32] = km.own()

    assert "give the header kFieldOutPageBytes; rename one" in refused(RawBytes)

    class Alike(Gated):  # names that only look alike
        out: km.L1[km.bf16, 2, 32, 32] = km.own()
        outpages2: km.L1[km.bf16, 1, 32, 32] = km.own()
        out_page_bytes: km.L1[km.bf16, 1, 32, 32] = km.own()  # out is not raw

    header = made(Alike).built.files["alike.h"]
    for name in ("kFieldOutPages", "kFieldOutpages2Pages", "kFieldOutPageBytesPages"):
        assert f"constexpr uint32_t {name} = " in header


def test_the_vector_option_if_present_is_accepted():
    class Vector(Gated, compute={"enable_trisc2_rvv": "if_present"}):
        pass

    assert made(Vector).compute["enable_trisc2_rvv"] == "if_present"


class Noted(km.Program):
    record: km.L1[km.u32, 1, 64] = km.own(push="pack", raw=True)

    @km.handler(
        km.kernel(
            writer=HAND + "reader.hpp", pack=HAND + "math.hpp", entry="note_k::note"
        )
    )
    def note(self, records: km.Tensor, tiles: int):
        """The record page: pack fills it, mover B writes it."""


def case(text, handler):
    """One handler's case of a kernel source, to the next case."""
    start = text.index(f"case kHandler{handler}: {{")
    end = text.find("case kHandler", start + 1)
    return text[start : end if end > 0 else len(text)]


def test_a_kernel_handler_on_a_raw_field_is_one_call_on_each_cpu():
    files = made(Noted, records=words()).built.files
    movement = case(files["movement.cpp"], "Note")
    writer = movement.index("#if !defined(COMPILE_FOR_BRISC)  // mover B")
    call = "if (!note_k::note_writer(movement, command, records, tiles)) {"
    assert writer < movement.index(call)
    compute = case(files["compute.cpp"], "Note")
    pack = compute.index("#if defined(TRISC_PACK)")
    assert pack < compute.index("note_k::note_pack(compute, command, tiles);")
    assert "constexpr uint32_t kFieldRecordPageBytes = 256;" in files["noted.h"]
