"""A row longer than its stream's block, and a per-row operand in the stream form (no
card): the range read once a pass, a page pair a row (one where one sum pass), its rows
a count or a word under a most, written in the last pass; a field's `.col()` and a
stream tile's `.row()`; what the passes do not take refused by name."""

import pytest
from test_class_lines import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compile.tile_functions import known_count
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source, movement_source

W = 32  # tiles a row


@km.tile_function
def layer(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * W))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * W))
    return d * km.rsqrt(var + 1e-5).col() * weight.row() + bias.row()


@km.tile_function
def apply(x, w, s):
    return x * km.rsqrt(s.col() * (1.0 / 2048) + 1e-6) * w.row()


class Long(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        for row in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            results.push(layer(row, self.weight, self.bias))


class Plain(km.Program):
    @km.handler
    def rows(self, x: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        for t in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            results.push(t)


class Counted(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor, rows: int):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(0, rows * W), write=True, block=8)
        for row in km.Stream(x.pages(0, rows * W), block=8, row=W):
            results.push(layer(row, self.weight, self.bias))


@km.tile_function
def rms(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * W))
    return row * km.rsqrt(ms + 1e-6).col() * weight.row()


class Most(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def layer(
        self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor, rows: int
    ):
        r0 = self.rank * 6
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(
            out.tiles[r0 : min(r0 + 6, r0 + rows), 0:W], write=True, block=8
        )
        for row in km.Stream(x.tiles[r0 : min(r0 + 6, r0 + rows), 0:W], block=8, row=W):
            results.push(layer(row, self.weight, self.bias))

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, rows: int):
        r0 = self.rank * 6
        self.weight.fill(km.Stream(w.pages(0, 16)))
        results = km.Stream(
            out.tiles[r0 : min(r0 + 6, r0 + rows), 0:W], write=True, block=8
        )
        for row in km.Stream(x.tiles[r0 : min(r0 + 6, r0 + rows), 0:W], block=8, row=W):
            results.push(rms(row, self.weight))


@km.tile_function
def normed(row, weight):
    n = row * km.rsqrt(km.reduce_sum(row * row, exact=True) * (1.0 / 1024) + 1e-6).col()
    return n * weight.row(), n


@km.tile_function
def three(row, weight):
    n = row * km.rsqrt(km.reduce_sum(row * row, exact=True) * (1.0 / 1024) + 1e-6).col()
    return n * weight.row(), n, n * 2.0


class Results(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def whole(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, b: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        ys = km.Stream(out.pages(0, 2 * W), write=True, block=W)
        ns = km.Stream(b.pages(0, 2 * W), write=True, block=W)
        for row in km.Stream(x.pages(0, 2 * W), block=W, row=W):
            y, n = normed(row, self.weight)
            ys.push(y)
            ns.push(n)

    @km.handler
    def parts(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, b: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        ys = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        ns = km.Stream(b.pages(0, 2 * W), write=True, block=8)
        for row in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            y, n = normed(row, self.weight)
            ys.push(y)
            ns.push(n)


class Three(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, b: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        ys = km.Stream(out.pages(0, 2 * W), write=True, block=W)
        ns = km.Stream(b.pages(0, 2 * W), write=True, block=W)
        for row in km.Stream(x.pages(0, 2 * W), block=W, row=W):
            y, n, m = three(row, self.weight)
            ys.push(y)
            ns.push(n)


class Halves(km.Program):
    weight: km.L1[km.bf16, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        for row in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            results.push(layer(row, self.weight, self.bias))


class Split(km.Program):
    s: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, s: km.Tensor, out: km.Tensor):
        self.s.fill(km.Stream(s.pages(0, 1)))
        results = km.Stream(out.pages(0, 4), write=True, block=2)
        for xt, wt in zip(
            km.Stream(x.pages(0, 4), block=2), km.Stream(w.pages(0, 4), block=2)
        ):
            results.push(apply(xt, wt, self.s))


@km.tile_function
def square_sums(row):
    return km.reduce_sum(row * row, exact=True).col()


class SplitNorm(km.Program):
    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def norm(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        for row in km.Stream(x.pages(0, 64), block=8, row=64):
            self.part.write(square_sums(row))
        results = km.Stream(out.pages(0, 4), write=True, block=2)
        for xt, wt in zip(
            km.Stream(x.pages(0, 4), block=2), km.Stream(w.pages(0, 4), block=2)
        ):
            results.push(apply(xt, wt, self.part))


def built(cls):
    tensors = {n: Tensor(Pages(4096), (64, 32 * W), "float32", "tile") for n in "xo"}
    tensors["out"] = tensors.pop("o")
    for n in "wb":
        tensors[n] = Tensor(Pages(4096), (16 * 32, 32), "float32", "tile")
    if cls is Halves:
        tensors["w"] = Tensor(Pages(2048), (16 * 32, 32), "bfloat16", "tile")
    if cls is Split:
        tensors = {n: Tensor(Pages(4096), (32, 128), "float32", "tile") for n in "xw"}
        tensors["s"] = Tensor(Pages(4096), (32, 32), "float32", "tile")
        tensors["out"] = Tensor(Pages(4096), (32, 128), "float32", "tile")
    if cls is SplitNorm:
        tensors = {
            n: Tensor(Pages(2048), (32, 2048), "bfloat16", "tile")
            for n in ("x", "w", "out")
        }
    if cls is Most:
        tensors = {n: tensors[n] for n in ("x", "w", "b", "out")}
        for n in ("x", "out"):
            tensors[n] = Tensor(Pages(4096), (6 * 32, 32 * W), "float32", "tile")
    if cls in (Results, Three):
        tensors = {n: tensors[n] for n in ("x", "w", "out")}
        tensors["b"] = Tensor(Pages(4096), (64, 32 * W), "float32", "tile")
    if cls is Plain:
        tensors = {n: tensors[n] for n in ("x", "out")}
    return compile_class(cls, {}, tensors)


def test_the_row_is_read_once_a_pass_and_written_in_the_last():
    compiled = built(Long)
    case = compute_source(compiled)
    case = case[case.index("case kHandlerRows") :]
    assert "km_block0 < km_blocks0 * 3;" in case  # two sum levels, then the writes
    assert "const uint32_t km_part0 = km_block0 % km_blocks0 % 4;" in case
    assert "km_block0 / km_blocks0 == 2 && compute.reserve_back(" in case
    assert "km::block_half(kWrite0, kWrite0Pattern, km_block0 % km_blocks0)" in case
    assert "MATH((km_tile::layer_sum1(0)));" in case
    assert "copy_tile(kChunkRowsLayer, 1, 3);" in case  # the sums, back for pass 2
    assert "copy_tile(kFieldBias, km_part0 * 4 + km_chunk0, 5);" in case
    pages = {b.name: b.pages for b in compiled.buffers}
    assert pages["kChunkRowsLayer"] == (4,)  # a pair a row, 2 rows
    assert ", 3, kRead0, kRead0Pattern" in movement_source(compiled)


def test_what_the_passes_do_not_take_is_refused():
    with pytest.raises(Refused, match="a row read in parts is a row function's"):
        built(Plain)
    with pytest.raises(Refused, match="its rows a constant count or a stated most"):
        built(Counted)
    assert known_count("(2 * 32)") == 64 and known_count("rows * 32") is None


def test_a_row_count_word_takes_the_pages_of_its_most():
    compiled = built(Most)
    pages = {b.name: b.pages for b in compiled.buffers}
    assert pages["kChunkRowsLayer"] == (12,)  # two sum levels: a pair a row, 6 rows
    assert pages["kChunkRowsRms"] == (6,)  # one sum level: the values' tile a row
    case = compute_source(compiled)
    layer_case = case[case.index("case kHandlerLayer") : case.index("case kHandlerRms")]
    rms_case = case[case.index("case kHandlerRms") :]
    assert "rows = command.a4;" in layer_case  # the count read at run time
    assert "km_block0 < km_blocks0 * 3;" in layer_case
    assert "km_block0 < km_blocks0 * 2;" in rms_case
    assert "cb_reserve_back(kChunkRowsRms, 1);" in rms_case
    assert "pack_tile(3, kChunkRowsRms);" not in rms_case
    assert "cb_reserve_back(kChunkRowsLayer, 2);" in layer_case


def test_several_results_of_a_wide_row_each_pack_their_own_tiles():
    compiled = built(Results)
    case = compute_source(compiled)
    for name in ("Whole", "Parts"):
        at = case.index(f"case kHandler{name}")
        one = case[at : case.find("case kHandler", at + 1) % len(case)]
        assert "pack_tile<true>(0, km_write0_out, km_tile0 + 2 * km_chunk0);" in one
        assert "pack_tile<true>(5, km_write1_out, km_tile0 + 2 * km_chunk0);" in one
        assert "pack_tile<true>(6, km_write1_out, km_tile0 + 2 * km_chunk0 + 1);" in one
        assert one.count("MATH((km_tile::normed(0)));") == 1
    unit = compiled.units["normed"][1]
    assert (unit.results, unit.chunk_results) == (2, [0, 5])  # past the weight's 4
    with pytest.raises(Refused, match="3 results; a wide row's call holds 2"):
        built(Three)


def test_a_bf16_field_is_unpacked_in_its_own_setup():
    case = compute_source(built(Halves))
    case = case[case.index("case kHandlerRows") :]
    own = "KM_UNPACK_FORMAT_A(kFieldWeight);\ncopy_init(kFieldWeight);\ncopy_tile("
    assert own in "\n".join(line.strip() for line in case.splitlines())
    assert "copy_init(kChunkRowsLayer);\ncopy_tile(kFieldBias" in "\n".join(
        line.strip() for line in case.splitlines()
    )


def test_a_field_by_col_is_broadcast_and_a_stream_row_reads_its_half():
    case = compute_source(built(Split))
    case = case[case.index("case kHandlerApply") :]
    assert "unary_bcast<BroadcastType::COL>(kFieldS, 0, " in case
    assert "unary_bcast<BroadcastType::ROW>(km_half1, " in case


def test_a_row_s_per_row_value_goes_to_a_field_read_by_col():
    case = compute_source(built(SplitNorm))
    case = case[case.index("case kHandlerNorm") :]
    assert "km_block0 < km_blocks0;" in case  # one pass: no write pass
    assert "MATH((km_tile::square_sums_value(0)));" in case
    assert "pack_tile(0, kFieldPart);" in case
    assert "unary_bcast<BroadcastType::COL>(kFieldPart, 0, " in case
    assert case.count("cb_pop_front(kFieldPart, kFieldPartPages);") == 2  # write, end
