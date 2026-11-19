"""The tile compiler's words against the chip of the load (chips.Chip.tile_refusal):
Blackhole takes every one; Wormhole refuses by name the instructions, modifiers and
next-cycle reads it runs otherwise; a class's words come from its make (no card)."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device import chips
from komorebi.device.tensor import Tensor
from komorebi.dsl import program as dsl_program
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.tile import emit, matrix, trace, unit_program
from komorebi.dsl.tile import unit as unit_module
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.errors import ProgramError
from komorebi.session import kernels

WORMHOLE = chips.chip("wormhole_b0").tile_refusal


def word(op, kind="simple", reads=(), writes=(), **args):
    return (op, args, "", kind, list(reads), list(writes))


def header(name="silu", fn=trace.silu, inputs=1):
    """A unit program of `fn` over `inputs` streams and its emitted header."""
    params = [TensorParam(n, False, F32) for n in "abc"[:inputs]]
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [*params, y], [], {})
    with trace.Tracer(ir):
        proxies, ys = [trace.TensorProxy(p) for p in params], trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            trace.store(ys.tiles[t], fn(*[trace.load(p.tiles[t]) for p in proxies]))
    unit = unit_program(ir)
    return unit, emit(unit)


def silu_header():
    return header()


def test_wormhole_s_vector_words_take_the_mode_less_4_with_the_base_set():
    """SFPLOAD, SFPSTORE take address mode 7 on Blackhole; on Wormhole 3 with
    ADDR_MOD_SET_Base set around the function's words, as tt-llk's SFPU start does."""
    _, text = silu_header()
    tile = text[text.index("inline void silu(") :]
    lines = [line.strip() for line in tile.split("\n")]
    first = lines.index("TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);")
    assert lines[first - 1] == "#if defined(ARCH_WORMHOLE)"
    load = next(i for i, line in enumerate(lines) if line.startswith("TTI_SFPLOAD("))
    assert first < load and lines[load - 1] == "#if defined(ARCH_WORMHOLE)"
    assert lines[load].endswith(", 3, 0);") and lines[load + 1] == "#else"
    assert lines[load + 2].endswith(", 7, 0);") and lines[load + 3] == "#endif"
    off = lines.index("TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);")
    assert off > load and "STALL_CFG" in lines[off - 1]


def test_wormhole_s_products_run_with_the_base_cleared():
    """A product's MVMUL takes address modes 0 to 3: on Wormhole the base is cleared
    before its words and set again after."""
    _, text = header("g", lambda a, b, c: trace.silu(a @ b) * (a @ c), 3)
    lines = [line.strip() for line in text[text.index("inline void g(") :].split("\n")]
    first = next(i for i, line in enumerate(lines) if line.startswith("TTI_MVMUL("))
    off = max(i for i in range(first) if "ADDR_MOD_SET_Base_ADDR32, 0" in lines[i])
    on = [i for i in range(first) if "ADDR_MOD_SET_Base_ADDR32, 1" in lines[i]]
    assert on and max(on) < off  # set at the top, cleared before the product
    after = min(i for i in range(first, len(lines)) if "Base_ADDR32, 1" in lines[i])
    assert all(not lines[i].startswith("TTI_SFP") for i in range(first, after))


def test_wormhole_s_product_is_the_library_s_wormhole_image():
    """On Wormhole a product's words are tt-llk's Wormhole no-MOP image (modes 0 to 3,
    the ones its init sets), a fidelity phase each, after Blackhole's #else; they pass
    Wormhole's check."""
    unit, text = header("p", lambda a, b: a @ b, 2)
    lines = [line.strip() for line in text[text.index("inline void p(") :].split("\n")]
    wormhole = lines[lines.index("#if defined(ARCH_WORMHOLE)") : lines.index("#else")]
    modes = [int(x.split(", ")[2]) for x in wormhole if x.startswith("TTI_MVMUL(")]
    phases = max(matrix.PHASES[unit.fidelity], 1)
    assert modes == list(matrix.WORMHOLE_MODES) * phases
    blackhole = lines[lines.index("#else") : lines.index("#endif")]
    assert sum(x.startswith("TTI_MVMUL(0, 0, 4, 0)") for x in blackhole) == phases * 1
    for run in unit_module.chip_words(unit.tile["math"]):
        assert all(WORMHOLE(b, w) is None for b, w in zip([None, *run], run))


def test_wormhole_s_holds_are_an_sfpnop_counted_in_the_replay():
    """a + a + a: each SFPADD's result is read the next cycle (the second SFPADD, the
    store); on Wormhole an SFPNOP before each, inside the recorded words, the REPLAYs
    counting 6 where Blackhole's count 4; its words then pass Wormhole's check."""
    unit, text = header("t", lambda a: a + a + a)
    lines = [line.strip() for line in text[text.index("inline void t(") :].split("\n")]
    assert lines.count("TTI_REPLAY(0, 6, 1, 1);") == 1
    assert lines.count("TTI_REPLAY(0, 4, 1, 1);") == 1  # Blackhole's, after #else
    adds = [i for i, line in enumerate(lines) if line.startswith("TTI_SFPADD(")]
    assert lines[adds[1] - 2 : adds[1]] == ["TTI_SFPNOP;", "#endif"]
    assert lines.count("TTI_SFPNOP;") == 2
    for function in (unit.init["math"], unit.tile["math"]):
        for run in unit_module.chip_words(function):
            assert all(WORMHOLE(b, w) is None for b, w in zip([None, *run], run))


def chain(a):
    t = a
    for _ in range(20):
        t = t * 0.5 + a
    return t


def test_wormhole_records_32_entries_and_issues_the_rest_after_each_replay():
    """20 MADs, each read the next cycle: Blackhole records the row's 22 words; on
    Wormhole they and their SFPNOPs take more than the 32 entries, so its REPLAYs
    record and replay 32 and each replay is followed by the rest; its words pass."""
    unit, text = header("c", chain)
    lines = [line.strip() for line in text[text.index("inline void c(") :].split("\n")]
    replays = [line for line in lines if line.startswith("TTI_REPLAY(")]
    assert replays == [
        "TTI_REPLAY(0, 32, 1, 1);",
        "TTI_REPLAY(0, 22, 1, 1);",
        "TTI_REPLAY(0, 32, 0, 0);",
        "TTI_REPLAY(0, 22, 0, 0);",
    ]
    play = lines.index("TTI_REPLAY(0, 32, 0, 0);")
    tail = lines[play + 1 : lines.index("#else", play)]
    assert any(line.startswith("TTI_SFPMAD(") for line in tail)
    for function in (unit.init["math"], unit.tile["math"]):
        for run in unit_module.chip_words(function):
            assert all(WORMHOLE(b, w) is None for b, w in zip([None, *run], run))


def test_the_stall_on_the_matrix_unit_is_written_by_its_names():
    """p_stall::MATH is 0x10 on Blackhole (its word unchanged) and 0x80 on Wormhole."""
    _, text = silu_header()
    stall = "TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);"
    assert stall in text and "TTI_STALLWAIT(256" not in text


def test_wormhole_refuses_the_instructions_and_modifiers_it_does_not_have():
    assert "Wormhole does not have" in WORMHOLE(None, word("SFPGT", instr_mod1=1))
    assert "NEGATE_VA" in WORMHOLE(None, word("SFPMAD", "mad", instr_mod1=1))
    assert WORMHOLE(None, word("SFPMAD", "mad", instr_mod1=0)) is None
    assert "ARG_IMM_USE_VC" in WORMHOLE(None, word("SFPSHFT", instr_mod1=5))
    assert "integer cast" in WORMHOLE(None, word("SFPCAST", instr_mod1=2))
    assert WORMHOLE(None, word("SFPLOAD", sfpu_addr_mode=7)) is None  # the base set
    assert WORMHOLE(None, word("SFPLOAD", sfpu_addr_mode=3)) is None
    assert "address mode 4" in WORMHOLE(None, word("MVMUL", "matrix", addr_mode=4))
    assert WORMHOLE(None, word("STALLWAIT", wait_res=0x10)) is None  # by its name
    assert "p_stall::MATH by its name" in WORMHOLE(
        None, word("STALLWAIT", wait_res=0x20)
    )
    assert WORMHOLE(None, word("STALLWAIT", wait_res=0x4)) is None  # C2: the same


def test_wormhole_refuses_a_result_read_the_cycle_after_it_is_not_held():
    mad = word("SFPMAD", "mad", writes=[3], instr_mod1=0)
    reader = word("SFPSTORE", "store", reads=[3], sfpu_addr_mode=3)
    assert "LReg 3, SFPMAD's result, the next cycle" in WORMHOLE(mad, reader)
    assert WORMHOLE(word("SFPNOP"), reader) is None
    lanes = word("SFPSHFT2", writes=[2], instr_mod1=3)
    assert "lane mode" in WORMHOLE(lanes, word("SFPMOV", reads=[2], instr_mod1=0))
    swap = word("SFPSWAP", writes=[0, 1], instr_mod1=1)
    assert "collide" in WORMHOLE(swap, word("SFPLOADI", instr_mod0=8))
    template = word("SFPSWAP", instr_mod1=1, lreg_dest=13)
    assert "runs it twice" in WORMHOLE(template, word("SFPIADD", instr_mod1=1))
    assert WORMHOLE(template, word("SFPNOP")) is None
    assert "replays 1 to 32" in WORMHOLE(None, word("REPLAY", len=0))
    assert WORMHOLE(None, word("REPLAY", len=32)) is None


class Gated(km.Program):
    @km.tile_function
    def gate(x, y):
        return km.silu(x) * y

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(self.gate(x, y))


class Difference(Gated):
    @km.tile_function
    def gate(x, y):
        return x - y  # -y in a load macro's template: negated at its load on Wormhole


class Tangent(Gated):
    @km.tile_function
    def gate(x, y):
        return km.tanh(x) * y  # a comparison with no register free: no Wormhole form


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def _class_words(cls):
    page = formats.tile_bytes("bfloat16")
    given = {n: Tensor(Pages(page), (2048, 32), "bfloat16", "tile") for n in "ab"}
    given["out"] = Tensor(Pages(page), (2048, 32), "bfloat16", "tile")
    words = dsl_program.tile_words(compile_class(cls, {}, given))
    ((function, runs),) = words
    assert function.startswith("gate (in run, line ") and sum(map(len, runs)) > 20
    return SimpleNamespace(name="gated", tile_words=words)


def test_a_class_s_words_load_on_blackhole_and_wormhole():
    """silu's reciprocal and comparison, a template's negated operand, in Wormhole's
    forms (hazards.wormhole)."""
    for cls in (Gated, Difference):
        program = _class_words(cls)
        kernels.tile_words_checked([program], "blackhole")
        kernels.tile_words_checked([program], "wormhole_b0")


def test_a_word_without_a_wormhole_form_is_refused_by_name():
    program = _class_words(Tangent)
    kernels.tile_words_checked([program], "blackhole")
    with pytest.raises(ProgramError, match=r"gated: tile function gate \(in run, line"):
        kernels.tile_words_checked([program], "wormhole_b0")
