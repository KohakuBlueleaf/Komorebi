"""A handler whose body is hand-written kernel files (km.kernel) beside DSL handlers:
its call on each role's CPUs, the files' include lines, the pairs (km.Pair) and
constants in the header, the state a DSL handler sets at its entry (its Dst clear
built for Wormhole on the mock, tree lane), a merged build, each refusal (no card)."""

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.device import mock
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.errors import ProgramError

HAND = "kernel_hand/"  # beside this file
FILE = str(Path(__file__).resolve())


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 2048, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor():
    return Tensor(Pages(2048), (2048, 32), "bfloat16", "tile")


@km.tile_function
def plus(a, b):
    return a + b


class Mixed(km.Program):
    BLOCK = km.constant(8)
    a_in = km.Pair(km.bf16, pages=32)
    b_in = km.Pair(km.bf16, pages=BLOCK * 4)
    result = km.Pair(km.bf16, pages=32, write=True)

    @km.handler(
        km.kernel(
            movement=HAND + "movement.hpp",
            compute=HAND + "compute.hpp",
            entry="gate_k::gate",
        )
    )
    def gate(
        self, a: km.Tensor, b: km.Tensor, out: km.Tensor, tiles: int, p: km.Pattern
    ):
        """out = silu(a) * b."""

    @km.handler
    def add(
        self, a: km.Tensor, b: km.Tensor, out: km.Tensor, tiles: int, p: km.Pattern
    ):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=8, pattern=p)
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=8, pattern=p),
            km.Stream(b.pages(first, tiles), block=8, pattern=p),
        ):
            results.push(plus(x, y))


class Clean(Mixed):
    @km.handler(km.kernel(compute=HAND + "compute.hpp", clean=True))
    def gate(self, tiles: int):
        pass


class Split(km.Program):
    @km.handler(
        km.kernel(
            reader=HAND + "reader.hpp",
            writer=HAND + "movement.hpp",
            math=HAND + "math.hpp",
            unpack=HAND + "math.hpp",
        )
    )
    def split(self, a: km.Tensor, n: int, scale: float):
        """Each role its own file."""

    @km.handler(km.kernel(reader=HAND + "reader.hpp", writer=HAND + "reader.hpp"))
    def both(self):
        pass


class Gates(km.Program):
    a_in = km.Pair(km.bf16, pages=32)

    @km.handler(km.kernel(movement=HAND + "movement.hpp"))
    def gate(self, a: km.Tensor):
        pass


class Adds(km.Program):
    a_in = km.Pair(km.bf16, pages=32)

    @km.handler
    def add(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x in km.Stream(a.pages(0, 8), block=8):
            results.push(x)

    @km.handler(km.kernel(movement=HAND + "movement.hpp"))
    def again(self, a: km.Tensor):
        pass


def made(cls, **given):
    tensors = {"a": tensor(), "b": tensor(), "out": tensor()}
    return cls(log=lambda line: None, l1=1 << 20, **tensors, **given)


def case(text, handler):
    """One handler's case of a kernel source, to the next case."""
    start = text.index(f"case kHandler{handler}: {{")
    end = text.find("case kHandler", start + 1)
    return text[start : end if end > 0 else len(text)]


def def_line(name):
    lines = Path(__file__).read_text().splitlines()
    return next(
        n for n, line in enumerate(lines, 1) if line.startswith(f"    def {name}")
    )


def test_movement_is_one_call_on_both_movers():
    text = case(made(Mixed).built.files["movement.cpp"], "Gate")
    assert "COMPILE_FOR_BRISC" not in text
    assert "const A a(km_word_a);" in text and "const Out out(km_word_out);" in text
    assert f'#line {def_line("gate")} "{FILE}"' in text
    call = "if (!gate_k::gate_movement(movement, command, a, b, out, tiles, p)) {"
    assert call in text and "return false;" in text


def test_reader_and_writer_each_on_its_mover():
    text = case(made(Split).built.files["movement.cpp"], "Split")
    reader = text.index("#if defined(COMPILE_FOR_BRISC)  // mover A")
    writer = text.index("#if !defined(COMPILE_FOR_BRISC)  // mover B")
    assert reader < text.index("split_reader(movement, command, a, n, scale)") < writer
    assert writer < text.index("split_writer(movement, command, a, n, scale)")


def test_compute_is_one_call_on_the_three_cpus_without_tensors():
    text = case(made(Mixed).built.files["compute.cpp"], "Gate")
    assert "gate_k::gate_compute(compute, command, tiles, p);" in text
    assert "#if" not in text and "km_word" not in text


def test_one_compute_cpu_is_under_its_if():
    text = case(made(Split).built.files["compute.cpp"], "Split")
    math = text.index("#if defined(TRISC_MATH)")
    assert math < text.index("split_math(compute, command, n, scale);")
    assert "#if defined(TRISC_UNPACK)" in text and "TRISC_PACK" not in text


def test_files_are_included_after_the_class_header_once_under_their_cpus():
    files = made(Split).built.files
    movement = files["movement.cpp"].split("\n")
    header = movement.index('#include "split.h"')
    built = "../../" + HAND
    assert movement[header + 1 : header + 7] == [
        f'#include "{built}reader.hpp"',
        "#if !defined(COMPILE_FOR_BRISC)",
        f'#include "{built}movement.hpp"',
        "#endif",
        "",
        "void kernel_main() {",
    ]  # reader.hpp: both movers (split's reader, both's two roles)
    compute = files["compute.cpp"].split("\n")
    header = compute.index('#include "split.h"')
    assert compute[header + 1 : header + 7] == [
        "#if defined(TRISC_UNPACK)",
        f'#include "{built}math.hpp"',
        "#endif",
        "#if defined(TRISC_MATH)",
        f'#include "{built}math.hpp"',
        "#endif",
    ]
    mixed = made(Mixed).built.files["compute.cpp"].split("\n")
    header = mixed.index('#include "mixed.h"')
    assert mixed[header + 1] == f'#include "{built}compute.hpp"'


def test_declared_pairs_and_constants_in_the_header():
    program = made(Mixed)
    report = "\n".join(program.built.report)
    for pair in ("kRead0 (read, buffers 0, 1)", "kRead1 (read, buffers 2, 3)"):
        assert pair in report
    assert "kPairAIn (read, buffers 4, 5): bfloat16, 32 + 32 pages of 2048 B" in report
    assert "kPairBIn (read, buffers 6, 7)" in report
    assert "kPairResult (write, buffers 18, 19)" in report
    header = program.built.files["mixed.h"]
    assert (
        "// pair b_in: bfloat16, 32 pages a half; km.Pair: streamed by hand kernels\n"
        "constexpr km::BufferPair kPairBIn = {6, 7};\n"
        "constexpr uint32_t kPairBInPages = 32;\n"
    ) in header
    assert "kPairBInBlock" not in header and "kPairBInPattern" not in header
    assert "constexpr uint32_t kConstBlock = 8;" in header
    assert "kConstBlock = 4;" in made(Mixed, BLOCK=4).built.files["mixed.h"]
    assert "kConst" not in made(Gates).built.files["gates.h"]  # no constant


def test_dsl_handlers_set_the_startup_state_beside_a_kernel_not_clean():
    text = made(Mixed).built.files["compute.cpp"]
    add = case(text, "Add")
    assert add.index("reconfig_data_format(0, 0);") < add.index("const uint32_t tiles")
    assert "km::pack_format<16>();" in add and "pack_reconfig_data_format" not in add
    assert "TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_1, 0);" in add
    assert add.index("#if defined(TRISC_MATH)") < add.index("TTI_ZEROACC")
    assert "TTI_ZEROACC" not in case(text, "Gate")
    assert '#include "api/compute/reconfig_data_format.h"' in text
    assert '#include "komorebi/device/core/km_pack_format.hpp"' in text
    clean = made(Clean).built.files["compute.cpp"]
    assert "TTI_ZEROACC" not in clean and "reconfig_data_format" not in clean
    assert "pack_format" not in clean


def test_a_merged_build_numbers_and_includes_once():
    merged = km.merge(made(Gates), made(Adds), log=lambda line: None)
    assert merged.compiled.handlers == {"gate": 1, "add": 2, "again": 3}
    text = merged.built.files["movement.cpp"]
    assert text.count(f'#include "../../{HAND}movement.hpp"') == 1
    pair = [b for b in merged.compiled.buffers if b.name == "kPairAIn"]
    assert len(pair) == 1 and (pair[0].index_a, pair[0].index_b) == (2, 3)
    with pytest.raises(ProgramError, match=r"overlay= marks Gates.gate, a km.kernel"):
        km.merge(made(Gates), made(Adds), overlay=[Gates.gate], log=lambda line: None)


def refused(cls, **given):
    with pytest.raises(ProgramError) as raised:
        made(cls, **given)
    return str(raised.value)


def test_a_kernel_without_a_file_and_roles_given_twice_are_refused():
    class Nothing(km.Program):
        @km.handler(km.kernel())
        def gate(self):
            pass

    class Twice(km.Program):
        @km.handler(km.kernel(movement=HAND + "movement.hpp", reader=HAND + "r.hpp"))
        def gate(self):
            pass

    class Thrice(km.Program):
        @km.handler(km.kernel(compute=HAND + "compute.hpp", pack=HAND + "math.hpp"))
        def gate(self):
            pass

    assert re.match(
        r"Nothing.gate, line \d+, `@km.handler\(km.kernel\(\)\)`: "
        r"km.kernel\(...\) names no file: .*; give movement=",
        refused(Nothing),
    )
    assert "movement= is the file of mover_a, mover_b, and reader= names one of" in (
        refused(Twice)
    )
    assert "give compute= alone, or unpack= and math= and pack=" in refused(Thrice)


def test_a_file_not_there_and_handler_options_are_refused():
    class Missing(km.Program):
        @km.handler(km.kernel(movement=HAND + "absent.hpp"))
        def gate(self):
            pass

    class Overlaid(km.Program):
        @km.handler(km.kernel(movement=HAND + "movement.hpp"), overlay=True)
        def gate(self):
            pass

    absent = Path(__file__).resolve().parent / HAND / "absent.hpp"
    assert f"movement= names {absent}, which is no file" in refused(Missing)
    assert "overlay= with km.kernel: a hand kernel's handler is resident" in (
        refused(Overlaid)
    )


def test_lines_in_a_kernel_handler_and_a_constant_parameter_are_refused():
    class Lined(km.Program):
        @km.handler(km.kernel(movement=HAND + "movement.hpp"))
        def gate(self, n: int):
            km.cxx("use(n);", takes=[n])

    class Constant(km.Program):
        @km.handler(km.kernel(movement=HAND + "movement.hpp"))
        def gate(self, n: int, size: km.constant[int] = 4):
            pass

    assert re.match(
        r"Lined.gate, line \d+, `km.cxx\(\"use\(n\);\", takes=\[n\]\)`: a handler is a "
        r"kernel or lines; a hand-written line in a DSL handler is km.cxx",
        refused(Lined),
    )
    assert "size is a constant parameter: a hand kernel's call passes" in (
        refused(Constant)
    )


def test_what_a_kernel_handler_does_not_support_is_refused_at_the_make():
    class Dynamic(km.Program):
        @km.handler(km.kernel(movement=HAND + "movement.hpp"))
        def gate(self):
            pass

        @km.handler
        def visit(self):
            for _ in self.arrive():
                self.consume()

    class Chained(Gates):
        @km.chain
        def steps(self, a):
            self.gate(a)

    islands = SimpleNamespace(
        graph=SimpleNamespace(islands=[0, 1], count=lambda island: 120)
    )
    assert 'with form="runtime": its hand files are included' in refused(
        Gates, form="runtime"
    )
    assert "with chain=True: a chain's records state" in refused(Gates, chain=True)
    assert "Chained.steps: the km.chain method calls gate" in refused(Chained)
    assert "beside the dynamic handler visit" in refused(Dynamic)
    assert "in a session of 2 islands" in refused(Gates, session=islands)
    with pytest.raises(Refused, match="handlers= lists sets of a km.kernel handler"):
        made(Gates, handlers={"gate": [{}]})


def test_pairs_past_the_indices_and_two_ways_in_a_merge_are_refused():
    class Many(km.Program):
        for index in range(14):
            locals()[f"p{index}"] = km.Pair(km.bf16, pages=2)

        @km.handler(km.kernel(movement=HAND + "movement.hpp"))
        def gate(self):
            pass

    class Other(km.Program):
        a_in = km.Pair(km.fp32, pages=32)

        @km.handler
        def nothing(self):
            pass

    assert re.search(
        r"^Many: km.Pair p13 \(read\) needs pair kPairP13, and indices 0 to 25 have "
        r"no two free side by side: 13 pairs \(kPairP0 0-1 \(bfloat16, declared\);",
        refused(Many),
    )
    with pytest.raises(Refused, match=r"km.Pair a_in is \(bfloat16, pages=32, "):
        km.merge(made(Gates), made(Other), log=lambda line: None)


# A mock load on Wormhole of a merge with a hand handler (not clean) beside DSL
# handlers (test_shared_buffers.hand_merge): its compute built with the entry's lines.
WORMHOLE_LOAD = """
import sys
sys.path[:0] = sys.argv[1:]
import komorebi as km
from test_shared_buffers import hand_merge

with km.Session() as session:
    session.load(hand_merge(session=session)).stop()
"""


@pytest.mark.tree
def test_the_entry_s_clear_of_dst_builds_for_wormhole(tmp_path):
    here = Path(__file__).parent
    env = dict(os.environ, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor((0,), "wormhole_b0"))
    env[mock.MEMORY] = str(tmp_path / "kernel_buffer.json")
    env["TT_VISIBLE_DEVICES"] = "0"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(km.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, "-c", WORMHOLE_LOAD, str(here), str(here / "constants")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "TTI_ZEROACC" not in done.stdout + done.stderr, done.stderr[-3000:]
    assert done.returncode == 0, done.stderr[-3000:]
