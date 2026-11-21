"""The kernels' stack check: frames and chains of linked ELFs, overlay parts,
cycles, its refusals and warnings."""

import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from komorebi import _env
from komorebi.device import native
from komorebi.errors import DispatcherStackError, ProgramError, StackError
from komorebi.program.checks import stack
from komorebi.session.dispatch import tensix

# A stand-in listing: frames 16, 2,256 (two steps), 96, 64 down to a reset's copy, a
# tail jump to a leaf, and memset's jump into its own table.
LISTING = """
00001000 <_start>:
    1000:	addi	sp,sp,-16
    1004:	jal	1100 <kernel_main>
    1008:	addi	sp,sp,16
    100c:	ret

00001100 <kernel_main>:
    1100:	addi	sp,sp,-2032
    1104:	lui	t0,0xfffff
    1108:	addi	t0,t0,3872
    110c:	add	sp,sp,t0
    1110:	jal	1200 <reset>
    1114:	jal	1400 <memset>
    1118:	sw	sp,20(zero)
    111c:	mv	sp,s0
    1120:	ret

00001200 <reset>:
    1200:	addi	sp,sp,-96
    1204:	jal	1300 <copy>
    1208:	addi	sp,sp,96
    120c:	j	1380 <leaf>

00001300 <copy>:
    1300:	addi	sp,sp,-64
    1304:	addi	sp,sp,64
    1308:	ret

00001380 <leaf>:
    1380:	ret

00001400 <memset>:
    1400:	jalr	-96(a3)
    1404:	ret
"""
FUNCS = {
    0x1000: (16, "_start"),
    0x1100: (36, "kernel_main"),
    0x1200: (16, "reset"),
    0x1300: (12, "copy"),
    0x1380: (4, "leaf"),
    0x1400: (8, "memset"),
}
# A stand-in firmware: _start sets sp 16 B under the top, main (frame 80) calls the
# kernel through a register.
FIRMWARE = """
00003a20 <_start>:
    3a28:	lui	sp,0xffb01
    3a2c:	addi	sp,sp,-16
    3a30:	jal	3a58 <main>

00003a58 <main>:
    3a58:	addi	sp,sp,-80
    3a5c:	jalr	a5
    3a60:	addi	sp,sp,80
"""
FIRMWARE_FUNCS = {0x3A20: (20, "_start"), 0x3A58: (12, "main")}


def test_the_deepest_chain_counts_each_frame_and_a_tail_jump_frees_its_caller():
    found = stack.scan(LISTING, FUNCS)
    assert [found[at].frame for at in sorted(found)] == [16, 2256, 96, 64, 0, 0]
    need = stack.deepest(found, 0x1000)
    assert need.chain == ["_start", "kernel_main", "reset", "copy"]
    assert need.bytes == 16 + 2256 + 96 + 64


def test_the_firmware_s_frames_reach_its_call_of_the_kernel():
    found = stack.scan(FIRMWARE, FIRMWARE_FUNCS)
    before, chain = stack.firmware_frames(found, {"__stack_top": 0xFFB01000})
    assert (before, chain.chain) == (16 + 80, ["_start", "main"])


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ("    1114:\tjalr\ta5", "an indirect call"),
        ("    1114:\tsub\tsp,sp,a5", "sub sp by a register"),
        ("    1114:\tjal\t1000 <_start>", "recursion"),
        ("    1114:\tjal\t9000 <far>", "a call outside the ELF"),
    ],
)
def test_what_the_count_cannot_follow_is_refused_by_name(line, why):
    listing = LISTING.replace("    1114:\tjal\t1400 <memset>", line)
    with pytest.raises(ProgramError, match=why):
        stack.deepest(stack.scan(listing, FUNCS), 0x1000)


def test_an_indirect_call_reaches_each_function_whose_address_is_taken():
    taken = "    1110:\taddi\ta5,a5,768 # 1300 <copy>"
    listing = LISTING.replace("    1110:\tjal\t1200 <reset>", taken)
    listing = listing.replace("    1114:\tjal\t1400 <memset>", "    1114:\tjalr\ta5")
    need = stack.deepest(stack.scan(listing, FUNCS), 0x1000)
    assert need.chain == ["_start", "kernel_main", "copy"]


def test_short_names_drop_templates_parameters_and_return_types():
    assert stack._short("kernel_main()") == "kernel_main"
    assert (
        stack._short(
            "unsigned long td::Resets::run<kernel_main()::{lambda(unsigned long)#1}>"
            "(unsigned long, unsigned long) const"
        )
        == "td::Resets::run"
    )


def _margin(role, thread, room, firmware, frames):
    need = stack.Need([f"f{k}" for k in range(len(frames))], frames)
    return stack.Margin(role, thread, room, firmware, need)


def test_a_dispatcher_image_over_its_stack_is_refused_by_name(monkeypatch, tmp_path):
    over = _margin("out", "BRISC", 2144, 96, [16, 2256, 96, 80, 64])
    fits = _margin("in", "NCRISC", 2208, 144, [16, 464, 64])
    monkeypatch.setattr(stack, "margins", lambda *arguments: [fits, over])
    with pytest.raises(DispatcherStackError) as raised:
        tensix.check_stack("p", [], tmp_path, tmp_path, "wormhole_b0")
    text = str(raised.value)
    assert "the Tensix dispatcher's out needs 2,608 B of stack and has 2,144" in text
    assert "on Wormhole" in text and "the firmware's 96 B" in text
    assert raised.value.margins == [fits, over]
    monkeypatch.setattr(stack, "margins", lambda *arguments: [fits])
    assert tensix.check_stack("p", [], tmp_path, tmp_path, "blackhole") == [fits]


def test_a_margin_under_the_floor_is_refused_and_one_at_it_passes(monkeypatch):
    room = 96 + 2240 + stack.FLOOR
    at = _margin("out", "BRISC", room, 96, [2240])
    monkeypatch.setattr(stack, "margins", lambda *arguments: [at])
    assert tensix.check_stack("p", [], None, None, "wormhole_b0") == [at]
    under = _margin("out", "BRISC", room - 1, 96, [2240])
    monkeypatch.setattr(stack, "margins", lambda *arguments: [under])
    with pytest.raises(DispatcherStackError, match="15 B to spare where 16 B are"):
        tensix.check_stack("p", [], None, None, "wormhole_b0")


def test_the_dispatcher_s_images_by_role(tmp_path):
    folder = tensix.directory()
    kernels = [
        (str(folder / tensix.OUT), ["k/dispatcher/1/brisc/brisc.elf"]),
        (str(folder / tensix.IN), ["k/td_in/1/ncrisc/ncrisc.elf"]),
        (str(tmp_path / "td-thread" / "thread-ab.cpp"), ["k/t/1/trisc1/trisc1.elf"]),
        (str(tmp_path / "reader.cpp"), ["k/reader/1/ncrisc/ncrisc.elf"]),
    ]
    assert [(role, thread) for role, thread, _ in tensix.stack_images(kernels)] == [
        ("out", "BRISC"),
        ("in", "NCRISC"),
        ("thread", "TRISC1"),
    ]


def toolchain():
    try:
        tree = native.metal_tree()
    except Exception:  # noqa: BLE001 (no tree: the test is skipped)
        pytest.skip("needs TT_METAL_HOME")
    gcc = _env.sfpi_bin(tree) / "riscv-tt-elf-gcc"
    if not gcc.is_file():
        pytest.skip(f"needs {gcc.name}")
    return tree, gcc


def link(tmp_path, gcc, name, source, symbols, suffix="c"):
    path = tmp_path / f"{name}.{suffix}"
    path.write_text(source)
    elf = tmp_path / name / "brisc.elf"
    elf.parent.mkdir()
    flags = ["-mcpu=tt-wh-tensix", "-mabi=ilp32", "-O2", "-g", "-nostdlib"]
    flags += ["-fno-exceptions", "-fno-rtti"] if suffix == "cc" else []
    defined = [f"-Wl,--defsym={symbol}={value:#x}" for symbol, value in symbols]
    command = [str(gcc), *flags, "-Wl,-e,_start", *defined, str(path), "-o", str(elf)]
    subprocess.run(command, check=True)
    return elf


# A stand-in kernel whose one call has a frame over the room its data leaves, and a
# firmware that keeps 96 B of the stack before it calls the kernel.
KERNEL = """
__attribute__((noinline)) int deep(int k) {
    volatile char line[2400];
    line[k] = 1;
    return line[k + 1];
}
void _start(void) { deep(0); }
"""
STAND_IN_FIRMWARE = """
__asm__(
    ".section .start,\\"ax\\"\\n.global _start\\n.type _start,@function\\n"
    "_start:\\n lui sp,0xffb01\\n addi sp,sp,-16\\n call main\\n 1: j 1b\\n"
    ".size _start, .-_start\\n.text\\n"
);
int main(void) {
    volatile int pad[16];
    void (*kernel)(void) = *(void (**)(void))0x100;
    pad[0] = 0;
    kernel();
    return pad[0];
}
"""


@pytest.mark.tree
def test_a_stand_in_kernel_over_its_stack_is_counted_from_its_elf(tmp_path):
    tree, gcc = toolchain()
    elf = link(tmp_path, gcc, "kernel", KERNEL, [("__stack_base", 0xFFB00800)])
    firmware = link(
        tmp_path, gcc, "firmware", STAND_IN_FIRMWARE, [("__stack_top", 0xFFB01000)]
    )
    found = stack.margin("out", "BRISC", elf, tree, firmware)
    assert found.room == 0x800
    assert found.firmware >= 16 + 64  # the argument words and main's frame
    assert found.need.chain[-1].split(" ")[0] == "deep"  # _start's tail jump to it
    assert found.need.frames[-1] >= 2400
    assert found.margin < stack.FLOOR


def test_a_cycle_counts_twice_and_a_call_to_no_known_function_stays_open():
    again = LISTING.replace(
        "    1114:\tjal\t1400 <memset>", "    1114:\tjal\t1000 <_start>"
    )
    best, opened, cycles = stack.count(stack.scan(again, FUNCS), 0x1000)
    assert best.bytes == 2 * (16 + 2256) + 96 + 64 and opened is None
    assert cycles == [("kernel_main", "_start")]
    part = LISTING.replace("    1114:\tjal\t1400 <memset>", "    1114:\tjalr\ta5")
    best, opened, cycles = stack.count(stack.scan(part, FUNCS), 0x1000)
    assert (opened.chain, opened.bytes, cycles) == (["_start", "kernel_main"], 2272, [])
    taken = part.replace(
        "    1110:\tjal\t1200 <reset>", "    1110:\taddi\ta5,a5,768 # 1300 <copy>"
    )
    best, opened, cycles = stack.count(stack.scan(taken, FUNCS), 0x1000)
    assert best.chain == ["_start", "kernel_main", "copy"]  # or a part: still open
    assert (opened.chain, opened.bytes) == (["_start", "kernel_main"], 2272)
    found = stack.scan(taken, FUNCS)
    found[0x1100].ready = 1  # the debug lines' word: a wait's ready test
    best, opened, cycles = stack.count(found, 0x1000)
    assert best.chain == ["_start", "kernel_main", "copy"] and opened is None


# Two waits, each making its ready test's address and calling the one shared poll
# (km_core.hpp's SharedReady): only b's test calls something deep.
POLLS = """
00002000 <_start>:
    2000:	jal	2100 <a>
    2004:	jal	2200 <b>
    2008:	ret

00002100 <a>:
    2100:	addi	sp,sp,-16
    2104:	addi	a0,a0,1024 # 2400 <_ZN2km12ready_calledIaEEbPv>
    2108:	jal	2300 <_ZN2km16poll_live_sharedEv>
    210c:	addi	sp,sp,16
    2110:	ret

00002200 <b>:
    2200:	addi	sp,sp,-16
    2204:	addi	a0,a0,1280 # 2500 <_ZN2km12ready_calledIbEEbPv>
    2208:	jal	2300 <_ZN2km16poll_live_sharedEv>
    220c:	addi	sp,sp,16
    2210:	ret

00002300 <_ZN2km16poll_live_sharedEv>:
    2300:	addi	sp,sp,-48
    2304:	jalr	a5
    2308:	addi	sp,sp,48
    230c:	ret

00002400 <_ZN2km12ready_calledIaEEbPv>:
    2400:	addi	sp,sp,-32
    2404:	addi	sp,sp,32
    2408:	ret

00002500 <_ZN2km12ready_calledIbEEbPv>:
    2500:	addi	sp,sp,-32
    2504:	jal	2600 <deep>
    2508:	addi	sp,sp,32
    250c:	ret

00002600 <deep>:
    2600:	addi	sp,sp,-1008
    2604:	addi	sp,sp,1008
    2608:	ret
"""
POLL_FUNCS = {
    0x2000: (12, "_start"),
    0x2100: (20, "a"),
    0x2200: (20, "b"),
    0x2300: (16, "_ZN2km16poll_live_sharedEv"),
    0x2400: (12, "_ZN2km12ready_calledIaEEbPv"),
    0x2500: (16, "_ZN2km12ready_calledIbEEbPv"),
    0x2600: (12, "deep"),
}


def test_a_ready_test_s_call_reaches_the_tests_its_poll_s_caller_made():
    found = stack.scan(POLLS, POLL_FUNCS)
    found[0x2300].ready = 1  # the debug lines' word: a wait's ready test
    best, opened, _ = stack.count(found, 0x2100)
    assert best.chain[-1] == "_ZN2km12ready_calledIaEEbPv" and opened is None
    best, opened, _ = stack.count(found, 0x2000)
    assert best.chain[1:] == [
        "b", "_ZN2km16poll_live_sharedEv", "_ZN2km12ready_calledIbEEbPv", "deep"
    ] and opened is None  # fmt: skip


def test_a_ready_test_s_call_with_no_known_maker_reaches_every_test():
    found = stack.scan(POLLS, POLL_FUNCS)
    found[0x2300].ready = 1
    best, _, _ = stack.count(found, 0x2300)  # no frame above the poll
    assert best.chain[-1] == "deep"
    found[0x2100].indirect = found[0x2100].ready = 1  # a test's call outside a poll
    best, _, _ = stack.count(found, 0x2100)
    assert best.chain[-1] == "deep"


def test_a_ready_test_kept_in_memory_is_reached_from_every_poll():
    stored = POLLS.replace(
        "    2208:\tjal\t2300 <_ZN2km16poll_live_sharedEv>", "    2208:\tsw\ta0,0(a1)"
    )
    found = stack.scan(stored, POLL_FUNCS)
    assert found[0x2500].stored and not found[0x2400].stored
    found[0x2300].ready = 1
    best, _, _ = stack.count(found, 0x2100)  # a's poll may load b's kept test
    assert best.chain[-2:] == ["_ZN2km12ready_calledIbEEbPv", "deep"]


# A part that reads a word at its own slot's first address (km_overlay_entry's), then
# waits with a ready test.
SLOT_LOAD = """
00100000 <km_overlay_entry>:
    100000:\taddi\tsp,sp,-32
    100004:\tlui\tt0,0x100
    100008:\tlw\ta7,0(t0) # 100000 <km_overlay_entry>
    10000c:\tlw\ta4,0(t0)
    100010:\taddi\ta0,a0,256 # 100100 <_ZN2km12ready_calledIaEEbPv>
    100014:\tjal\t100200 <_ZN2km16poll_live_sharedEv>
    100018:\taddi\tsp,sp,32
    10001c:\tret

00100100 <_ZN2km12ready_calledIaEEbPv>:
    100100:\taddi\tsp,sp,-16
    100104:\taddi\tsp,sp,16
    100108:\tret

00100200 <_ZN2km16poll_live_sharedEv>:
    100200:\taddi\tsp,sp,-48
    100204:\tjalr\ta5
    100208:\taddi\tsp,sp,48
    10020c:\tret
"""
SLOT_FUNCS = {
    0x100000: (32, "km_overlay_entry"),
    0x100100: (12, "_ZN2km12ready_calledIaEEbPv"),
    0x100200: (16, "_ZN2km16poll_live_sharedEv"),
}


def test_a_load_from_a_function_s_address_takes_no_address_and_makes_no_cycle():
    found = stack.scan(SLOT_LOAD, SLOT_FUNCS)
    assert not found[0x100000].taken and found[0x100100].taken
    found[0x100200].ready = 1
    best, opened, cycles = stack.count(found, 0x100000)
    assert best.chain == [
        "km_overlay_entry",
        "_ZN2km16poll_live_sharedEv",
        "_ZN2km12ready_calledIaEEbPv",
    ]
    assert best.bytes == 96 and opened is None and cycles == []  # the part once


# A function f whose body (`{body}`) may take g's address.
TAKES = """
00003000 <f>:
    3000:\taddi\tsp,sp,-16
{body}
    3080:\taddi\tsp,sp,16
    3084:\tret

00003100 <g>:
    3100:\tret

00003200 <h>:
    3200:\tret
"""
TAKES_FUNCS = {0x3000: (0x88, "f"), 0x3100: (4, "g"), 0x3200: (4, "h")}


@pytest.mark.parametrize(
    ("body", "taken", "stored"),
    [
        (
            ["lui\ta5,0x3", "addi\ta5,a5,256 # 3100 <g>", "lw\ta0,0(a5)", "li\ta5,1"],
            0, 0,
        ),
        (["lui\tt0,0x3", "lw\ta7,256(t0) # 3100 <g>", "lw\ta4,0(t0)"], 0, 0),
        (["addi\ta5,a5,256 # 3100 <g>", "sw\ta5,0(a1)"], 1, 1),
        (["lui\ta5,0x3", "addi\ta5,a5,256 # 3100 <g>", "sw\ta5,4(sp)"], 1, 0),
        (["addi\ta0,a0,256 # 3100 <g>", "jal\t3200 <h>"], 1, 0),
        (["addi\ta5,a5,256 # 3100 <g>", "mv\ta4,a5", "li\ta5,0", "jalr\ta4"], 1, 0),
        (["lui\ta5,0x3", "addi\ta5,a5,256 # 3100 <g>", "jalr\ta5"], 1, 0),
        (["addi\ta5,a5,256 # 3100 <g>", "bnez\ta0,3080 <f+0x80>"], 1, 0),
        (["addi\ta5,a5,256 # 3100 <g>", "li\ta5,0", "bnez\ta0,3080 <f+0x80>"], 0, 0),
    ],
    ids=[
        "load-base", "note-on-a-load", "stored", "lui-addi-stored-in-a-frame",
        "an-argument", "moved-then-jumped-through", "lui-addi-called",
        "held-where-its-block-ends", "overwritten-first",
    ],
)  # fmt: skip
def test_an_address_is_taken_at_a_use_but_a_load_or_store_s_base(body, taken, stored):
    lines = [f"    {0x3004 + 4 * k:x}:\t{insn}" for k, insn in enumerate(body)]
    found = stack.scan(TAKES.format(body="\n".join(lines)), TAKES_FUNCS)
    assert (found[0x3100].taken, found[0x3100].stored) == (bool(taken), bool(stored))
    assert (0x3100 in found[0x3000].takes) == bool(taken)


def test_every_kernel_risc_is_an_image_but_the_dispatcher_s(tmp_path):
    folder = tensix.directory()
    kernels = [
        (str(folder / tensix.OUT), ["k/dispatcher/1/brisc/brisc.elf"]),
        (str(tmp_path / "reader.cpp"), ["k/reader/1/brisc/brisc.elf"]),
        (
            str(tmp_path / "compute.cpp"),
            [f"k/c/1/trisc{k}/trisc{k}.elf" for k in "012"],
        ),
    ]
    skip = [elf for _, _, elf in tensix.stack_images(kernels)]
    images = stack.kernel_images(kernels, skip)
    assert [(role, thread) for role, thread, _ in images] == [
        ("reader.cpp", "BRISC"),
        ("compute.cpp", "TRISC0"),
        ("compute.cpp", "TRISC1"),
        ("compute.cpp", "TRISC2"),
    ]
    parts = stack.part_images(
        {"c/kernels/reader/1/brisc/brisc.elf": ["p.elf", "p.elf"]}
    )
    assert parts == [
        ("reader with its overlay parts", "BRISC", "c/kernels/reader/1/brisc/brisc.elf",
         ("p.elf",)),
    ]  # fmt: skip


def test_a_kernel_under_the_floor_is_refused_by_name_and_under_256_b_warned():
    fits = _margin("reader.cpp", "BRISC", 2144, 96, [16, 608, 112])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert stack.judge("p", [fits], "Wormhole") == [fits]
    near = _margin("writer.cpp", "NCRISC", 1872, 144, [1088, 576])
    with pytest.warns(UserWarning, match=r"writer.cpp \(NCRISC\) 1,872 / 1,808 / 64 B"):
        stack.judge("p", [fits, near], "Wormhole")
    over = stack.Margin(
        "reader.cpp", "BRISC", 1344, 96, stack.Need(["f"], [1240]), True
    )
    with pytest.raises(StackError) as raised:
        stack.judge("p", [fits, over], "Wormhole")
    text = str(raised.value)
    assert "reader.cpp (BRISC) needs at least 1,336 B of stack and has 1,344" in text
    assert "8 B to spare where 16 B are kept" in text and "on Wormhole" in text
    cycle = stack.Margin(
        "writer.cpp", "NCRISC", 2112, 144, stack.Need(["f"], [600]), True, (("a", "b"),)
    )
    with pytest.warns(UserWarning, match="a calls b again on its own path"):
        stack.judge("p", [cycle], "Wormhole")
    near_parts = _margin("reader with its overlay parts", "BRISC", 2112, 96, [1936])
    with pytest.warns(UserWarning, match="the need an upper bound"):  # with its parts
        assert stack.judge("p", [near_parts], "Wormhole", bound=True) == [near_parts]


def test_a_kernel_with_its_parts_under_the_floor_is_refused_naming_the_part():
    chain = [
        "_start (brisck.cc:52)",
        "kernel_main (reader.cpp:14)",
        "km_overlay_entry (km_overlay_entry.hpp:98)",
        "collective (service_reader.cpp:102)",
        "km::land_serve (km_ask.hpp:3830)",
        "km::write_l1 (km_far.hpp:389)",
        "km::Core::poll_at (km_core.hpp:1269)",
    ]
    need = stack.Need(chain, [16, 656, 16, 80, 144, 208, 48])
    over = stack.Margin(
        "reader with its overlay parts", "BRISC", 1168, 96, need,
        part="the service reader part",
    )  # fmt: skip
    assert over.margin == -96
    with pytest.raises(StackError) as raised:
        stack.judge("p", [over], "Wormhole", bound=True)
    text = str(raised.value)
    assert "reader with its overlay parts (BRISC) needs 1,264 B" in text
    assert "-96 B to spare where 16 B are kept" in text
    assert "Counted with the service reader part" in text
    assert "ends km::land_serve -> km::write_l1 -> km::Core::poll_at" in text
    assert "an upper bound" in text and "(its largest, kernel_main: 656 B)" in text
    at_floor = stack.Margin("reader with its overlay parts", "BRISC", 1280, 96, need)
    assert at_floor.margin == stack.FLOOR
    with pytest.warns(UserWarning, match="the need an upper bound"):
        assert stack.judge("p", [at_floor], "Wormhole", bound=True) == [at_floor]


def test_the_load_judges_every_kernel_risc_without_a_tensix_dispatcher(monkeypatch):
    from komorebi.session.parts import load  # noqa: PLC0415

    over = stack.Margin("reader.cpp", "BRISC", 1344, 96, stack.Need(["f"], [1392]))
    monkeypatch.setattr(
        stack, "margins", lambda images, *rest: [over] if images else []
    )
    monkeypatch.setattr(native, "metal_tree", lambda tree=None: tree)
    monkeypatch.setattr(native, "cache_directory", lambda: Path("."))
    host = SimpleNamespace(_dispatcher="x280", tree=None, arch="wormhole_b0", log=print)
    kernels = [("reader.cpp", ["k/reader/1/brisc/brisc.elf"])]
    with pytest.raises(StackError, match="-144 B to spare"):
        load.LoadMixin._check_stack(host, SimpleNamespace(name="p"), kernels)
    built = SimpleNamespace(parts={"k/reader/1/brisc/brisc.elf": ["p.elf"]})
    with pytest.raises(StackError, match="-144 B to spare.* an upper bound"):
        load.LoadMixin._check_parts(
            host, [SimpleNamespace(name="p", overlays_built=built)]
        )


# A stand-in kernel that calls an overlay part through a pointer it loads (it takes no
# function's address), and the part: its entry's frame over the room.
CALLER = """
void _start(void) {
    void (*part)(void) = *(void (*volatile *)(void))0x100;
    part();
    *(volatile int *)0x104 = 1;  /* not a tail jump */
}
"""
PART = """
__attribute__((noinline)) int km_overlay_entry(int k) {
    volatile char line[2400];
    line[k] = 1;
    return line[k + 1];
}
"""


@pytest.mark.tree
def test_a_kernel_with_an_overlay_part_over_its_stack_is_counted_from_both(tmp_path):
    tree, gcc = toolchain()
    elf = link(tmp_path, gcc, "kernel", CALLER, [("__stack_base", 0xFFB00800)])
    part = link(tmp_path, gcc, "service-reader-0", PART, [])
    firmware = link(
        tmp_path, gcc, "firmware", STAND_IN_FIRMWARE, [("__stack_top", 0xFFB01000)]
    )
    alone = stack.margin("reader", "BRISC", elf, tree, firmware)
    assert alone.at_least and alone.margin > stack.FLOOR
    found = stack.margin("reader", "BRISC", elf, tree, firmware, parts=[part])
    assert not found.at_least and found.need.frames[-1] >= 2400
    assert found.need.chain[-1].split(" ")[0] == "km_overlay_entry"
    assert found.margin < stack.FLOOR and found.part == "the service reader part"


# The same call of a part from a kernel that also takes a function's address: the
# indirect call may reach that function or a part, so the part is still counted.
TAKER = """
__attribute__((noinline)) int ready(void) { return *(volatile int *)0x108; }
void _start(void) {
    *(int (*volatile *)(void))0x10c = ready;
    void (*part)(void) = *(void (*volatile *)(void))0x100;
    part();
    *(volatile int *)0x104 = 1;  /* not a tail jump */
}
"""


@pytest.mark.tree
def test_a_kernel_that_takes_a_function_s_address_is_counted_with_its_part(tmp_path):
    tree, gcc = toolchain()
    elf = link(tmp_path, gcc, "kernel", TAKER, [("__stack_base", 0xFFB00800)])
    part = link(tmp_path, gcc, "service-reader-0", PART, [])
    firmware = link(
        tmp_path, gcc, "firmware", STAND_IN_FIRMWARE, [("__stack_top", 0xFFB01000)]
    )
    found, _ = stack.read(elf, tree)
    assert [f.name for f in found.values() if f.taken] == ["ready"]
    alone = stack.margin("reader", "BRISC", elf, tree, firmware)
    assert alone.at_least and alone.margin > stack.FLOOR
    found = stack.margin("reader", "BRISC", elf, tree, firmware, parts=[part])
    assert found.need.chain[-1].split(" ")[0] == "km_overlay_entry"
    assert found.margin < stack.FLOOR and found.part == "the service reader part"


# A part that calls a part through a pointer (the runtime form's program part calling a
# handler part), its own frame beside the part it calls.
CALLING_PART = """
__attribute__((noinline)) int km_overlay_entry(int k) {
    volatile char line[600];
    int (*part)(int) = *(int (*volatile *)(int))0x100;
    line[k] = part(k);
    return line[k + 1];
}
"""


def _parts(tmp_path, *sources):
    tree, gcc = toolchain()
    elf = link(tmp_path, gcc, "kernel", CALLER, [("__stack_base", 0xFFB00800)])
    firmware = link(
        tmp_path, gcc, "firmware", STAND_IN_FIRMWARE, [("__stack_top", 0xFFB01000)]
    )
    parts = [
        link(tmp_path, gcc, f"{kind}-{k}", source, [])
        for k, (kind, source) in enumerate(sources)
    ]
    return stack.margin("writer", "NCRISC", elf, tree, firmware, parts=parts)


@pytest.mark.tree
def test_a_part_s_call_of_a_part_is_counted_with_the_part_it_calls(tmp_path):
    found = _parts(tmp_path, ("program", CALLING_PART), ("handler", SMALL_PART))
    entries = [
        i for i, name in enumerate(found.need.chain) if "km_overlay_entry" in name
    ]
    assert len(entries) == 2  # the program part's entry, then the handler part's
    assert found.need.frames[entries[0]] >= 600 and found.need.frames[-1] >= 1400
    assert found.part == "the program part" and found.margin < stack.FLOOR
    assert found.at_least and found.cycles == (("program-0", "program-0"),)
    for kind, source in (("program", CALLING_PART), ("handler", SMALL_PART)):
        (tmp_path / kind).mkdir()
        alone = _parts(tmp_path / kind, (kind, source))
        assert alone.margin >= stack.FLOOR  # each part alone fits: 600 B or 1,400 B


@pytest.mark.tree
def test_a_part_that_may_call_itself_is_a_cycle_counted_once(tmp_path):
    found = _parts(tmp_path, ("program", CALLING_PART))
    assert found.at_least and found.cycles == (("program-0", "program-0"),)
    assert sum("km_overlay_entry" in name for name in found.need.chain) == 1


# A kernel whose deep wait calls its ready test through a pointer (km::SharedReady, as
# km_core.hpp), and with WITH_PART a shallow call of a part (km::Overlay's call).
WAITER = """
namespace km {
struct SharedReady {
    bool (*test)(void*);
    void* ready;
    bool operator()() const { return test(ready); }
};
__attribute__((noinline)) void poll(SharedReady ready) {
    volatile char line[600];
    line[0] = 0;
    while (!ready()) {
        line[1] = 1;
    }
}
template <typename Kernel> struct Overlay {
    unsigned call() {
        return reinterpret_cast<unsigned (*)()>(*(unsigned* volatile*)0x100)();
    }
};
}  // namespace km
__attribute__((noinline)) bool held(void* at) { return *(volatile int*)at != 0; }
extern "C" void _start() {
    km::poll({held, (void*)0x108});
#if WITH_PART
    km::Overlay<int>().call();
#endif
    *(volatile int*)0x104 = 1;
}
"""
# A part deeper than the wait: it fits at the kernel's call of it, not below the wait.
SMALL_PART = PART.replace("2400", "1400")


def _waiter(tmp_path, with_part):
    tree, gcc = toolchain()
    source = f"#define WITH_PART {int(with_part)}\n{WAITER}"
    elf = link(tmp_path, gcc, "kernel", source, [("__stack_base", 0xFFB00800)], "cc")
    part = link(tmp_path, gcc, "service-reader-0", SMALL_PART, [])
    firmware = link(
        tmp_path, gcc, "firmware", STAND_IN_FIRMWARE, [("__stack_top", 0xFFB01000)]
    )
    alone = stack.margin("reader", "BRISC", elf, tree, firmware)
    return alone, stack.margin("reader", "BRISC", elf, tree, firmware, parts=[part])


@pytest.mark.tree
def test_a_wait_s_ready_test_reaches_the_taken_functions_and_no_part(tmp_path):
    alone, found = _waiter(tmp_path, with_part=False)
    assert not alone.at_least and "km::poll" in alone.need.text()
    assert not found.at_least and found.part == ""
    assert found.need.bytes == alone.need.bytes


@pytest.mark.tree
def test_a_part_s_call_is_counted_at_its_own_site_not_below_a_wait(tmp_path):
    alone, found = _waiter(tmp_path, with_part=True)
    assert alone.at_least
    assert found.need.chain[-1].split(" ")[0] == "km_overlay_entry"
    assert "km::poll" not in found.need.text()  # the part at _start's call
    assert found.part == "the service reader part" and found.margin >= stack.FLOOR
