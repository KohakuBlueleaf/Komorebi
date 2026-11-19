"""The dispatcher's stack check: frames and chains of linked ELFs, and its refusal."""

import subprocess

import pytest

from komorebi import _env
from komorebi.device import native
from komorebi.errors import DispatcherStackError, ProgramError
from komorebi.program.checks import stack
from komorebi.session.dispatch import tensix

# A stand-in listing: _start calls kernel_main, which calls a reset (frames 16, 2,256
# with a frame over 2,047 B, 96, 64) and tail-jumps to a leaf; memset jumps into its
# own table.
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


def link(tmp_path, gcc, name, source, symbols):
    path = tmp_path / f"{name}.c"
    path.write_text(source)
    elf = tmp_path / name / "brisc.elf"
    elf.parent.mkdir()
    flags = ["-mcpu=tt-wh-tensix", "-mabi=ilp32", "-O2", "-g", "-nostdlib"]
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
