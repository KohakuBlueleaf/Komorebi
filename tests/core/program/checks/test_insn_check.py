"""The check for instructions the Tensix cores lack, on ELFs the toolchain builds."""

import subprocess

import pytest

from komorebi import _env
from komorebi.device import native
from komorebi.errors import KernelInstructionError, ProgramError
from komorebi.program.checks import instructions as insn_check

# The pack thread's flags when its vector unit is on (tt-metal's blackhole HAL).
PACK_MARCH = "-march=rv32im_zmmul_zaamo_zba_zbb_xtttensixbh_zve32f"


def toolchain():
    try:
        tree = native.metal_tree()
    except Exception:  # noqa: BLE001 (no tree: the test is skipped)
        pytest.skip("needs TT_METAL_HOME")
    gcc = _env.sfpi_bin(tree) / "riscv-tt-elf-gcc"
    objdump, _ = insn_check.tools(tree)
    if not (gcc.is_file() and objdump.is_file()):
        pytest.skip(f"needs {gcc.name} and {objdump.name}")
    return tree, gcc


def build(tmp_path, gcc, source, thread="trisc2"):
    source_file = tmp_path / "kernel.c"
    source_file.write_text(source)
    elf = tmp_path / f"{thread}.elf"
    flags = ["-mcpu=tt-bh-tensix", PACK_MARCH, "-mabi=ilp32", "-O2", "-g"]
    subprocess.run(
        [str(gcc), *flags, "-c", str(source_file), "-o", str(elf)], check=True
    )
    return elf


@pytest.mark.tree
def test_a_float_divide_on_the_pack_thread_is_rejected_naming_its_function(tmp_path):
    tree, gcc = toolchain()
    elf = build(tmp_path, gcc, "float ratio(float a, float b) { return a / b; }\n")
    with pytest.raises(KernelInstructionError) as raised:
        insn_check.check_kernels(
            "p", [("compute.cpp", [str(elf)])], tree, tmp_path / "cache"
        )
    first = raised.value.findings[0]
    assert (first.thread, first.instruction) == ("TRISC2", "fdiv.s")
    assert first.function == "ratio" and first.line.endswith("kernel.c:1")
    assert "TRISC2" in str(raised.value) and "ratio" in str(raised.value)
    assert not (tmp_path / "cache").exists()  # a rejected build is not remembered


@pytest.mark.tree
def test_a_clean_kernel_passes_and_is_not_disassembled_again(tmp_path, monkeypatch):
    tree, gcc = toolchain()
    source = "int partition(int a, int b) { return a / b; }\n"
    source += "float mul(float a) { return a * a; }\n"
    elf = build(tmp_path, gcc, source)
    cache = tmp_path / "cache"
    insn_check.check_kernels("p", [("compute.cpp", [str(elf)])], tree, cache)
    assert len(list(cache.iterdir())) == 1
    monkeypatch.setattr(
        insn_check, "check_elf", lambda *arguments: pytest.fail("checked again")
    )
    insn_check.check_kernels("p", [("compute.cpp", [str(elf)])], tree, cache)


def test_the_groups_and_the_switch():
    assert insn_check.group_of("fsqrt.s") is not None
    assert insn_check.group_of("vfrec7.v") is not None
    assert insn_check.group_of("vdivu.vv") is not None
    for allowed in ("fmul.s", "fcvt.s.wu", "div", "divu", "frflags", "vfmul.vv"):
        assert insn_check.group_of(allowed) is None
    assert not insn_check.enabled({"KMR_INSN_CHECK": "0"})
    assert insn_check.enabled({})


def test_a_missing_tool_or_elf_raises_a_program_error(tmp_path):
    elf = tmp_path / "trisc2.elf"
    elf.write_bytes(b"\x7fELF")
    with pytest.raises(ProgramError, match="objdump not found"):
        insn_check.check_kernels("p", [("c.cpp", [str(elf)])], tmp_path, tmp_path)
    missing = str(tmp_path / "none.elf")
    with pytest.raises(ProgramError, match="c.cpp built no ELF"):
        insn_check.check_kernels("p", [("c.cpp", [missing])], tmp_path, tmp_path)


@pytest.mark.tree
def test_without_addr2line_a_finding_is_still_rejected(tmp_path, monkeypatch):
    tree, gcc = toolchain()
    elf = build(tmp_path, gcc, "float ratio(float a, float b) { return a / b; }\n")
    objdump, _ = insn_check.tools(tree)
    monkeypatch.setattr(insn_check, "tools", lambda name: (objdump, tmp_path / "none"))
    with pytest.raises(KernelInstructionError) as raised:
        insn_check.check_kernels("p", [("c.cpp", [str(elf)])], tree, tmp_path / "c")
    assert (raised.value.findings[0].function, raised.value.findings[0].line) == (
        "ratio",
        "",
    )


@pytest.mark.tree
def test_a_call_of_libgccs_64_bit_division_is_rejected_at_its_caller(tmp_path):
    tree, gcc = toolchain()
    source = tmp_path / "kernel.c"
    source.write_text(
        "unsigned long long rows(unsigned long long a, unsigned b) { return a / b; }\n"
        "unsigned shifted(unsigned long long a) { return a / 8; }\n"
        "void _start(void) {}\n"
    )
    elf = tmp_path / "brisc.elf"
    flags = ["-mcpu=tt-bh-tensix", PACK_MARCH, "-mabi=ilp32", "-O2", "-g"]
    link = ["-nostdlib", "-Wl,-e,_start", str(source), "-lgcc", "-o", str(elf)]
    subprocess.run([str(gcc), *flags, *link], check=True)  # linked: the call is named
    with pytest.raises(KernelInstructionError) as raised:
        insn_check.check_kernels(
            "p", [("reader.cpp", [str(elf)])], tree, tmp_path / "c"
        )
    (first,) = raised.value.findings  # by 8 a shift: no call
    assert (first.thread, first.instruction) == ("BRISC", "__udivdi3")
    assert first.group == insn_check.LIBGCC_DIVISION and first.function == "rows"
    assert first.line.endswith("kernel.c:1")
    assert "which the core lacks" not in str(raised.value)
