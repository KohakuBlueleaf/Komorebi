"""Builds overlays: compiled with the persistent kernel's own compile command (logged by
the runtime's JIT build), linked for a fixed L1 slot against the persistent kernel's
ELF with --just-symbols, checked (entry, no relocations, inside the slot, no call into
the persistent kernel's text) and cut to the bytes that are loaded."""

import hashlib
import random
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LDM = (0xFFB00000, 0xFFB00000 + 8 * 1024)  # each core's own data memory and stack
BUILD_DIR = "{build}"  # the persistent kernel's JIT build folder of this core
ALIGN = 64  # the loaded image is padded to whole DRAM words of this size
DROP_COMPILE = ("-MMD", "-c")  # flags of the runtime's compile that name its files
DROP_LINK_PREFIX = ("-Wl,--just-symbols=", "-T", "-Wl,--emit-relocs")
SCRIPT = """\
OUTPUT_ARCH(riscv)
ENTRY(ov_entry)
MEMORY {{ SLOT (rwx) : ORIGIN = {slot:#x}, LENGTH = {size:#x} }}
SECTIONS
{{
  .text : {{ KEEP(*(.ov_entry)) *(.text .text.*) }} > SLOT
  .rodata : {{ *(.rodata .rodata.* .srodata .srodata.*) }} > SLOT
  .data : {{ *(.data .data.* .sdata .sdata.*) . = ALIGN(4); }} > SLOT
  .bss (NOLOAD) : {{
    __ov_bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(4);
    __ov_bss_end = .;
  }} > SLOT
  /DISCARD/ : {{ *(.eh_frame*) *(.note*) *(.init_array*) *(.fini_array*) }}
}}
"""


class OverlayError(ValueError):
    """An overlay that would not run as built: refused before it is loaded."""


@dataclass
class Commands:
    compile: list  # the runtime's compile argv for the persistent kernel's core
    link: list  # its link argv, without the runtime's files
    tool: str  # the compiler driver; binutils sit beside it


@dataclass
class Overlay:
    name: str
    image: bytes  # the bytes loaded at the slot, padded to ALIGN
    slot: int
    text: int  # bytes of .text
    bss: tuple  # (start, end) zeroed by the entry
    elf: Path


def parse_commands(text, kernel, target):
    """The compile and link commands of `kernel` (its JIT output folder name) for one
    core (`target`: brisc, trisc0, ...) from the runtime's logged build commands. The
    include folders relative to the runtime's build folder become BUILD_DIR."""
    comp = link = None
    for line in text.splitlines():
        if f"/kernels/{kernel}/" not in line or f"/{target}/" not in line:
            continue
        if "g++ compile cmd:" in line:
            argv = shlex.split(
                line.split("g++ compile cmd:", 1)[1].split(" (build.cpp")[0]
            )
            comp = []
            skip = False
            for a in argv:
                if skip:
                    skip = False
                elif a in ("-o", "-MF"):
                    skip = True
                elif (
                    a in DROP_COMPILE
                    or a.startswith("-DFULL_KERNEL_NAME=")
                    or a.endswith((".cc", ".cpp"))
                ):
                    continue
                elif a == "-I.":
                    comp.append("-I" + BUILD_DIR)
                elif a == "-I..":
                    comp.append("-I" + BUILD_DIR + "/..")
                else:
                    comp.append(a)
        elif "g++ link cmd:" in line:
            body = line.split("g++ link cmd:", 1)[1].split(" (build.cpp")[0]
            argv = shlex.split(body.split("&&", 1)[1])
            link = []
            skip = False
            for a in argv:
                if skip:
                    skip = False
                elif a == "-o":
                    skip = True
                elif a.startswith(DROP_LINK_PREFIX) or a.endswith(".o"):
                    continue
                else:
                    link.append(a)
    if comp is None or link is None:
        raise OverlayError(f"no compile and link command of kernel {kernel!r} logged")
    return Commands(comp, link, comp[0])


def tool(commands, name):
    return commands.tool.replace("riscv-tt-elf-g++", f"riscv-tt-elf-{name}")


def check_slot(slot, size, reserved):
    """Refuses a slot [slot, slot + size) that overlaps a reserved (name, lo, hi)
    range, is not 64-byte aligned, or lies in a core's local data memory."""
    if slot % ALIGN or size % ALIGN or size <= 0:
        raise OverlayError(f"slot {slot:#x} + {size:#x}: not {ALIGN}-byte aligned")
    if slot < LDM[1] and LDM[0] < slot + size:
        raise OverlayError(f"slot {slot:#x} is in the local data memory")
    for name, lo, hi in reserved:
        if slot < hi and lo < slot + size:
            raise OverlayError(
                f"slot [{slot:#x}, {slot + size:#x}) overlaps {name} [{lo:#x}, {hi:#x})"
            )


def chain_steps(steps, seed, last=None):
    """The generated steps (k, r, k2) of a chain body; `last` replaces the final k2
    (two bodies that differ in one late instruction)."""
    rng = random.Random(seed)
    out = [
        (rng.getrandbits(32), rng.randrange(1, 32), rng.getrandbits(32))
        for _ in range(steps)
    ]
    if last is not None:
        k, r, _ = out[-1]
        out[-1] = (k, r, last)
    return out


def chain_header(steps):
    lines = [
        "// Generated by builder.py: the steps of one chain overlay.",
        "#pragma once",
    ]
    body = " ".join(f"S({k:#010x}u, {r}u, {k2:#010x}u)" for k, r, k2 in steps)
    lines.append(f"#define OV_STEPS(S) {body}")
    return "\n".join(lines) + "\n"


def mix(x):
    return (((x << 7) | (x >> 25)) & 0xFFFFFFFF) ^ 0x5A5A5A5A


def chain_result(steps, arg):
    """The host's reference of a chain overlay's result for `arg`."""
    x = arg ^ 1
    for k, r, k2 in steps:
        y = x ^ k
        x = ((((y << r) | (y >> (32 - r))) & 0xFFFFFFFF) + k2) & 0xFFFFFFFF
    return mix(x)


def run(argv, cwd=None):
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if done.returncode:
        raise OverlayError(f"{argv[0]} failed:\n{done.stdout}{done.stderr}")
    return done.stdout


def symbols(commands, elf):
    """{name: (value, section index or 'ABS'/'UND', type, binding)} of an ELF's
    symbols."""
    out = {}
    for line in run([tool(commands, "readelf"), "-sW", str(elf)]).splitlines():
        f = line.split()
        if len(f) == 8 and f[0].endswith(":") and f[0] != "Num:":
            out[f[7]] = (int(f[1], 16), f[6], f[3], f[4])
    return out


def loader_text(commands, elf):
    """The persistent kernel's .text range at its link address: the runtime loads it
    at another address (execute in place), so an overlay must not refer into it."""
    for line in run([tool(commands, "readelf"), "-SW", str(elf)]).splitlines():
        f = line.replace("[ ", "[").split()
        if len(f) > 5 and f[1] == ".text":
            at, size = int(f[3], 16), int(f[5], 16)
            return at, at + size
    raise OverlayError(f"{elf}: no .text")


def build(commands, name, source, header, slot, size, loader_elf, out, resets=()):
    """Compiles `source` (with `header` written as overlay_gen.h), links it
    for [slot, slot + size) against the persistent kernel's ELF; returns an Overlay."""
    key = hashlib.sha256(
        f"{commands}{source.read_bytes()}{header}{slot}{size}".encode()
        + Path(loader_elf).read_bytes()
        + (HERE / "overlay_entry.h").read_bytes()
        + (HERE / "overlay.h").read_bytes()
    ).hexdigest()[:12]
    work = Path(out) / f"{name}-{key}"
    work.mkdir(parents=True, exist_ok=True)
    if header is not None:
        (work / "overlay_gen.h").write_text(header)
    obj, elf, script = work / "body.o", work / "overlay.elf", work / "overlay.ld"
    script.write_text(SCRIPT.format(slot=slot, size=size))
    if not elf.exists():
        build_dir = str(Path(loader_elf).parent)
        flags = [a.replace(BUILD_DIR, build_dir) for a in commands.compile]
        run(
            [*flags, f"-I{work}", f"-I{HERE}", "-c", str(source), "-o", str(obj)],
            cwd=work,
        )
        undefined = run([tool(commands, "gcc-nm"), "-u", str(obj)]).split()
        run(
            [
                *commands.link,
                f"-T{script}",
                f"-Wl,--just-symbols={loader_elf}",
                f"-Wl,-Map={work / 'overlay.map'}",
                str(obj),
                "-o",
                str(elf),
            ],
            cwd=work,
        )
        (work / "undefined.txt").write_text("\n".join(u for u in undefined if u != "U"))
    return check(commands, name, elf, slot, size, loader_elf, work, resets)


def check(commands, name, elf, slot, size, loader_elf, work, resets=()):
    """Checks a linked overlay and cuts its image (see the module's docstring)."""
    header = run([tool(commands, "readelf"), "-hlSW", str(elf)])
    if "Type:                              EXEC" not in header:
        raise OverlayError(f"{name}: not an executable")
    entry = int(re.search(r"Entry point address:\s+(0x[0-9a-f]+)", header).group(1), 16)
    if entry != slot:
        raise OverlayError(f"{name}: entry {entry:#x} is not the slot's first word")
    if re.search(r"\]\s+\.rela?\.", header) or ".dynamic" in header:
        raise OverlayError(f"{name}: relocations left")
    syms = symbols(commands, elf)
    if syms.get("ov_entry", (None,))[0] != slot:
        raise OverlayError(f"{name}: ov_entry is not at the slot")
    if any(sec == "UND" and n for n, (_, sec, _, _) in syms.items()):
        raise OverlayError(f"{name}: undefined symbols left")
    lo, hi = loader_text(commands, loader_elf)
    theirs = symbols(commands, loader_elf)
    own = [
        n
        for n, (at, sec, kind, _) in syms.items()
        if kind == "OBJECT" and sec not in ("ABS", "UND") and slot <= at < slot + size
    ]
    # A persistent kernel's global defined again is a second copy of its state; a
    # file-static one (a header's cache) must be among the `resets` the entry redoes.
    twice = [
        n
        for n in own
        if theirs.get(n, (0, "UND"))[1] != "UND"
        and (theirs[n][3] != "LOCAL" or n not in resets)
    ]
    if twice:
        raise OverlayError(
            f"{name}: its own copy of the persistent kernel's state: {twice[:8]}"
        )
    used = (work / "undefined.txt").read_text().split()
    for u in used:
        at = theirs.get(u, (None,))[0]
        if at is not None and lo <= at < hi:
            raise OverlayError(
                f"{name}: refers to {u}, in the persistent kernel's text (loaded "
                "elsewhere at run time): pass it as a pointer instead"
            )
    loads = re.findall(
        r"LOAD\s+0x[0-9a-f]+ (0x[0-9a-f]+) 0x[0-9a-f]+ (0x[0-9a-f]+) (0x[0-9a-f]+)",
        header,
    )
    end = slot
    for vaddr, filesz, memsz in loads:
        v, m = int(vaddr, 16), int(memsz, 16)
        if v < slot or v + m > slot + size:
            raise OverlayError(
                f"{name}: loaded range [{v:#x}, {v + m:#x}) outside the slot"
            )
        end = max(end, v + int(filesz, 16))
    bss = (syms["__ov_bss_start"][0], syms["__ov_bss_end"][0])
    if bss[0] < slot or bss[1] > slot + size:
        raise OverlayError(f"{name}: .bss outside the slot")
    binary = work / "overlay.bin"
    run([tool(commands, "objcopy"), "-O", "binary", str(elf), str(binary)])
    image = binary.read_bytes()[: end - slot]
    image += bytes(-len(image) % ALIGN)
    text = int(re.search(r"\.text\s+PROGBITS\s+\S+\s+\S+\s+(\S+)", header).group(1), 16)
    return Overlay(name, image, slot, text, bss, elf)
