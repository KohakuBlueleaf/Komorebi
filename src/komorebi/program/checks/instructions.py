"""Rejects a kernel whose linked ELF holds an instruction its Tensix core lacks (a core
that meets one stops without a report). Each group is documented as absent for the
Blackhole baby RISC-V cores; the build allows them where the pack thread's vector
flags imply the F extension."""

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from komorebi import _env
from komorebi.errors import KernelInstructionError, ProgramError

SWITCH = "KMR_INSN_CHECK"  # "0" turns the check off

# (group, pattern on the mnemonic). Not rejected, though documented as absent: access
# to fflags and frm (the raster's SETUP pass uses it on main and runs).
GROUPS = (
    ("float divide or square root", re.compile(r"f(div|sqrt)\.[sh]$")),
    ("vector integer divide or remainder", re.compile(r"v(div|rem)u?\.")),
    (
        "vector float divide, square root or estimate",
        re.compile(r"v(fr?div|fsqrt|frsqrt7|frec7)\."),
    ),
)
THREADS = {
    "brisc": "BRISC",
    "ncrisc": "NCRISC",
    "trisc0": "TRISC0",
    "trisc1": "TRISC1",
    "trisc2": "TRISC2",
}
_LABEL = re.compile(r"^[0-9a-f]+ <(.+)>:$")
_INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)")


@dataclass
class Finding:
    """One instruction the core lacks: where and what."""

    thread: str
    source: str
    instruction: str
    group: str
    function: str
    line: str
    address: int


def enabled(environ=None):
    """False when KMR_INSN_CHECK is "0"."""
    return (os.environ if environ is None else environ).get(SWITCH, "1") != "0"


def group_of(mnemonic):
    """The name of the absent group `mnemonic` belongs to, else None."""
    for name, pattern in GROUPS:
        if pattern.match(mnemonic):
            return name
    return None


def scan(listing):
    """(address, mnemonic, group, label) for every absent instruction in an objdump
    -d listing; label is the nearest symbol before it that is not a local label."""
    out, label = [], "?"
    for text in listing.splitlines():
        found = _LABEL.match(text)
        if found:
            if not found.group(1).startswith("."):
                label = found.group(1)
            continue
        found = _INSN.match(text)
        if found:
            group = group_of(found.group(2))
            if group:
                out.append((int(found.group(1), 16), found.group(2), group, label))
    return out


def tools(tree):
    """The toolchain's objdump and addr2line (those that built the kernels)."""
    bin_dir = _env.sfpi_bin(tree)
    return bin_dir / "riscv-tt-elf-objdump", bin_dir / "riscv-tt-elf-addr2line"


def where(addr2line, elf, address):
    """(function, file:line) from the ELF's debug info, or (None, "") (also without
    addr2line: a finding is reported without its line rather than hidden)."""
    try:
        run = subprocess.run(
            [str(addr2line), "-f", "-C", "-e", str(elf), f"{address:#x}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, ""
    lines = run.stdout.splitlines()
    if run.returncode or len(lines) < 2:
        return None, ""
    function = lines[0] if lines[0] != "??" else None
    line = lines[1] if not lines[1].startswith("??") else ""
    return function, line.split(" (discriminator")[0]


def check_elf(elf, source, tree):
    """The findings of one thread's ELF."""
    objdump, addr2line = tools(tree)
    if not objdump.is_file():
        raise ProgramError(
            f"{objdump} not found: the instruction check needs the toolchain that "
            f"built the kernels ({SWITCH}=0 turns the check off)"
        )
    run = subprocess.run(
        [str(objdump), "-d", "--no-show-raw-insn", str(elf)],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode:
        raise ProgramError(f"{objdump.name} failed on {elf}: {run.stderr.strip()}")
    thread = THREADS.get(Path(elf).stem, Path(elf).stem)
    out = []
    for address, mnemonic, group, label in scan(run.stdout):
        function, line = where(addr2line, elf, address)
        out.append(
            Finding(thread, source, mnemonic, group, function or label, line, address)
        )
    return out


def _marker(cache, elf):
    status = Path(elf).stat()
    key = f"{Path(elf).resolve()}|{status.st_size}|{status.st_mtime_ns}"
    # a changed check checks again
    key += f"|{_env.code_key('program/checks/instructions.py')}"
    return Path(cache) / hashlib.sha256(key.encode()).hexdigest()[:32]


def check_kernels(program, kernels, tree, cache):
    """Checks every ELF of `kernels` ([(source, [elf paths])]); raises
    KernelInstructionError on the first program with findings. A clean ELF leaves a
    marker in `cache`, so that the same build is not disassembled again."""
    findings = []
    for source, elves in kernels:
        for elf in elves:
            if not elf:
                continue
            if not Path(elf).is_file():
                raise ProgramError(f"{program}: {source} built no ELF at {elf}")
            marker = _marker(cache, elf)
            if marker.exists():
                continue
            found = check_elf(elf, Path(source).name, tree)
            if found:
                findings += found
            else:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
    if findings:
        raise KernelInstructionError(program, findings)
