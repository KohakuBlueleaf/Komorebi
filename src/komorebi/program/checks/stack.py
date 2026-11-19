"""The stack a Tensix kernel's linked ELF needs on its RISC and the stack it has: the
frames of its deepest call chain from _start, after the firmware's own frames up to
its call of the kernel (docs/reference/host.md, "The dispatcher's stack"); the tt-1xx
firmware enables no interrupt, so nothing else lands on that stack."""

import bisect
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from komorebi import _env
from komorebi.errors import ProgramError

# The least margin a dispatcher image keeps once its need (the firmware's frames,
# then its deepest chain) is counted: one aligned frame for what the count misses.
FLOOR = 16
# Functions that hand control to the kernel without an indirect call.
HANDOVERS = ("wh_iram_trampoline",)
# Library leaves whose indirect jump lands inside themselves (newlib's assembly
# memset jumps into its own table): no frame, no callee.
LEAVES = ("memset", "memcpy")
_INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
_TARGET = re.compile(r"^([0-9a-f]+)\b")
_NAMED = re.compile(r"#\s*([0-9a-f]+) <[^>+]+>$")  # objdump's note of an address
THREADS = {
    "brisc": "BRISC",
    "ncrisc": "NCRISC",
    "trisc0": "TRISC0",
    "trisc1": "TRISC1",
    "trisc2": "TRISC2",
}
_SP_LOADS = ("lw", "mv", "addi")
_STORES = ("sw", "sh", "sb")  # sp restored from a register or memory: no frame


@dataclass
class Function:
    """One function of an ELF: its frame, its callees, what the count cannot follow.
    `sets_sp` is the absolute sp it sets (a firmware's _start), else None; `taken`
    when an instruction other than a call takes its address."""

    name: str
    frame: int = 0
    calls: set = field(default_factory=set)
    tails: set = field(default_factory=set)
    indirect: int = 0
    unknown: list = field(default_factory=list)
    sets_sp: int | None = None
    taken: bool = False


@dataclass
class Need:
    """The stack one image needs: its deepest chain (names, frames) and its bytes."""

    chain: list
    frames: list

    @property
    def bytes(self):
        return sum(self.frames)

    def text(self):
        """The chain, each function with its frame."""
        return " -> ".join(
            f"{name}: {frame:,} B" for name, frame in zip(self.chain, self.frames)
        )


@dataclass
class Margin:
    """One dispatcher image: its role and RISC, the stack it has (`room`), the
    firmware's frames before it, and its own need."""

    role: str
    thread: str
    room: int
    firmware: int
    need: Need

    @property
    def needed(self):
        return self.firmware + self.need.bytes

    @property
    def margin(self):
        return self.room - self.needed

    def line(self):
        return f"{self.role} ({self.thread}) {self.margin:,} B"


def functions(table):
    """{address: (size, name)} of the FUNC symbols of readelf -sW's `table`."""
    out = {}
    for line in table.splitlines():
        words = line.split()
        sized = len(words) >= 8 and words[2].isdigit() and int(words[2]) > 0
        if sized and words[3] == "FUNC":
            out[int(words[1], 16)] = (int(words[2]), words[7])
    return out


def symbols(table):
    """{name: value} of every named symbol of readelf -sW's `table`."""
    out = {}
    for line in table.splitlines():
        words = line.split()
        if len(words) >= 8 and words[0].endswith(":") and words[0][:-1].isdigit():
            out[words[7]] = int(words[1], 16)
    return out


def _immediate(text):
    try:
        return int(text, 0)
    except ValueError:
        return None


def scan(listing, funcs):
    """{address: Function} of objdump -d's `listing`, over `funcs` (functions())."""
    starts = sorted(funcs)
    out = {at: Function(funcs[at][1]) for at in starts}
    ends = {at: at + funcs[at][0] for at in starts}
    current, run, constants = None, 0, {}
    for line in listing.splitlines():
        found = _INSN.match(line)
        if not found:
            continue
        pc, op = int(found.group(1), 16), found.group(2)
        if current is None or not current <= pc < ends[current]:
            k = bisect.bisect_right(starts, pc) - 1
            current = starts[k] if k >= 0 and pc < ends[starts[k]] else None
            run, constants = 0, {}
            if current is None:
                continue
        function = out[current]
        words = [w.strip() for w in found.group(3).split("#")[0].split(",")]
        if op == "lui" and words[0] == "sp":
            function.sets_sp = (_immediate(words[1]) or 0) << 12
        elif op == "addi" and words[:2] == ["sp", "sp"]:
            step = _immediate(words[2]) or 0
            if function.sets_sp is not None and function.frame == 0:
                function.sets_sp += step
            else:
                run -= step
                function.frame = max(function.frame, run)
        elif op == "addi" and len(words) == 3 and words[0] == words[1]:
            if words[0] in constants:  # lui then addi: a constant of 32 bits
                constants[words[0]] += _immediate(words[2]) or 0
        elif op in ("li", "lui") and len(words) == 2:
            value = _immediate(words[1])
            if value is not None and op == "lui":  # 20 bits, signed
                value = (value - (1 << 20) if value >> 19 else value) << 12
            if value is not None:
                constants[words[0]] = value
        elif op in ("add", "sub") and words[:2] == ["sp", "sp"]:
            step = constants.get(words[2])
            if step is None:
                function.unknown.append(f"{op} sp by a register at {pc:#x}")
            else:
                run -= step if op == "add" else -step
                function.frame = max(function.frame, run)
        elif words and words[0] == "sp" and op not in _SP_LOADS and op[0] != "s":
            function.unknown.append(f"{op} {found.group(3).strip()} at {pc:#x}")
        elif op == "jal" and (len(words) == 1 or words[0] == "ra"):
            target = _TARGET.match(words[-1])
            if target:
                function.calls.add(int(target.group(1), 16))
        elif op == "j":
            target = _TARGET.match(words[0])
            at = int(target.group(1), 16) if target else None
            if at in out and at != current:  # another function's start: a tail call
                function.tails.add(at)
        elif op == "jalr" and words[0] not in ("zero", "x0"):
            function.indirect += 1
        named = _NAMED.search(found.group(3))
        if named and op[0] not in "bj" and int(named.group(1), 16) in out:
            out[int(named.group(1), 16)].taken = True  # its address in a register
        kept = op in ("li", "lui", *_STORES) or op[0] in "bj"
        if not kept and words and words[:2] != [words[0], words[0]]:
            constants.pop(words[0], None)  # a register written otherwise
    return out


def deepest(found, entry, stop=None):
    """The Need of the deepest chain from `entry` over `found` (scan()), an indirect
    call reaching each function whose address is taken; with `stop` (a predicate), the
    deepest chain ending at a function it holds, or None. ProgramError for recursion,
    a call outside the ELF, an indirect call with no such function, sp by a register."""

    def walk(at, path):
        function = found[at]
        if function.unknown:
            raise ProgramError(f"{function.name}: {function.unknown[0]}")
        calls = set(function.calls)
        if function.indirect and stop is None and function.name not in LEAVES:
            if not taken:
                raise ProgramError(f"{function.name}: an indirect call")
            calls |= taken
        best = Need([function.name], [function.frame]) if stop is None else None
        if stop is not None and stop(function):
            best = Need([function.name], [function.frame])
        for target, tail in [(t, False) for t in calls] + [
            (t, True) for t in function.tails
        ]:
            if target not in found:
                raise ProgramError(f"{function.name}: a call outside the ELF")
            if target in path:
                raise ProgramError(f"{function.name}: recursion")
            below = walk(target, (*path, target))
            if below is None:
                continue
            if tail:
                chain = below
            else:
                chain = Need(
                    [function.name, *below.chain], [function.frame, *below.frames]
                )
            if best is None or chain.bytes > best.bytes:
                best = chain
        return best

    taken = {at for at, function in found.items() if function.taken}
    return walk(entry, (entry,))


def _entry(found):
    for at, function in found.items():
        if function.name == "_start":
            return at
    raise ProgramError("no _start")


def _tool(tree, name):
    path = _env.sfpi_bin(tree) / f"riscv-tt-elf-{name}"
    if not path.is_file():
        raise ProgramError(
            f"{path} not found: the dispatcher's stack check needs the toolchain "
            "that built the kernels"
        )
    return path


def _run(tool, *args):
    run = subprocess.run(
        [str(tool), *args], capture_output=True, text=True, check=False
    )
    if run.returncode:
        raise ProgramError(f"{tool.name} failed: {run.stderr.strip()}")
    return run.stdout


def read(elf, tree):
    """(scan(), symbols()) of `elf`, by the toolchain of `tree`."""
    table = _run(_tool(tree, "readelf"), "-sW", str(elf))
    listing = _run(_tool(tree, "objdump"), "-d", "--no-show-raw-insn", str(elf))
    return scan(listing, functions(table)), symbols(table)


def _short(name):
    """A demangled name without its template arguments, parameters and return type:
    "unsigned long td::Resets::run<...>(...) const" is "td::Resets::run"."""
    out, depth = "", 0
    for char in name:
        depth += char == "<"
        if not depth:
            out += char
        depth -= char == ">" and depth > 0
    if out.endswith(" const"):
        out = out[: -len(" const")]
    if out.endswith(")"):  # its parameters: from the last top-level "("
        depth = 0
        for at in range(len(out) - 1, -1, -1):
            depth += (out[at] == ")") - (out[at] == "(")
            if depth == 0:
                out = out[:at]
                break
    depth, cut = 0, 0
    for at, char in enumerate(out):  # a return type: before the last top-level space
        depth += (char in "({") - (char in ")}")
        if char == " " and depth == 0:
            cut = at + 1
    return out[cut:]


def named(found, need, elf, tree):
    """`need` with each function as its short demangled name and its source line
    (addr2line of the toolchain of `tree`; the raw name when it says nothing)."""
    at = {function.name: address for address, function in found.items()}
    addresses = [f"{at[name]:#x}" for name in need.chain]
    try:
        lines = _run(
            _tool(tree, "addr2line"), "-f", "-C", "-e", str(elf), *addresses
        ).splitlines()
    except (ProgramError, OSError):
        return need
    chain = []
    for k, name in enumerate(need.chain):
        function, line = (lines[2 * k : 2 * k + 2] + ["??", "??"])[:2]
        text = _short(function) if function != "??" else name
        line = Path(line.split(" (discriminator")[0]).name
        chain.append(f"{text} ({line})" if not line.startswith("??") else text)
    return Need(chain, need.frames)


def handover(function):
    """True for a firmware function that hands control to the kernel."""
    return function.indirect > 0 or function.name in HANDOVERS


def firmware_frames(found, names):
    """(the bytes on the stack at the kernel's entry, the firmware's chain to it):
    from the top to the sp its _start sets, then the deepest chain to a handover."""
    top = names.get("__stack_top")
    if top is None:
        raise ProgramError("the firmware has no __stack_top")
    entry = _entry(found)
    start = found[entry]
    reserved = top - start.sets_sp if start.sets_sp is not None else 0
    chain = deepest(found, entry, handover)
    if chain is None:
        raise ProgramError("no call of the kernel in the firmware")
    return reserved + chain.bytes, chain


def firmware_elf(elf):
    """The firmware ELF of the RISC a kernel ELF of tt-metal's cache was built for:
    <cache>/kernels/<name>/<hash>/<risc>/<risc>.elf beside <cache>/firmware."""
    elf = Path(elf)
    return elf.parents[4] / "firmware" / elf.parent.name / elf.name


def margin(role, thread, elf, tree, firmware=None):
    """The Margin of kernel `elf` (role, RISC name) against its firmware's ELF."""
    firmware = Path(firmware or firmware_elf(elf))
    if not firmware.is_file():
        raise ProgramError(f"{role}: no firmware ELF at {firmware}")
    try:
        found, names = read(elf, tree)
        need = named(found, deepest(found, _entry(found)), elf, tree)
    except ProgramError as error:
        raise ProgramError(
            f"{role} ({thread}): its stack is not counted: {error}"
        ) from error
    try:
        fw_found, fw_names = read(firmware, tree)
        before, _ = firmware_frames(fw_found, fw_names)
    except ProgramError as error:
        raise ProgramError(f"{role} ({thread}) firmware {firmware}: {error}") from error
    base = names.get("__stack_base")
    if base is None:
        raise ProgramError(f"{role} ({thread}): {elf} has no __stack_base")
    return Margin(role, thread, fw_names["__stack_top"] - base, before, need)


def _marker(cache, elf, firmware):
    key = ""
    for path in (Path(elf), Path(firmware)):
        status = path.stat()
        key += f"{path.resolve()}|{status.st_size}|{status.st_mtime_ns}|"
    key += _env.code_key("program/checks/stack.py")  # a changed check counts again
    return Path(cache) / hashlib.sha256(key.encode()).hexdigest()[:32]


def margins(images, tree, cache):
    """[Margin] of `images` ([(role, thread, elf)]); each counted once per build:
    a margin at or above FLOOR is kept in `cache` with its numbers."""
    out = []
    for role, thread, elf in images:
        marker = _marker(cache, elf, firmware_elf(elf))
        if marker.is_file():
            kept = json.loads(marker.read_text())
            need = Need(kept["chain"], kept["frames"])
            out.append(Margin(role, thread, kept["room"], kept["firmware"], need))
            continue
        found = margin(role, thread, elf, tree)
        out.append(found)
        if found.margin >= FLOOR:
            marker.parent.mkdir(parents=True, exist_ok=True)
            part = marker.with_suffix(f".{id(found):x}")
            part.write_text(
                json.dumps(
                    {
                        "room": found.room,
                        "firmware": found.firmware,
                        "chain": found.need.chain,
                        "frames": found.need.frames,
                    }
                )
            )
            part.replace(marker)
    return out
