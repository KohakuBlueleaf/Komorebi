"""The stack a Tensix kernel's linked ELF needs on its RISC and the stack it has: the
firmware's frames up to its call of the kernel, the kernel's deepest call chain from
_start, an overlay part's at the kernel's call of it (docs/reference/host.md, "A
kernel's stack"); the tt-1xx firmware enables no interrupt."""

import bisect
import functools
import hashlib
import json
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from komorebi import _env
from komorebi.errors import ProgramError, StackError

# The least margin a kernel's image keeps once its need (the firmware's frames, then
# its deepest chain) is counted: one aligned frame for what the count misses.
FLOOR = 16
# A margin under this is named in a warning: the firmware's own least stack.
WARN = 256
# Functions that hand control to the kernel without an indirect call.
HANDOVERS = ("wh_iram_trampoline",)
# Library leaves whose indirect jump lands inside themselves (newlib's assembly
# memset jumps into its own table): no frame, no callee.
LEAVES = ("memset", "memcpy")
PART_ENTRY = "km_overlay_entry"  # an overlay part's first byte (program/overlay.py)
# Inlined functions whose indirect call is a wait's ready test (km_core.hpp): it reaches
# a function whose address is taken, never a part.
READY_TESTS = ("km::SharedReady::operator()",)
# A ready test's function (km::ready_called<Ready>): its address is taken only where a
# wait builds its SharedReady, which goes by value into a shared poll (mangled names).
READY_CALLED = "2km12ready_calledI"
SHARED_POLLS = (
    "2km16poll_stop_shared", "2km16poll_live_shared",
    "2km9poll_stopINS_11SharedReadyE", "2km9poll_liveINS_11SharedReadyE",
)  # fmt: skip
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
    """One function of an ELF: its frame, callees, what the count cannot follow;
    `sets_sp` the absolute sp it sets (a firmware's _start); `taken` when its address
    is taken (`stored`: and kept by a store outside a frame); `takes` the functions
    whose address it takes; `sites` its indirect calls, `ready` how many are tests."""

    name: str
    frame: int = 0
    calls: set = field(default_factory=set)
    tails: set = field(default_factory=set)
    indirect: int = 0
    unknown: list = field(default_factory=list)
    sets_sp: int | None = None
    taken: bool = False
    stored: bool = False
    takes: set = field(default_factory=set)
    sites: list = field(default_factory=list)
    ready: int = 0


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
    """One kernel image: its role and RISC, the stack it has (`room`), the firmware's
    frames before it, its own need; `at_least` when the need is a lower bound (a call
    that may enter a part, a cycle), `cycles` [(caller, callee)] counted
    once; `part` the overlay part on its deepest chain (its folder's kind), else ""."""

    role: str
    thread: str
    room: int
    firmware: int
    need: Need
    at_least: bool = False
    cycles: tuple = ()
    part: str = ""

    @property
    def needed(self):
        return self.firmware + self.need.bytes

    @property
    def margin(self):
        return self.room - self.needed

    def line(self):
        """Room / need / margin, the need "at least" when it is a lower bound."""
        least, most = ("at least ", "at most ") if self.at_least else ("", "")
        return (
            f"{self.role} ({self.thread}) {self.room:,} / {least}{self.needed:,} / "
            f"{most}{self.margin:,} B"
        )


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


def _take(out, function, at):
    """`function` takes the address of out[at]."""
    out[at].taken = True
    function.takes.add(at)


def _values(op, words):
    """The registers an instruction reads as values: a load's or store's base is an
    address it uses, not a value it passes on (a jump's register is read)."""
    if op[0] == "j":
        return [word.split("(")[-1].rstrip(")") for word in words]
    read = words if op in _STORES or op[0] == "b" else words[1:]
    return [word for word in read if "(" not in word]


def scan(listing, funcs):
    """{address: Function} of objdump -d's `listing`, over `funcs` (functions()); an
    address objdump notes (a function's) is taken when the register holding it is
    stored, read by anything but a load or store as its base, or still held where its
    block ends; a note on a load or store is the address it reads or writes."""
    starts = sorted(funcs)
    out = {at: Function(funcs[at][1]) for at in starts}
    ends = {at: at + funcs[at][0] for at in starts}
    current, run, constants, holds = None, 0, {}, {}
    for line in listing.splitlines():
        found = _INSN.match(line)
        if not found:
            continue
        pc, op = int(found.group(1), 16), found.group(2)
        if current is None or not current <= pc < ends[current]:
            for at in holds.values():  # held at its function's end: escapes
                _take(out, out[current], at)
            k = bisect.bisect_right(starts, pc) - 1
            current = starts[k] if k >= 0 and pc < ends[starts[k]] else None
            run, constants, holds = 0, {}, {}
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
            function.sites.append(pc)
        named = _NAMED.search(found.group(3))
        address = int(named.group(1), 16) if named else None
        memory = "(" in found.group(3).split("#")[0] and op[0] != "j"
        if address in out and op[0] not in "bj" and not memory:
            holds[words[0]] = address  # in a register: taken at a use but a base
        elif op in _STORES and words[0] in holds and len(words) == 2:
            _take(out, function, holds[words[0]])
            if words[1].split("(")[-1].rstrip(")") not in ("sp", "s0"):
                out[holds[words[0]]].stored = True  # kept where another may load it
        elif op == "mv" and len(words) == 2 and words[1] in holds:
            holds[words[0]] = holds[words[1]]
        else:
            for word in _values(op, words):
                if word in holds:
                    _take(out, function, holds[word])
            if words and op not in _STORES and op[0] not in "bj":
                holds.pop(words[0], None)
        if op[0] in "bj" or op == "ret":  # the block ends: a held address escapes
            for at in holds.values():
                _take(out, function, at)
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


def count(found, entry):
    """(the deepest chain from `entry`, the deepest to an indirect call that may enter
    an overlay part or None, the cycles [(caller, callee)]) over `found` (scan()): an
    indirect call reaches each function whose address is taken (a ready test's: those
    of tested()) and, but for a ready test, a part; a cycle is counted twice."""
    taken = {at for at, function in found.items() if function.taken}
    tests = {
        at for at in taken if READY_CALLED in found[at].name and not found[at].stored
    }
    cycles = []

    def tested(path):
        """What a ready test's call in a shared poll on `path` reaches: the ready tests
        whose address the polls or the frame that called them take, and every other
        function whose address is taken; anywhere else, all of them."""
        mine = set()
        for k, at in enumerate(reversed(path)):
            mine |= found[at].takes & tests
            if not any(poll in found[at].name for poll in SHARED_POLLS):
                return (taken - tests) | mine if k else taken
        return taken

    def walk(at, path):
        function = found[at]
        if function.unknown:
            raise ProgramError(f"{function.name}: {function.unknown[0]}")
        calls, opened = set(function.calls), None
        if function.indirect and function.name not in LEAVES:
            if function.indirect > function.ready:  # a part's frames: in its own ELF
                calls |= taken
                opened = Need([function.name], [function.frame])
            else:  # only ready tests
                calls |= tested(path)
        best = Need([function.name], [function.frame])
        for target, tail in [(t, False) for t in calls] + [
            (t, True) for t in function.tails
        ]:
            if target not in found:
                raise ProgramError(f"{function.name}: a call outside the ELF")
            if path.count(target) > 1:  # a cycle's third pass: not counted
                continue
            if target in path and len(set(path)) == len(path):  # the first time
                pair = (function.name, found[target].name)
                cycles.extend([pair] if pair not in cycles else [])
            below, below_open = walk(target, (*path, target))
            for chain, mine in ((below, "best"), (below_open, "opened")):
                if chain is None:
                    continue
                if not tail:
                    chain = Need(
                        [function.name, *chain.chain], [function.frame, *chain.frames]
                    )
                if mine == "best" and chain.bytes > best.bytes:
                    best = chain
                if mine == "opened" and (opened is None or chain.bytes > opened.bytes):
                    opened = chain
        return best, opened

    best, opened = walk(entry, (entry,))
    return best, opened, cycles


def _entry(found):
    for at, function in found.items():
        if function.name == "_start":
            return at
    raise ProgramError("no _start")


def _tool(tree, name):
    path = _env.sfpi_bin(tree) / f"riscv-tt-elf-{name}"
    if not path.is_file():
        raise ProgramError(
            f"{path} not found: the stack check needs the toolchain "
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
    """(scan(), symbols()) of `elf`, by the toolchain of `tree`; read once a process
    while the file is unchanged (a kernel is counted with each of its parts)."""
    try:
        status = Path(elf).stat()
    except OSError as error:
        raise ProgramError(f"{elf}: {error.strerror}") from error
    return _read(str(elf), status.st_size, status.st_mtime_ns, str(tree))


@functools.lru_cache(maxsize=64)
def _read(elf, size, mtime, tree):
    table = _run(_tool(tree, "readelf"), "-sW", elf)
    listing = _run(_tool(tree, "objdump"), "-d", "--no-show-raw-insn", elf)
    return ready_tests(scan(listing, functions(table)), elf, tree), symbols(table)


def ready_tests(found, elf, tree):
    """`found` (scan()) with each function's `ready`: its indirect calls whose innermost
    inlined function (addr2line -i of the toolchain of `tree`) is one of READY_TESTS.
    Any other call, or none when addr2line says nothing, may enter a part."""
    sites = [pc for function in found.values() for pc in function.sites]
    if not sites:
        return found
    try:
        lines = _run(
            _tool(tree, "addr2line"), "-a", "-f", "-C", "-i", "-e", str(elf),
            *(f"{pc:#x}" for pc in sites),
        ).splitlines()  # fmt: skip
    except (ProgramError, OSError):
        return found
    inner = {}  # address: its innermost function, the line after the address's own
    for k, line in enumerate(lines[:-1]):
        if line.startswith("0x"):
            inner[int(line, 16)] = _short(lines[k + 1])
    for function in found.values():
        function.ready = sum(inner.get(pc) in READY_TESTS for pc in function.sites)
    return found


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


def part_needs(parts, tree):
    """{part: (its deepest chain, whether a lower bound, its cycles)} of `parts` (one
    stack's overlay parts): a part's call that may enter a part (the runtime program
    part's of a handler part) goes into the deepest of `parts`; a part met again on its
    chain is a cycle (its build folders), counted once: every chain a lower bound."""
    counted, out, on, looped = {}, {}, [], []
    for part in parts:
        found, names = read(part, tree)
        entry = names.get(PART_ENTRY)
        if entry not in found:
            raise ProgramError(f"{part}: no {PART_ENTRY}")
        best, opened, cycles = count(found, entry)
        at = None if opened is None else named(found, opened, part, tree)
        counted[part] = named(found, best, part, tree), at, list(cycles)

    def deepest(part):
        if part in out:
            return out[part]
        need, at, cycles = counted[part]
        on.append(part)
        for other in parts if at is not None else ():
            if other in on:
                looped.append((Path(part).parent.name, Path(other).parent.name))
                continue
            own, _, more = deepest(other)
            cycles += [pair for pair in more if pair not in cycles]
            if at.bytes + own.bytes > need.bytes:
                need = Need([*at.chain, *own.chain], [*at.frames, *own.frames])
        on.pop()
        out[part] = need, bool(cycles), cycles
        return out[part]

    for part in parts:
        deepest(part)
    if looped:
        return {
            part: (need, True, [*cycles, *dict.fromkeys(looped)])
            for part, (need, _, cycles) in out.items()
        }
    return out


def margin(role, thread, elf, tree, firmware=None, parts=()):
    """The Margin of kernel `elf` (role, RISC name) against its firmware's ELF; with
    `parts` (overlay parts' ELFs), each one's chain (part_needs: a part calling a
    part too) at the kernel's call of a part."""
    firmware = Path(firmware or firmware_elf(elf))
    if not firmware.is_file():
        raise ProgramError(f"{role}: no firmware ELF at {firmware}")
    deepest = ""
    try:
        found, names = read(elf, tree)
        best, opened, cycles = count(found, _entry(found))
        need, least = named(found, best, elf, tree), opened is not None or bool(cycles)
        if opened is not None and parts:
            least = bool(cycles)
            at = named(found, opened, elf, tree)
            for part, (own, part_least, part_cycles) in part_needs(parts, tree).items():
                least = least or part_least
                cycles += [pair for pair in part_cycles if pair not in cycles]
                if at.bytes + own.bytes > need.bytes:
                    need = Need([*at.chain, *own.chain], [*at.frames, *own.frames])
                    deepest = part_kind(part)
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
    room = fw_names["__stack_top"] - base
    cycles = tuple((_demangled(a, tree), _demangled(b, tree)) for a, b in cycles)
    return Margin(role, thread, room, before, need, least, cycles, deepest)


def part_kind(elf):
    """An overlay part's name from its build folder ("service-reader-<key>": the
    service reader part), else the ELF's file name."""
    folder = Path(elf).parent.name
    kind = folder.rsplit("-", 1)[0] if "-" in folder else ""
    return f"the {kind.replace('-', ' ')} part" if kind else Path(elf).name


def _demangled(name, tree):
    """`name` demangled and shortened (_short), else as it is."""
    try:
        return _short(_run(_tool(tree, "c++filt"), name).strip()) or name
    except (ProgramError, OSError):
        return name


def _marker(cache, elf, firmware, parts=()):
    key = ""
    for path in (Path(elf), Path(firmware), *map(Path, parts)):
        status = path.stat()
        key += f"{path.resolve()}|{status.st_size}|{status.st_mtime_ns}|"
    key += _env.code_key("program/checks/stack.py")  # a changed check counts again
    return Path(cache) / hashlib.sha256(key.encode()).hexdigest()[:32]


def margins(images, tree, cache):
    """[Margin] of `images` ([(role, thread, elf)] or [(role, thread, elf, parts)]);
    each counted once per build: a margin at or above FLOOR is kept in `cache` with
    its numbers."""
    out = []
    for role, thread, elf, *more in images:
        parts = tuple(more[0]) if more else ()
        marker = _marker(cache, elf, firmware_elf(elf), parts)
        if marker.is_file():
            kept = json.loads(marker.read_text())
            need = Need(kept["chain"], kept["frames"])
            cycles = tuple(tuple(pair) for pair in kept.get("cycles", ()))
            out.append(
                Margin(
                    role, thread, kept["room"], kept["firmware"], need,
                    kept.get("at_least", False), cycles, kept.get("part", ""),
                )  # fmt: skip
            )
            continue
        found = margin(role, thread, elf, tree, parts=parts)
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
                        "at_least": found.at_least,
                        "cycles": [list(pair) for pair in found.cycles],
                        "part": found.part,
                    }
                )
            )
            part.replace(marker)
    return out


ARCH_NAMES = {"wormhole_b0": "Wormhole", "blackhole": "Blackhole"}


def kernel_images(kernels, skip=()):
    """[(role, RISC, ELF)] of every Tensix kernel ELF among a build's `kernels`
    ([(source, [ELF paths])]) but those in `skip`: the role its source's name."""
    out, skip = [], {str(Path(elf)) for elf in skip}
    for source, elves in kernels:
        for elf in elves:
            if elf and str(Path(elf)) not in skip:
                risc = Path(elf).stem
                if risc in THREADS:
                    out.append((Path(source).name, THREADS[risc], elf))
    return out


def part_images(parts):
    """[(role, RISC, kernel ELF, part ELFs)] of `parts` ({kernel ELF: [part ELFs]}):
    a kernel with the overlay parts that run on its stack."""
    out = []
    for elf, own in sorted(parts.items()):
        risc = Path(elf).stem
        if risc in THREADS and own:
            role = f"{Path(elf).parents[2].name} with its overlay parts"
            out.append((role, THREADS[risc], elf, tuple(sorted(set(own)))))
    return out


def smallest(found):
    """The load log's line: the smallest margin of each RISC (room / need / margin)."""
    least = {}
    for each in found:
        if each.thread not in least or each.margin < least[each.thread].margin:
            least[each.thread] = each
    return "; ".join(least[thread].line() for thread in sorted(least))


def judge(program, found, arch, log=lambda line: None, bound=False):
    """Refuses `program` (StackError) when one of `found` (Margins) keeps less than
    FLOOR: exact, a lower bound or, with `bound` (a kernel with its parts), an upper
    bound; warns by name under WARN and for each cycle."""
    over = min(found, key=lambda each: each.margin, default=None)
    if over is not None and over.margin < FLOOR:
        raise StackError(program, over, arch, found, FLOOR, bound=bound)
    what = (
        "; the need an upper bound: the kernel's deepest chain to any call into a "
        "part, then the deepest part, so a run may never meet it"
        if bound
        else ""
    )
    for each in found:
        if each.margin < WARN:
            warnings.warn(
                f"{program}: the stack of {each.line()} (room / need / margin{what}), "
                f"under {WARN} B",
                stacklevel=3,
            )
        for caller, callee in each.cycles:
            warnings.warn(
                f"{program}: {each.role} ({each.thread}): {caller} calls {callee} "
                "again on its own path; the cycle is counted twice",
                stacklevel=3,
            )
    if found:
        log(f"stack (room / need / margin, the smallest a RISC): {smallest(found)}")
    return found
