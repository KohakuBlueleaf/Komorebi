"""Both data movers of a core on one stream, the host side: the pattern that deals a
stream's blocks to mover A (RISCV_0, NoC 0) and mover B (RISCV_1, NoC 1), and the check
of a program's pairs. The device side is komorebi/device/km_movement.hpp.
"""

import re
from fractions import Fraction
from pathlib import Path

from komorebi.errors import ProgramError

PATTERN_LENGTH_MAX = 24
LETTERS = re.compile(rf"[AB]{{1,{PATTERN_LENGTH_MAX}}}")
MOVERS = ("A", "B")


class Pattern:
    """Which mover owns block k of a stream: `letters`, 1 to 24 of A and B, block k
    going to the mover of letter k mod the length ("AAAB": three of every four to A)."""

    def __init__(self, letters):
        letters = str(letters)
        if not LETTERS.fullmatch(letters):
            raise ValueError(
                f"pattern {letters!r}: 1 to {PATTERN_LENGTH_MAX} letters A and B"
            )
        self.letters = letters

    @property
    def word(self):
        """The word km::Pattern::of reads: the length, then a bit per B at bit 8 on."""
        owners = sum(1 << i for i, letter in enumerate(self.letters) if letter == "B")
        return len(self.letters) | owners << 8

    def define(self):
        """The word as the text of a define (hex), for a pattern fixed at build."""
        return f"0x{self.word:X}"

    def owner(self, block):
        """The letter of the mover that owns block `block`."""
        return self.letters[block % len(self.letters)]

    def share(self, mover):
        """The part of the blocks mover `mover` ("A" or "B") moves, as a Fraction."""
        if mover not in MOVERS:
            raise ValueError(f"mover {mover!r}: 'A' or 'B'")
        return Fraction(self.letters.count(mover), len(self.letters))

    def blocks(self, mover, count):
        """How many of blocks 0 to count - 1 mover `mover` moves."""
        if mover not in MOVERS:
            raise ValueError(f"mover {mover!r}: 'A' or 'B'")
        whole, rest = divmod(count, len(self.letters))
        return whole * self.letters.count(mover) + self.letters[:rest].count(mover)

    def __int__(self):
        return self.word

    def __str__(self):
        return self.letters

    def __repr__(self):
        return f"Pattern({self.letters!r})"

    def __eq__(self, other):
        return isinstance(other, Pattern) and other.letters == self.letters

    def __hash__(self):
        return hash(self.letters)


# A call that fills a circular buffer and one that drains it, as a kernel source names
# them; the first argument is the buffer.
FILLING = re.compile(r"\b(?:cb_reserve_back|cb_push_back|reserve_back|reserve)\s*\(")
DRAINING = re.compile(r"\b(?:cb_wait_front|cb_pop_front|wait_front)\s*\(")
ARGUMENT = re.compile(r"\s*([A-Za-z_]\w*|0x[0-9A-Fa-f]+|\d+)u?\s*[,)]")
CONSTANT = re.compile(r"constexpr\s+uint32_t\s+(\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)u?\s*;")
INCLUDE = re.compile(r'#include\s+"([^"]+)"')
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _texts(path, seen=None):
    """The text of a source and of the local headers it includes, comments removed."""
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen or not path.is_file():
        return []
    seen.add(path)
    text = COMMENT.sub("", path.read_text())
    found = [text]
    for name in INCLUDE.findall(text):
        found += _texts(path.parent / name, seen)
    return found


def buffer_roles(path):
    """{buffer index: {"fill", "drain"}} of the calls in a kernel source whose buffer
    argument is a number or a `constexpr uint32_t` constant of the source or its
    local headers; a buffer named through a variable is not seen."""
    texts = _texts(path)
    constants = {}
    for text in texts:
        for name, value in CONSTANT.findall(text):
            constants[name] = int(value, 0)
    source = texts[0] if texts else ""
    roles = {}
    for role, call in (("fill", FILLING), ("drain", DRAINING)):
        for matched in call.finditer(source):
            argument = ARGUMENT.match(source, matched.end())
            if not argument:
                continue
            name = argument[1]
            index = constants.get(name) if name[0].isalpha() else int(name, 0)
            if index is not None:
                roles.setdefault(index, set()).add(role)
    return roles


def check_pairs(name, pairs, sources):
    """Raises ProgramError when a source, as far as it can be seen, names a half of a
    pair both to fill and to drain, or two sources fill (or drain) one half."""
    halves = {}
    for pair in pairs:
        for half in (pair.mover_a, pair.mover_b):
            halves[half] = pair
    kernels = {}
    for kernel, path in sources.items():
        kernels.setdefault(Path(path).resolve(), []).append(kernel)
    seen = {"fill": {}, "drain": {}}
    for path, kernel_names in kernels.items():
        label = "/".join(kernel_names)
        for index, roles in buffer_roles(path).items():
            if index not in halves:
                continue
            if roles == {"fill", "drain"}:
                raise ProgramError(
                    f"{name}: {label} both fills and drains buffer {index}, a half of "
                    "a pair: a half has one filling and one draining CPU"
                )
            for role in roles:
                other = seen[role].setdefault(index, label)
                if other != label:
                    raise ProgramError(
                        f"{name}: {other} and {label} both {role} buffer {index}, a "
                        "half of a pair: a half has one filling and one draining CPU"
                    )
