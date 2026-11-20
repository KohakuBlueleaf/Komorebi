"""The device code counts bits without libgcc on Wormhole (no card): __builtin_popcount
is called only by km::popcount (core/km_bits.hpp), in the #else of ARCH_WORMHOLE; on
Wormhole it is a call to __popcountsi2, which an overlay's part cannot reach."""

import re
from pathlib import Path

HEADERS = Path(__file__).resolve().parents[3] / "src/komorebi_native/include/komorebi"
CONDITION = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def calls(path):
    """(line, in the #else of `#if defined(ARCH_WORMHOLE)`) of each
    __builtin_popcount in `path`."""
    stack, found = [], []
    for number, line in enumerate(path.read_text().split("\n"), 1):
        matched = CONDITION.match(line)
        if matched:
            kind, rest = matched[1], matched[2].strip()
            if kind in ("if", "ifdef", "ifndef"):
                stack.append([rest == "defined(ARCH_WORMHOLE)", False])
            elif kind in ("elif", "else") and stack:
                stack[-1][1] = True
            elif kind == "endif" and stack:
                stack.pop()
            continue
        if "__builtin_popcount" in line.split("//")[0]:
            found.append((number, any(ours and other for ours, other in stack)))
    return found


def test_the_popcount_builtin_is_called_by_km_popcount_alone_and_not_on_wormhole():
    paths = sorted(HEADERS.rglob("*.hpp")) + sorted(HEADERS.rglob("*.h"))
    assert paths
    found = {
        str(path.relative_to(HEADERS)): calls(path) for path in paths if calls(path)
    }
    assert list(found) == ["device/core/km_bits.hpp"]
    assert [guarded for _, guarded in found["device/core/km_bits.hpp"]] == [True]
