"""Every NoC atomic of the device sources that names a return address puts the command
buffer's return address back at once (km::atomic_return_reset, hazards.md)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "src" / "komorebi_native"
NAMED = re.compile(r"noc_fast_atomic_increment<[^<>;]*,\s*true>\s*\(")


def statement_end(text, start):
    """The index just past the call's closing parenthesis and its semicolon."""
    depth = 0
    at = text.index("(", start)
    while True:
        if text[at] == "(":
            depth += 1
        elif text[at] == ")":
            depth -= 1
            if depth == 0:
                break
        at += 1
    end = text.index(";", at)
    assert text[at + 1 : end].strip() == "", text[start : end + 1]
    return end + 1


def named_atomics():
    """(path, line, the source after the call) of each atomic naming its return."""
    found = []
    for path in sorted(SOURCES.rglob("*")):
        if path.suffix not in (".hpp", ".h", ".cpp"):
            continue
        text = path.read_text()
        for match in NAMED.finditer(text):
            end = statement_end(text, match.start())
            line = text.count("\n", 0, match.start()) + 1
            found.append((path.relative_to(ROOT), line, text[end:]))
    return found


def test_every_named_return_is_put_back():
    found = named_atomics()
    assert len(found) >= 10  # the runtime's reservations: none was missed by the scan
    missing = [
        f"{path}:{line}"
        for path, line, after in found
        if not after.lstrip().startswith("atomic_return_reset(")
    ]
    assert not missing, (
        "an atomic that names its return address, not followed at once by "
        f"km::atomic_return_reset: {missing}"
    )
