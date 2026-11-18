"""reset_core's order, a stand-in check (no device; the native code needs a card):
in km_host.cpp's reset_core, every call that can refuse its arguments comes before
the core is put into reset. Prints each call's line and the verdict."""

import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text().split("\n")
start = next(
    i
    for i, line in enumerate(source)
    if "reset_core(" in line and "std::array" in source[i - 1] + line
)
end = next(
    i
    for i in range(start, len(source))
    if "cluster.deassert_risc_reset_at_core" in source[i]
)
body = source[start:end]
found = {}
for name in (
    "connections_of(",
    "firmware_tables(",
    "drops.size() % 4",
    "kernels_on_core(",
    "assert_risc_reset_at_core(",
    "close_connections(",
):
    hits = [start + i + 1 for i, line in enumerate(body) if name in line]
    found[name] = hits[0] if hits else None
    print(f"{name:30} line {found[name]}")
reset = found["assert_risc_reset_at_core("]
checks = [
    n for n in found if n not in ("assert_risc_reset_at_core(", "close_connections(")
]
late = [n for n in checks if found[n] is None or found[n] > reset]
throws_after = [
    start + i + 1
    for i, line in enumerate(body)
    if "throw std::invalid_argument" in line and start + i + 1 > reset
]
print(
    "checks after the reset:",
    late or "none",
    "| invalid_argument after the reset:",
    throws_after or "none",
)
ok = not late and not throws_after
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
