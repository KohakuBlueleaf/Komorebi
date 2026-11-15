"""The unit program's math CPU to C++: a header with `<name>_init()` (once per stream)
and `<name>(dst)` (one tile in Dst tile `dst`), one coprocessor instruction a line, for
a compute kernel to include and call on its math thread. No kernel skeleton, no
movers; the waits around it are the caller's."""

import re

from komorebi.dsl.tile.unit import (
    OPS,
    Call,
    Check,
    Insn,
    Loop,
    Sync,
    UnitError,
    encode,
)

# A float literal or type in the text: compute code carries no float (rule no-float).
FLOAT = re.compile(r"\bfloat\b|\bdouble\b|\d\.\d*f?\b|\d+e[-+]?\d+")


def _lines(items, depth):
    pad, out = "    " * depth, []
    for item in items:
        if isinstance(item, Loop):
            var = item.name
            if item.unroll:
                out.append(f"#pragma GCC unroll {item.count}")
            out.append(
                f"{pad}for (uint32_t {var} = 0; {var} < {item.count}; ++{var}) {{"
            )
            out += _lines(item.body, depth + 1)
            out.append(f"{pad}}}")
        elif isinstance(item, Insn):
            encode(item.op, item.args)
            args = ", ".join(str(item.args[n]) for n, _ in OPS[item.op].args)
            # A macro without arguments (TTI_SFPNOP) takes no parentheses.
            call = f"({args})" if OPS[item.op].args else ""
            out.append(f"{pad}TTI_{item.op}{call};")
        elif isinstance(item, Call):
            out.append(f"{pad}{item.text};")
        elif not isinstance(item, Sync):
            raise UnitError(f"no C++ for {item!r}")
    return out


def _math_part(items):
    """The math CPU's own lines: the syncs and the library units around the vector
    code (copy_tile, tile_regs_*) are the caller's."""
    return [
        x
        for x in items
        if not isinstance(x, Sync)
        and not (isinstance(x, Call) and x.text.startswith("copy_tile"))
    ]


def emit(unit):
    """The C++ header of `unit`'s math CPU; refuses text that carries a float."""
    name = unit.name
    lines = [
        f"// {name}: emitted by komorebi.dsl.tile from its unit program ({unit.dest} "
        "Dst); math CPU only.",
        f"// Writes {', '.join(unit.clobbers) or 'no constant register'}; the caller "
        "acquires Dst and waits.",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef TRISC_MATH",
        "namespace km_tile {",
        "",
        f"    inline void {name}_init() {{",
        *_lines(unit.init["math"], 2),
        "    }",
        "",
        f"    inline void {name}(const uint32_t dst) {{",
        *_lines(_math_part(unit.tile["math"]), 2),
        "    }",
        "",
        "}  // namespace km_tile",
        "#endif",
        "",
    ]
    text = "\n".join(lines)
    code = "\n".join(line for line in lines if not line.lstrip().startswith("//"))
    found = FLOAT.search(code)
    if found:
        raise UnitError(f"the emitted body carries a float ({found.group()!r})")
    unit.checks.append(
        Check("no-float", "no float in compute code", "ok: none in the text")
    )
    return text
