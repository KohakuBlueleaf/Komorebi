"""The unit program's math CPU to C++: a header with `<name>_init()` (once per stream)
and `<name>(dst)` (one tile in Dst tile `dst`), one coprocessor instruction a line, for
a compute kernel to include and call on its math thread. No kernel skeleton, no
movers; the waits around it are the caller's."""

import re

from komorebi.dsl.tile.macro import check_back_door, check_templates
from komorebi.dsl.tile.unit import (
    OPS,
    Call,
    Check,
    Insn,
    Loop,
    Product,
    Runtime,
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
            var, count = item.name, item.count
            if isinstance(count, Runtime) and count.guard:
                out.append(f"{pad}if ({count.text}) {{")
                out += _lines(item.body, depth + 1)
                out.append(f"{pad}}}")
                continue
            if isinstance(count, Runtime):
                count = count.text
            elif item.unroll:
                out.append(f"#pragma GCC unroll {item.count}")
            out.append(f"{pad}for (uint32_t {var} = 0; {var} < {count}; ++{var}) {{")
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
        elif isinstance(item, Product) and item.insns:
            out.append(f"{pad}// {item.text}: {len(item.insns)} words")
            out += _lines(item.insns, depth)
        elif not isinstance(item, Sync):
            raise UnitError(f"no C++ for {item!r}")
    return out


def _math_part(items):
    """The math CPU's own lines: the syncs and the library units around the vector
    code (copy_tile, the products, tile_regs_*) are the caller's."""
    return [
        x
        for x in items
        if not isinstance(x, Sync)
        and not (isinstance(x, Product) and not x.insns)
        and not (isinstance(x, Call) and x.text.startswith("copy_tile"))
    ]


WAIT = re.compile(r"cb_wait_front\((\w+)")
PUSHED = re.compile(r"cb_push_back\((\w+)|pack_tile\([^,]+,\s*(\w+)\)")


def _items(items):
    for x in items:
        if isinstance(x, Loop):
            yield from _items(x.body)
        else:
            yield x


def check_buffers(unit):
    """Rule "one producer": each buffer of the unit's table (the unpack CPU's waits name
    who pushes; the pack CPU's packs and pushes are its own) has one producer: a second
    one's pages never reach a wait (each push writes its own count into the buffer's
    received counter). Returns the check's text; raises UnitError."""
    producers = {}
    for cpu in ("unpack", "pack"):
        for x in _items(unit.init[cpu] + unit.tile[cpu]):
            text = getattr(x, "text", "")
            if cpu == "unpack" and isinstance(x, Sync):
                for name in WAIT.findall(text):
                    producers.setdefault(name, set()).add(x.other)
            elif cpu == "pack":
                for found in PUSHED.findall(text):
                    name = next(n for n in found if n)
                    producers.setdefault(name, set()).add("pack")
    two = {k: v for k, v in producers.items() if len(v) > 1}
    if two:
        name, who = next(iter(two.items()))
        raise UnitError(
            f"{unit.name}: buffer {name} is pushed by {' and '.join(sorted(who))}: one "
            "producer a buffer (a state kept in place is neither pushed nor popped)"
        )
    table = ", ".join(f"{k} by {''.join(v)}" for k, v in sorted(producers.items()))
    return f"ok: {table or 'no buffer'}"


def emit(unit):
    """The C++ header of `unit`'s math CPU; refuses text that carries a float, a
    macro whose template the init does not store or the tile's code overwrites, a
    word with a VD of 12 or more that is no template store, and a buffer with two
    producers."""
    name = unit.name
    unit.checks.append(
        Check(
            "macro-templates",
            "each macro's template stored, kept",
            check_templates(unit),
        )
    )
    unit.checks.append(
        Check(
            "back-door",
            "a VD of 12 or more only in a template store",
            check_back_door(unit),
        )
    )
    unit.checks.append(
        Check("one-producer", "one pushing CPU or kernel a buffer", check_buffers(unit))
    )
    signature = "const uint32_t dst"
    if unit.band:
        signature += ", const uint32_t rows"
    lines = [
        f"// {name}: emitted by komorebi.dsl.tile from its unit program ({unit.dest} "
        "Dst); math CPU only.",
        f"// Writes {', '.join(unit.clobbers) or 'no constant register'}; the caller "
        "acquires Dst and waits.",
        *(
            ["// rows: the live rows (1 .. 32), run in bands of 4."]
            if unit.band
            else []
        ),
        *[
            f"// Product into Dst tile dst + {p.dst}: {p.a} @ {p.w} at "
            f"{unit.fidelity}; the caller unpacks llk_unpack_AB_matmul({p.a}, "
            f"{p.w}, ...)."
            for p in unit.products
        ],
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
        f"    inline void {name}({signature}) {{",
        *_lines(_math_part(unit.tile["math"]), 2),
        "    }",
        "",
    ]
    for part, items in unit.parts.items():
        lines += [
            f"    // after the caller's {part} products: {name}_{part}(dst)",
            f"    inline void {name}_{part}(const uint32_t dst) {{",
            *_lines(_math_part(items), 2),
            "    }",
            "",
        ]
    lines += [
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
