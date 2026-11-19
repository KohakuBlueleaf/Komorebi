"""The unit program's math CPU to C++: a header with `<name>_init()` (once per stream)
and `<name>(dst)` (one tile in Dst tile `dst`), one coprocessor instruction a line, for
a compute kernel to include and call on its math thread. No kernel skeleton, no
movers; the waits around it are the caller's."""

import re

from komorebi.device import chips
from komorebi.dsl.tile.hazards import wormhole as wormhole_forms
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
    insns,
    wormhole_run,
)

# A float literal or type in the text: compute code carries no float (rule no-float).
FLOAT = re.compile(r"\bfloat\b|\bdouble\b|\d\.\d*f?\b|\d+e[-+]?\d+")
# Fields written by their tt-llk names, which each chip's header numbers (Blackhole's
# words unchanged): {op: {field: {Blackhole's number: name}}}
NAMED = {"STALLWAIT": chips.STALL_NAMES}


# Wormhole's ADDR_MOD_SET_Base around a function's vector words, cleared around its
# products: as tt-llk's _llk_math_eltwise_sfpu_start_ and _done_ set and clear it
BASE_ON = ["TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);"]
BASE_OFF = [
    "TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, ckernel::p_stall::WAIT_SFPU);",
    "TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);",
]


def _arg(op, name, value):
    """A field's C++ text: its tt-llk name (NAMED) or its number."""
    return NAMED.get(op, {}).get(name, {}).get(value, str(value))


def _based(item):
    """`item` a vector word with an address mode of 4 or more."""
    return item.op in chips.BASED and item.args["sfpu_addr_mode"] >= chips.ADDR_MODS


def _wormhole(pad, wormhole, blackhole=None):
    """Lines for Wormhole alone (`wormhole`), else `blackhole`'s, by ARCH_WORMHOLE."""
    out = ["#if defined(ARCH_WORMHOLE)", *[pad + line for line in wormhole]]
    if blackhole:
        out += ["#else", *[pad + line for line in blackhole]]
    return out + ["#endif"]


def _word(item):
    """The TTI_ line of `item` (its fields checked as encode does)."""
    encode(item.op, item.args)
    args = ", ".join(_arg(item.op, n, item.args[n]) for n, _ in OPS[item.op].args)
    # A macro without arguments (TTI_SFPNOP) takes no parentheses.
    call = f"({args})" if OPS[item.op].args else ""
    return f"TTI_{item.op}{call};"


def _body(items, depth):
    """A function's lines: with the base set around them when a word needs it, and
    Wormhole's SFPNOPs and replay counts (unit.wormhole_run)."""
    based = any(_based(x) for x in insns(items))
    lines = _lines(items, depth, based, wormhole_run(items))
    if not based:
        return lines
    pad = "    " * depth
    return [*_wormhole(pad, BASE_ON), *lines, *_wormhole(pad, BASE_OFF)]


def _recount(item, wormhole):
    """`item` a REPLAY whose count on Wormhole holds its recorded SFPNOPs, or that
    issues recorded words past the replay buffer after it."""
    if not wormhole:
        return False
    counts = wormhole.counts
    changed = id(item) in counts and counts[id(item)] != item.args["len"]
    return changed or id(item) in wormhole.tails


class _Without:
    """A WormholeRun without `item`'s SFPNOP (written already)."""

    def __init__(self, wormhole, item):
        self.nops = wormhole.nops - {id(item)}
        self.counts, self.tails = wormhole.counts, wormhole.tails


def _lines(items, depth, based=False, wormhole=None):
    pad, out = "    " * depth, []
    for item in items:
        if isinstance(item, Loop):
            var, count = item.name, item.count
            if isinstance(count, Runtime) and count.guard:
                out.append(f"{pad}if ({count.text}) {{")
                out += _lines(item.body, depth + 1, based, wormhole)
                out.append(f"{pad}}}")
                continue
            if isinstance(count, Runtime):
                count = count.text
            elif item.unroll:
                out.append(f"#pragma GCC unroll {item.count}")
            out.append(f"{pad}for (uint32_t {var} = 0; {var} < {count}; ++{var}) {{")
            out += _lines(item.body, depth + 1, based, wormhole)
            out.append(f"{pad}}}")
        elif isinstance(item, Insn) and item.wormhole:  # Wormhole's own words
            out.append("#if defined(ARCH_WORMHOLE)")
            out += _lines(item.wormhole, depth, based, wormhole)
            out.append("#else")
            plain = Insn(item.op, item.args, item.note, item.uses, item.runtime)
            out += _lines([plain], depth, based)
            out.append("#endif")
        elif isinstance(item, Insn) and wormhole and id(item) in wormhole.nops:
            out += _wormhole(pad, ["TTI_SFPNOP;"])  # a hold Wormhole does not make
            out += _lines([item], depth, based, _Without(wormhole, item))
        elif isinstance(item, Insn) and _recount(item, wormhole):
            count = wormhole.counts[id(item)]
            form = Insn(item.op, {**item.args, "len": count}, item.note)
            tail = _lines(wormhole.tails.get(id(item), []), depth, based, wormhole)
            out += ["#if defined(ARCH_WORMHOLE)", pad + _word(form), *tail]
            out += ["#else", pad + _word(item), "#endif"]
        elif isinstance(item, Insn) and item.runtime:
            out.append(f"{pad}{_runtime(item)};")
        elif isinstance(item, Insn) and _based(item):
            mode = item.args["sfpu_addr_mode"] - chips.ADDR_MODS
            form = Insn(item.op, {**item.args, "sfpu_addr_mode": mode}, item.note)
            out += _wormhole(pad, [_word(form)], [_word(item)])
        elif isinstance(item, Insn):
            out.append(pad + _word(item))
        elif isinstance(item, Call):
            out.append(f"{pad}{item.text};")
        elif isinstance(item, Product) and item.insns:
            out.append(f"{pad}// {item.text}: {len(item.insns)} words")
            # the matrix unit's address modes are 0 to 3: the base cleared around it
            out += _wormhole(pad, BASE_OFF) if based else []
            if item.wormhole:  # Wormhole's own image, in the modes its init sets
                out.append("#if defined(ARCH_WORMHOLE)")
                out += _lines(item.wormhole, depth, False, wormhole)
                out.append("#else")
                out += _lines(item.insns, depth)
                out.append("#endif")
            else:
                out += _lines(item.insns, depth, False, wormhole)
            out += _wormhole(pad, BASE_ON) if based else []
        elif not isinstance(item, Sync):
            raise UnitError(f"no C++ for {item!r}")
    return out


def _runtime(item):
    """TT_<op> of an instruction whose immediate the caller gives (`item.runtime`, its
    C++): the word built on the core; its other fields checked as TTI_'s are."""
    encode(item.op, item.args)
    args = [
        item.runtime if n == "imm16" else str(item.args[n])
        for n, _ in OPS[item.op].args
    ]
    return f"TT_{item.op}({', '.join(args)})"


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
    wormhole_forms.forms(unit)
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
    signature += "".join(f", const uint32_t {s}" for s in unit.scalars)
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
        *[f"// {s}: a run-time scalar, a float32's bits." for s in unit.scalars],
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
        *_body(unit.init["math"], 2),
        "    }",
        "",
        f"    inline void {name}({signature}) {{",
        *_body(_math_part(unit.tile["math"]), 2),
        "    }",
        "",
    ]
    for part, items in unit.parts.items():
        lines += [
            f"    // after the caller's {part} products: {name}_{part}(dst)",
            f"    inline void {name}_{part}(const uint32_t dst) {{",
            *_body(_math_part(items), 2),
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
