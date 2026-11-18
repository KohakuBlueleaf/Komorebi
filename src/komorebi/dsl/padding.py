"""A reduction or product along a tensor's padded axis, refused by name where its
padding could enter the result: at the build each handler's own rules, at each submit
the tensors' marks too (`padding`, set after each submit; docs/reference/tile.md,
"Padding"). A tensor in whole tiles skips all of it."""

import ast
import dataclasses
import inspect
import textwrap
from dataclasses import dataclass

from komorebi.dsl.tile import padding as tile_padding
from komorebi.errors import ProgramError

WRITES = ("write stream", "page write")  # the moves that write; the others read
WAY = (
    "make {name} whole tiles (a multiple of 32 along that axis), or multiply by a "
    "mask in the tile function (x * mask.row(), a field's row of 1 in the logical "
    'columns and 0 past them; tile.md, "Padding")'
)
EXTREME_WAY = (  # a max or a min: a mask's 0 is a value too
    "make {name} whole tiles on the host, its extra columns a value no row reaches "
    '(-inf for a max, inf for a min; tile.md, "Padding")'
)


@dataclass
class Facts:
    """A handler's tile calls (line, text, trace, {parameter: tensor} or None: any
    tensor it reads), products' lines, tensors read and written (its first write's
    line), and whether an inline body may write any tensor."""

    calls: list
    products: list
    reads: set
    writes: dict
    hidden: bool


def facts_of(compiled, methods):
    """{handler: Facts} of `compiled`; `methods` {handler: the method objects}."""
    facts = {}
    for name, objects in methods.items():
        mapped = {
            (h, line, cxx): params
            for h, line, _, cxx, params in compiled.padding_calls
            if h == name
        }
        calls = [
            (line, text, compiled.units[cxx][1].ir, mapped.get((h, line, cxx)))
            for h, line, text, _, cxx in compiled.tile_calls
            if h == name and getattr(compiled.units[cxx][1], "ir", None) is not None
        ]
        products = [(p[1], p[2]) for p in compiled.products if p[0] == name]
        moves = [m for m in compiled.moves if m.handler == name]
        writes = {}
        for move in moves:
            if move.how in WRITES:
                writes.setdefault(move.tensor, move.line)
        reads = {m.tensor for m in moves if m.how not in WRITES}
        facts[name] = Facts(calls, products, reads, writes, inline(objects))
    return facts


def inline(objects):
    """Whether a handler's source has an inline body (`km.cxx(...)`)."""
    for method in objects:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(method.function)))
        except (OSError, TypeError, SyntaxError):
            return True  # not read: a line not followed
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if getattr(f, "attr", None) == "cxx" or getattr(f, "id", None) == "cxx":
                    return True
    return False


def sides_of(tensors, built):
    """{parameter: Side} of the tensors not in whole tiles; `built`: at the build,
    every padding taken as zeros."""
    sides = {}
    for name, tensor in tensors.items():
        found = tile_padding.side(name, tensor)
        if found is not None:
            sides[name] = dataclasses.replace(found, clean=True) if built else found
    return sides


def joined(sides, names):
    """One Side for a parameter that may read any tensor of `names`."""
    found = [sides[n] for n in sorted(names) if n in sides]
    if not found:
        return None
    return tile_padding.Side(
        " or ".join(s.name for s in found),
        found[0].logical,
        found[0].padded,
        any(s.cols for s in found),
        any(s.rows for s in found),
        all(s.clean for s in found),
    )


def call_sides(ir, params, sides, reads):
    """{the traced function's parameter: Side} of a call."""
    names = [t.name for t in ir.tensors if not t.output]
    if params is None:
        every = joined(sides, reads)
        return {n: every for n in names if every is not None}
    return {p: sides[t] for p, t in params.items() if t in sides}


def refusal(handler, line, what, padded, tensors, built):
    """The ProgramError of `what` reading padding along its axis."""
    first = padded[0]
    axis = "columns" if first.cols else "rows"
    tensor = tensors.get(first.name.split(" or ")[0])
    way = WAY
    if what.startswith(("reduce_max", "reduce_min")):
        held = "padding (zeros, or what a handler wrote) is in every row"
        way = EXTREME_WAY
    elif built or first.clean:
        held = (
            "its padding holds zeros, and an operation before it does not keep "
            "zero (f(0) != 0)"
        )
    else:
        held = f"its padding was written by {getattr(tensor, 'padding_by', None)}"
    return ProgramError(
        f"{handler}, line {line}: {what} reads along the padded {axis} of "
        f"{first.name}, which is {first.logical} held as {first.padded} (padded to "
        f"whole tiles): {held}, so the padding would enter the result; "
        + way.format(name=first.name)
    )


def checked(handler, facts, tensors, built=False):
    """Refuses (ProgramError) a reduction or product of `handler` along a padded
    axis whose padding may enter its result; `tensors` {parameter: tensor}."""
    if facts is None:
        return
    sides = sides_of(tensors, built)
    if not sides:
        return  # every tensor in whole tiles
    for line, _, ir, params in facts.calls:
        found = tile_padding.problems(ir, call_sides(ir, params, sides, facts.reads))
        if found:
            at, what, padded = found[0]
            what += f" (at {at})" if at else ""
            raise refusal(handler, line, what, padded, tensors, built)
    dirty = [sides[n] for n in sorted(facts.reads) if n in sides and not sides[n].clean]
    for line, _ in facts.products:
        if dirty:
            raise refusal(handler, line, "a product", dirty, tensors, built)


def marked(handler, facts, tensors):
    """After a submit: each tensor not in whole tiles that `handler` writes is clean
    when its padding keeps zeros (every input's clean, every tile call keeps zero, no
    inline body), else dirty, by the handler's line."""
    if facts is None:
        return
    sides = sides_of(tensors, False)
    written = dict(facts.writes)
    if facts.hidden:  # an inline body may write any tensor it is given
        written = {n: facts.writes.get(n, "an inline body") for n in tensors}
    written = {n: line for n, line in written.items() if n in sides}
    if not written:
        return
    clean = not facts.hidden and all(
        sides[n].clean for n in facts.reads if n in sides and n not in written
    )
    clean = clean and all(
        tile_padding.keeps_zero(ir, call_sides(ir, params, sides, facts.reads))
        for _, _, ir, params in facts.calls
    )
    for name, line in written.items():
        tensor = tensors[name]
        if clean:
            tensor.padding, tensor.padding_by = "clean", None
        else:
            tensor.padding, tensor.padding_by = "dirty", f"{handler}, line {line}"
