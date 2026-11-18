"""A row loop with two steps in flight: `for r in range(n)` whose body writes a step of
state r into a field (`self.p.write(f(s, self.s[r]))`) that later lines take by a
product. Row r + 1's lines up to the write run before row r's products (skewed)."""

import ast

from komorebi.dsl.compile import scope, tile_lookup
from komorebi.dsl.compile.source import refuse
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Glue, has_line

WORD_CALLS = ("min", "max")


def word_line(statement):
    """Whether `statement` names a word of arithmetic (min and max the only calls):
    both parts of a skewed loop compute it."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return False
    return all(
        not isinstance(n, ast.Call)
        or (isinstance(n.func, ast.Name) and n.func.id in WORD_CALLS)
        for n in ast.walk(statement.value)
    )


def step_write(frame, statement, name):
    """The field of `self.f.write(g(s, self.s[name]))`, g a tile function, else None."""
    call = statement.value if isinstance(statement, ast.Expr) else None
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == spelling("write")
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Call)
    ):
        return None
    field_ = scope.field_name(frame, call.func.value)
    step = call.args[0]
    if (
        field_ is None
        or tile_lookup.tile_function_of(frame, step) is None
        or len(step.args) != 2
        or not isinstance(step.args[1], ast.Subscript)
        or scope.field_name(frame, step.args[1].value) is None
        or not isinstance(step.args[1].slice, ast.Name)
        or step.args[1].slice.id != name
    ):
        return None
    return field_


def reads_field(frame, statement, field_):
    """Whether `statement` is `acc += g(self.f, ...)`: a product of the field."""
    value = statement.value if isinstance(statement, ast.AugAssign) else None
    return (
        isinstance(value, ast.Call)
        and bool(value.args)
        and scope.field_name(frame, value.args[0]) == field_
    )


def split(frame, node, name):
    """(word lines before, the if around the step or None, part A, part B, field) of
    a loop that runs skewed, else None. Part A ends with the step's write; part B
    takes the field by a product; the field is written and read nowhere else in the
    handler, and B reads no value of A but its words."""
    body = list(node.body)
    prefix = []
    while body and word_line(body[0]):
        prefix.append(body.pop(0))
    test = None
    if len(body) == 1 and isinstance(body[0], ast.If) and not body[0].orelse:
        test, body = body[0], list(body[0].body)
    if not all(isinstance(s, ast.Assign | ast.AugAssign | ast.Expr) for s in body):
        return None
    at = next(
        (i for i, s in enumerate(body) if step_write(frame, s, name) is not None), None
    )
    if at is None:
        return None
    field_ = step_write(frame, body[at], name)
    first, second = body[: at + 1], body[at + 1 :]
    if field_ in frame.packed or not any(reads_field(frame, s, field_) for s in second):
        return None
    own = {id(n) for s in second for n in ast.walk(s)} | {id(body[at].value.func.value)}
    for n in ast.walk(frame.definition):
        if (
            isinstance(n, ast.Attribute)
            and scope.field_name(frame, n) == field_
            and id(n) not in own
        ):
            return None  # the field is written or read outside the loop's step
    made = {
        t.id
        for s in first
        if not word_line(s)
        for t in ast.walk(s)
        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
    }
    read = {
        n.id
        for s in second
        for n in ast.walk(s)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    if made & read:
        return None
    words = [s for s in first if word_line(s)]
    return prefix, test, first, [*words, *second], field_


def halves(frame, node, name, first, end, parts):
    """Compute's lines of the skewed loop's turn k: part A of row k (k < end), then
    part B of row k - 1 (k > first). The field holds two writes (frame.skewed)."""
    prefix, test, part_a, part_b, field_ = parts
    frame.skewed.add(field_)
    frame.compiled.skewed_lines |= {
        (frame.method.name, s.lineno) for s in [*part_a, *part_b]
    }
    counter = skew_counter(name)
    out = []
    for part, condition, row, what in (
        (part_a, f"{counter} < ({end})", counter, "up to its write"),
        (part_b, f"{counter} > ({first})", f"{counter} - 1", "its products"),
    ):
        lines = list(prefix)
        if test is not None:
            lines.append(ast.copy_location(ast.If(test.test, part, []), test))
        else:
            lines += part
        before = set(frame.scope)
        moved, computed = frame.statements(lines)
        scope.scoped(frame, "loop", node, before)
        if any(has_line(m) for m in moved):
            refuse(
                frame.where,
                node,
                "a loop whose step is written to a field and taken by a product runs "
                "two rows in flight, and this one moves pages too",
                "move the pages in a loop of their own",
            )
        out.append(
            Block(
                f"if ({condition}) {{  // row {name}: {what}",
                "}",
                [Glue(f"const uint32_t {name} = {row};", {name}), *computed],
            )
        )
    return out


def skew_counter(name):
    """The C++ counter of a skewed loop over `name`."""
    return f"km_skew_{name}"
