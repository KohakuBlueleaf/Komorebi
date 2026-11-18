"""A reduce's tile read by a later stream loop of the same handler: `whole =
km.reduce(acc, ...)` lowered to a hidden fp32 field of one tile the sums are
written to, km.all_reduce of that field (every rank then holds the group's sum),
and the loop's `whole` read as that field's tile."""

import ast
import copy

from komorebi.dsl.front.names import spelling


def hidden(name):
    """The hidden field that holds the reduce `name = km.reduce(...)` gave."""
    return f"km_whole_{name}"


def reduce_call(node):
    """(name, the reduced tile's name, the call) of `name = km.reduce(tile, ...)`."""
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
    ):
        return None
    call = node.value
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "km"
        and func.attr == spelling("reduce")
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and not any(k.arg == spelling("combine") for k in call.keywords)
    ):
        return None
    return node.targets[0].id, call.args[0].id, call


def uses(node, name):
    """Whether `node` reads `name`."""
    return any(
        isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
        for n in ast.walk(node)
    )


def stream_operand(node, name):
    """Whether `node` is a stream loop that reads `name` only as a tile function's
    operand (`f(a, name)`)."""
    if not isinstance(node, ast.For) or "km.Stream(" not in ast.unparse(node.iter):
        return False
    calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and any(isinstance(a, ast.Name) and a.id == name for a in n.args)
    ]
    reads = sum(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))
    operands = sum(
        isinstance(a, ast.Name) and a.id == name for c in calls for a in c.args
    )
    return operands > 0 and operands == reads


def found(definition):
    """{name: index of its assign} of each reduce whose value only later stream
    loops of the handler's body read (at least one)."""
    wholes = {}
    body = definition.body
    for index, node in enumerate(body):
        reduced = reduce_call(node)
        if reduced is None:
            continue
        later = [n for n in body[index + 1 :] if uses(n, reduced[0])]
        if later and all(stream_operand(n, reduced[0]) for n in later):
            wholes[reduced[0]] = index
    return wholes


class Read(ast.NodeTransformer):
    """`name` read as `self.<hidden field>`."""

    def __init__(self, name):
        self.name = name

    def visit_Name(self, node):
        if node.id != self.name or not isinstance(node.ctx, ast.Load):
            return node
        field = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=hidden(self.name),
            ctx=ast.Load(),
        )
        return ast.copy_location(field, node)


def lower(definition):
    """The handler's def with each such reduce lowered (found); the names lowered."""
    wholes = found(definition)
    if not wholes:
        return []
    body = []
    for index, node in enumerate(definition.body):
        reduced = reduce_call(node)
        if reduced is None or reduced[0] not in wholes or wholes[reduced[0]] != index:
            body.append(node)
            continue
        name, tile, call = reduced
        field = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()), attr=hidden(name), ctx=ast.Load()
        )
        write = ast.Expr(
            ast.Call(
                func=ast.Attribute(value=field, attr="write", ctx=ast.Load()),
                args=[ast.Name(id=tile, ctx=ast.Load())],
                keywords=[],
            )
        )
        everywhere = ast.Expr(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="km", ctx=ast.Load()),
                    attr=spelling("all_reduce"),
                    ctx=ast.Load(),
                ),
                args=[copy.deepcopy(field)],
                keywords=copy.deepcopy(call.keywords),
            )
        )
        for made in (write, everywhere):
            ast.copy_location(made, node)
            ast.fix_missing_locations(made)
            for inner in ast.walk(made):
                ast.copy_location(inner, node)
        body += [write, everywhere]
    for name in wholes:
        body = [Read(name).visit(node) for node in body]
    definition.body = body
    return list(wholes)
