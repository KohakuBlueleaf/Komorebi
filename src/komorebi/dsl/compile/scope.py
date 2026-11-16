"""A handler's names: those assigned more than once, a field named, a name read after
the if or loop that first assigned it, a parameter's name taken, the names an
expression reads."""

import ast

from komorebi.dsl.compile.source import refuse
from komorebi.dsl.names import kind_of, spelling


def assigned_names(frame):
    """Names assigned more than once in the handler (their C++ is not const)."""
    counts = {}
    for node in ast.walk(frame.definition):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target, node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                counts[target.id] = counts.get(target.id, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def field_name(frame, node):
    """The field `node` is (`self.name`), or None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in frame.compiled.fields
    ):
        return node.attr
    return None


def not_ended(frame, node, name):
    if name in frame.ended and name not in frame.scope:
        kind, line = frame.ended[name]
        refuse(
            frame.where,
            node,
            f"{name} is first assigned inside the {kind} at line {line}, and "
            f"ends with it",
            f"assign it before the {kind}",
        )


def not_parameter(frame, node, name):
    if any(param.name == name for param in frame.params):
        refuse(
            frame.where,
            node,
            f"{name} is a parameter, a command word: it is not assigned again",
            f"give the new value a name of its own: n = {name} + 1",
        )


def scoped(frame, kind, node, before):
    """The names first assigned inside an if or a loop end with it, as C++'s
    do: a line after it that reads one is refused (not_ended). Ranges are the
    build's alone and stay."""
    for name in set(frame.scope) - before:
        if getattr(frame.scope[name], "kind", None) in ("int", "word", "partition"):
            frame.scope.pop(name)
            frame.ended[name] = (kind, node.lineno)
        elif getattr(frame.scope[name], "kind", None) == "moved":
            frame.scope[name].ended = frame.scope[name].ended or (kind, node.lineno)


def uses(frame, node):
    """The names an expression reads (scope names and generated ones)."""
    found = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id in frame.scope:
            value = frame.scope[item.id]
            found.add(value.cxx or item.id)
        if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
            if item.value.id == "self":
                kind = kind_of(item.attr, ("rank", "size", "x", "y", "core"))
                if kind in ("x", "y", "core"):
                    found.add(f"km_self_{kind}")
                elif kind:
                    found.add(f"km_{kind}")
            if item.value.id in frame.counts and item.attr == spelling("pages"):
                found.add(frame.counts[item.value.id])
            entry = frame.scope.get(item.value.id)
            if entry is not None and entry.kind in ("entry", "entries"):
                found.add(frame.entries_word(item, entry))
    return found
