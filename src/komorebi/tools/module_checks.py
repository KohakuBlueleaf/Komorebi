"""Module checks of src/komorebi: statements a file (ast.stmt nodes at any depth, less
imports and string-constant expressions such as docstrings), over 1000 a warning, over
2000 an error; module-level import cycles; imports inside functions. The allowlist,
module_checks.toml beside this file, only shrinks. Usage: [--check]."""

import argparse
import ast
import sys
import tomllib
from pathlib import Path

from komorebi.tools.comment_lint import under
from komorebi.tools.cxx_format import repo_root

SOFT_CAP = 1000
HARD_CAP = 2000
PACKAGE = Path("src") / "komorebi"
ALLOWLIST = Path(__file__).resolve().with_name("module_checks.toml")


def statements(tree):
    """The statement nodes of a module's tree, imports and docstrings (any expression
    statement of a string constant) left out: comments and layout cost nothing."""
    total = 0
    for node in ast.walk(tree):
        skipped = isinstance(node, ast.Import | ast.ImportFrom)
        if skipped or not isinstance(node, ast.stmt):
            continue
        value = node.value if isinstance(node, ast.Expr) else None
        total += not (isinstance(value, ast.Constant) and isinstance(value.value, str))
    return total


def module_name(path):
    """`komorebi.dsl.compiler` of `src/komorebi/dsl/compiler.py` (a package: its
    `__init__.py`'s folder)."""
    parts = list(Path(path).with_suffix("").parts[1:])
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(tree):
    """(node, function) for each import of a tree: function is the dotted name of the
    innermost enclosing function, or None for a module-level import."""
    found = []

    def visit(node, names, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import | ast.ImportFrom):
                found.append((child, ".".join(names) if inside else None))
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                own = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                visit(child, names + [child.name], inside or own)
            elif isinstance(child, ast.Lambda):
                visit(child, names + ["<lambda>"], True)
            else:
                visit(child, names, inside)

    visit(tree, [], False)
    return found


def targets(node, module, package, modules):
    """The modules of the package that an import statement names: `from a import b`
    names a.b when that is a module, else a; parent packages are not counted."""
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names if alias.name in modules}
    base = node.module or ""
    if node.level:
        parts = module.split(".")[: len(module.split(".")) - node.level + package]
        base = ".".join(parts + ([base] if base else []))
    out = set()
    for alias in node.names:
        name = f"{base}.{alias.name}"
        if name in modules:
            out.add(name)
        elif base in modules:
            out.add(base)
    return out


def cycles(edges):
    """The import cycles: each strongly connected group of two or more modules (or a
    module importing itself), its names sorted; Tarjan's algorithm."""
    index, low, stack, on_stack, found = {}, {}, [], set(), []

    def connect(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for other in sorted(edges.get(node, ())):
            if other not in index:
                connect(other)
                low[node] = min(low[node], low[other])
            elif other in on_stack:
                low[node] = min(low[node], index[other])
        if low[node] == index[node]:
            group = []
            while not group or group[-1] != node:
                group.append(stack.pop())
                on_stack.discard(group[-1])
            if len(group) > 1 or node in edges.get(node, ()):
                found.append(sorted(group))

    for node in sorted(edges):
        if node not in index:
            connect(node)
    return sorted(found)


def measure(sources):
    """(statements by path, cycles, in-function imports) of {path: source text};
    an in-function import is (path, function, statement text)."""
    trees = {path: ast.parse(text) for path, text in sources.items()}
    modules = {module_name(path): path for path in trees}
    counts, edges, inner = {}, {}, []
    for path, tree in trees.items():
        counts[path] = statements(tree)
        module = module_name(path)
        package = Path(path).name == "__init__.py"
        for node, function in imports(tree):
            if function is None:
                edges.setdefault(module, set()).update(
                    targets(node, module, package, modules)
                )
            else:
                inner.append((path, function, ast.unparse(node)))
    return counts, cycles(edges), sorted(set(inner))


def size_warnings(counts):
    """A warning for each file over the soft cap."""
    return [
        f"{path}: {count} statements (soft cap 1000): keep it only if it is one large"
        " coherent unit"
        for path, count in sorted(counts.items())
        if HARD_CAP >= count > SOFT_CAP
    ]


def size_errors(counts, allow):
    """The errors of the hard cap against the allowlist's files over it today."""
    hard = allow.get("over_hard_cap", {})
    errors = []
    for path, count in sorted(counts.items()):
        recorded = hard[path]["statements"] if path in hard else None
        if count > HARD_CAP and recorded is None:
            errors.append(f"{path}: {count} statements, over the hard cap of 2000")
        elif recorded is not None and count <= HARD_CAP:
            errors.append(f"{path}: {count} statements, at 2000 or under: take it off")
        elif recorded is not None and count > recorded:
            errors.append(f"{path}: {count} statements, grew past its {recorded}")
    missing = sorted(set(hard) - set(counts))
    return errors + [f"{path}: listed but not found: take it off" for path in missing]


def cycle_errors(found, allow):
    """A cycle must lie inside a listed one; a listed one must be a cycle of today."""
    listed = [sorted(entry["modules"]) for entry in allow.get("cycle", [])]
    errors = [
        f"import cycle among {', '.join(group)}"
        for group in found
        if not any(set(group) <= set(entry) for entry in listed)
    ]
    return errors + [
        f"listed cycle {', '.join(entry)} is not a cycle of today: write the cycle it"
        " became, or take it off the list"
        for entry in listed
        if entry not in found
    ]


def import_errors(inner, allow):
    """An import inside a function must be listed; a listed one must exist."""
    listed = {
        (entry["path"], entry["function"], entry["import"])
        for entry in allow.get("function_import", [])
    }
    errors = [
        f"{path}: `{text}` inside {function}: not allowlisted"
        for path, function, text in inner
        if (path, function, text) not in listed
    ]
    return errors + [
        f"{path}: `{text}` inside {function}: gone, take it off the list"
        for path, function, text in sorted(listed - set(inner))
    ]


def check(sources, allow):
    """(errors, measured) of {path: source text} against the allowlist dict."""
    counts, found, inner = measure(sources)
    errors = size_errors(counts, allow) + cycle_errors(found, allow)
    return errors + import_errors(inner, allow), (counts, found, inner)


def table(measured):
    """The report: files over 1000 statements, the cycles, the in-function imports."""
    counts, found, inner = measured
    lines = ["statements | file"]
    over = sorted((count, path) for path, count in counts.items() if count > SOFT_CAP)
    lines += [f"{count:10d} | {path}" for count, path in reversed(over)]
    lines += [f"cycle among {', '.join(group)}" for group in found]
    lines += [f"in {function}: {path}: {text}" for path, function, text in inner]
    return "\n".join(lines)


def package_sources(root):
    """{path relative to the root: text} of the package's Python files."""
    folder = root / PACKAGE
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in under(folder, root)
        if path.suffix == ".py" and path.is_file()
    }


def main():
    """Prints the table, or with --check the errors; exit status 1 on an error."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on an error")
    args = parser.parse_args()
    root = repo_root().resolve()
    try:
        allow = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))
        errors, measured = check(package_sources(root), allow)
    except (OSError, SyntaxError, tomllib.TOMLDecodeError, KeyError) as error:
        print(f"module_checks: {error}", file=sys.stderr)
        return 2
    if not args.check:
        print(table(measured))
    for warning in size_warnings(measured[0]):
        print(f"warning: {warning}")
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
