"""Module checks of src/komorebi: statements a file (ast.stmt nodes, less imports and
docstrings), over 1000 a warning, over 2000 an error; no import cycle over all imports,
importlib calls included; an import inside a function only of an optional package; over
12 Python files directly in a folder of src/ or tests/ a warning. Config: the .toml."""

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from komorebi.tools.comment_lint import under
from komorebi.tools.cxx_format import repo_root

SOFT_CAP = 1000
HARD_CAP = 2000
FLAT_CAP = 12  # Python files directly in one folder before a warning
FOLDERS = ("src", "tests")
PACKAGE = Path("src") / "komorebi"
CONFIG = Path(__file__).resolve().with_name("module_checks.toml")
PYPROJECT = "pyproject.toml"
# Exempt: the package's lazy names (a module __getattr__: it imports without tt-metal).
EXEMPT = "src/komorebi/__init__.py"
CALLS = {"import_module", "__import__"}


class Import(NamedTuple):
    """One import: its file, line, enclosing function (None at module level), the
    modules it names (dotted), and its text."""

    path: str
    line: int
    function: str | None
    names: tuple
    text: str


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


def statement_names(node, module, package):
    """The dotted names an import statement imports: `from a import b` gives a.b
    (resolved against the modules later) and a; relative imports resolved."""
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    base = node.module or ""
    if node.level:
        parts = module.split(".")[: len(module.split(".")) - node.level + package]
        base = ".".join(parts + ([base] if base else []))
    return tuple(f"{base}.{alias.name}|{base}" for alias in node.names)


def call_name(node):
    """The literal an importlib.import_module or __import__ call names; "" when the
    argument is not a string literal; None when the node is no such call."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name not in CALLS or not node.args:
        return None
    first = node.args[0]
    literal = isinstance(first, ast.Constant) and isinstance(first.value, str)
    return first.value if literal and not first.value.startswith(".") else ""


def imports(path, tree):
    """Every import of a file, statements and importlib calls alike."""
    module, package = module_name(path), Path(path).name == "__init__.py"
    found = []

    def visit(node, names, inside):
        for child in ast.iter_child_nodes(node):
            function = ".".join(names) if inside else None
            if isinstance(child, ast.Import | ast.ImportFrom):
                text = ast.unparse(child)
                given = statement_names(child, module, package)
                found.append(Import(path, child.lineno, function, given, text))
            called = call_name(child)
            if called is not None:
                given = (called,) if called else ()
                text = ast.unparse(child)
                found.append(Import(path, child.lineno, function, given, text))
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                own = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                visit(child, names + [child.name], inside or own)
            elif isinstance(child, ast.Lambda):
                visit(child, names + ["<lambda>"], True)
            else:
                visit(child, names, inside)

    visit(tree, [], False)
    return found


def resolved(name, modules):
    """The package module a dotted name reaches (`a.b|a`: a.b when a module, else a),
    or None for a name outside the package."""
    for one in name.split("|"):
        if one in modules:
            return one
    return None


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


def is_pending(item, target, pending):
    """True when an import is one of the named edges left to their owners."""
    return any(
        entry["path"] == item.path and entry["imports"] == target for entry in pending
    )


def check(sources, config, optional=frozenset()):
    """(errors, (statements by path, the imports, the pending edges met)) of {path:
    source text}: `config` the TOML's dict, `optional` the import names of optional
    third-party packages."""
    trees = {path: ast.parse(text) for path, text in sources.items()}
    modules = {module_name(path) for path in trees}
    pending = config.get("pending", [])
    optional = set(optional) | set(config.get("optional", {}))
    counts = {path: statements(tree) for path, tree in trees.items()}
    errors, edges, found, used = size_errors(counts, config), {}, [], {}
    for path, tree in sorted(trees.items()):
        exempt = path == EXEMPT
        for item in imports(path, tree):
            found.append(item)
            if not item.names:
                errors.append(f"{path}:{item.line}: `{item.text}`: not a literal")
                continue
            for name in item.names:
                target = resolved(name, modules)
                if exempt and item.function is not None:
                    continue
                if target is not None and is_pending(item, target, pending):
                    used.setdefault((path, target), item.line)
                    continue
                if target is not None:
                    edges.setdefault(module_name(path), set()).add(target)
                outside = name.split("|")[-1].split(".")[0]
                if item.function is not None and (target or outside not in optional):
                    errors.append(
                        f"{path}:{item.line}: `{item.text}` inside {item.function}:"
                        " import it at the top (only an optional third-party package"
                        " is imported in a function)"
                    )
    errors += [f"import cycle among {', '.join(group)}" for group in cycles(edges)]
    errors += [
        f"{entry['path']}: the edge to {entry['imports']} ({entry['owner']}) is gone:"
        " take it off the pending list"
        for entry in pending
        if (entry["path"], entry["imports"]) not in used
    ]
    left = [
        f"{entry['path']}:{used[key]}: imports {entry['imports']}, left to"
        f" {entry['owner']}"
        for entry in pending
        if (key := (entry["path"], entry["imports"])) in used
    ]
    return list(dict.fromkeys(errors)), (counts, found, left)


def size_warnings(counts):
    """A warning for each file over the soft cap."""
    return [
        f"{path}: {count} statements (soft cap 1000): keep it only if it is one large"
        " coherent unit"
        for path, count in sorted(counts.items())
        if HARD_CAP >= count > SOFT_CAP
    ]


def flat_warnings(paths, flat_ok=()):
    """A warning for each folder holding more than FLAT_CAP Python files directly,
    unless `flat_ok` names it; `paths` are relative to the repository, with slashes."""
    counts = {}
    for path in paths:
        if path.endswith(".py"):
            folder = path.rpartition("/")[0]
            counts[folder] = counts.get(folder, 0) + 1
    return [
        f"{folder}/: {count} files side by side: group them by concern"
        for folder, count in sorted(counts.items())
        if count > FLAT_CAP and folder not in flat_ok
    ]


def size_errors(counts, config):
    """The errors of the hard cap against the files listed over it today."""
    hard = config.get("over_hard_cap", {})
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


def optional_packages(root):
    """The distribution names of pyproject.toml's optional dependencies, as import
    names (lower case, dashes as underscores)."""
    project = tomllib.loads((root / PYPROJECT).read_text(encoding="utf-8"))
    groups = project.get("project", {}).get("optional-dependencies", {})
    names = (
        re.match(r"[A-Za-z0-9_.-]+", one).group() for g in groups.values() for one in g
    )
    return {name.lower().replace("-", "_") for name in names}


def table(measured):
    """The report: files over 1000 statements, then the imports inside functions."""
    counts, found = measured
    lines = ["statements | file"]
    over = sorted((count, path) for path, count in counts.items() if count > SOFT_CAP)
    lines += [f"{count:10d} | {path}" for count, path in reversed(over)]
    lines += [
        f"in {item.function}: {item.path}:{item.line}: {item.text}"
        for item in found
        if item.function is not None
    ]
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
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        optional = optional_packages(root)
        errors, measured = check(package_sources(root), config, optional)
    except (OSError, SyntaxError, tomllib.TOMLDecodeError, KeyError) as error:
        print(f"module_checks: {error}", file=sys.stderr)
        return 2
    if not args.check:
        print(table(measured))
    tracked = [
        path.relative_to(root).as_posix()
        for folder in FOLDERS
        for path in under(root / folder, root)
    ]
    flat = flat_warnings(tracked, config.get("flat_ok", {}))
    for warning in size_warnings(measured[0]) + flat:
        print(f"warning: {warning}")
    print(f"exempt: {EXEMPT}: the package's lazy names (its module __getattr__)")
    for edge in measured[2]:
        print(f"pending: {edge}")
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
