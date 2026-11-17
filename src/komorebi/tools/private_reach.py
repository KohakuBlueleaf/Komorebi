"""Private reaches of the scripts: a file of projects/, examples/, experiments/ or
tutorial/ that names `x._name` (not on self or cls), imports `komorebi._x` or a `_x`
name from komorebi, passes `"_name"` to getattr/hasattr/setattr, or reaches an
internal attribute (private_reach.toml). Usage: python -m komorebi.tools.private_reach
"""

import argparse
import ast
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from komorebi.tools.comment_lint import under
from komorebi.tools.cxx_format import repo_root

FOLDERS = ("projects", "examples", "experiments", "tutorial")
CONFIG = Path(__file__).resolve().with_name("private_reach.toml")
OWN = {"self", "cls"}  # a script's class reaching its own names
ATTRIBUTE_CALLS = {"getattr", "hasattr", "setattr", "delattr"}


class Reach(NamedTuple):
    """One private reach: its file, line and what it names."""

    path: str
    line: int
    text: str


def private(name):
    """A private name: one leading underscore, not a dunder."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def foreign_names(tree):
    """The names a module binds by importing a package other than komorebi."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "komorebi":
                    out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and not ours(node):
            out.update(alias.asname or alias.name for alias in node.names)
    return out


def ours(node):
    """Whether an `from X import ...` names a komorebi module."""
    return (node.module or "").split(".")[0] == "komorebi"


def private_string(node, allowed):
    """The private name a getattr / hasattr / setattr / delattr call passes, or None."""
    if not isinstance(node.func, ast.Name) or node.func.id not in ATTRIBUTE_CALLS:
        return None
    name = node.args[1] if len(node.args) > 1 else None
    text = name.value if isinstance(name, ast.Constant) else None
    ok = isinstance(text, str) and private(text) and text not in allowed
    return text if ok else None


def root_name(node):
    """The name an attribute chain starts from (`a` of `a.b.c`), else None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def reaches(path, source, config):
    """The private reaches of one file's source."""
    tree = ast.parse(source, filename=path)
    foreign = foreign_names(tree)
    allowed = set(config.get("foreign", {}))
    internal = set(config.get("internal", {}))
    out = []

    def add(node, text):
        out.append(Reach(path, node.lineno, text))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = root_name(node.value)
            if root in OWN and isinstance(node.value, ast.Name) or root in foreign:
                continue
            if private(node.attr) and node.attr not in allowed:
                add(node, f".{node.attr}")
            elif node.attr in internal:
                add(node, f".{node.attr} ({config['internal'][node.attr]})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "komorebi" and any(map(private, parts)):
                    add(node, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and ours(node):
            parts = node.module.split(".")
            names = [alias.name for alias in node.names]
            if any(map(private, parts)) or any(map(private, names)):
                add(node, f"from {node.module} import {', '.join(names)}")
        elif isinstance(node, ast.Call) and private_string(node, allowed):
            add(node, f'{node.func.id}(..., "{private_string(node, allowed)}")')
    return sorted(out)


def check(sources, config):
    """(reaches, exempt files found) of {path: source}; an exempt file is skipped."""
    exempt = config.get("exempt", {})
    found = []
    for path, source in sorted(sources.items()):
        if path not in exempt:
            found += reaches(path, source, config)
    return found, sorted(set(exempt) & set(sources))


def main():
    """Prints every reach, the exempt files with their reasons; exit 1 on a reach."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on a reach")
    args = parser.parse_args()
    root = repo_root().resolve()
    try:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        sources = {
            path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for folder in FOLDERS
            if (root / folder).is_dir()
            for path in under(root / folder, root)
            if path.suffix == ".py"
        }
        found, exempt = check(sources, config)
    except (OSError, SyntaxError, tomllib.TOMLDecodeError) as error:
        print(f"private_reach: {error}", file=sys.stderr)
        return 2
    for path in exempt:
        print(f"exempt: {path}: {config['exempt'][path]}")
    for missing in sorted(set(config.get("exempt", {})) - set(sources)):
        print(f"error: {missing}: exempt, but no such script (drop its line)")
    for reach in found:
        print(f"{reach.path}:{reach.line}: {reach.text}")
    print(f"{len(found)} private reaches in {len(sources)} scripts")
    stale = set(config.get("exempt", {})) - set(sources)
    return 1 if args.check and (found or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
