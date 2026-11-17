"""Include cycles of the device and host headers: the #include lines of every header
under src/komorebi_native/include, by file name; a group of headers that include one
another (a strongly connected group) is a cycle. Usage: python -m
komorebi.tools.header_cycles [--check]"""

import argparse
import re
import sys
from pathlib import Path

from komorebi.tools.cxx_format import repo_root

INCLUDE = Path("src/komorebi_native/include")
SUFFIXES = (".h", ".hpp")
LINE = re.compile(r'\s*#\s*include\s*[<"]([^>"]+)[>"]')


def includes(sources):
    """{name: [(included name, line)]} of {name: source}; only names in `sources`."""
    graph = {}
    for name, source in sources.items():
        edges = []
        for number, text in enumerate(source.splitlines(), 1):
            found = LINE.match(text)
            if found and Path(found.group(1)).name in sources:
                edges.append((Path(found.group(1)).name, number))
        graph[name] = edges
    return graph


def cycles(graph):
    """The groups of two or more names that reach one another, each sorted."""
    index, low, stack, on, out = {}, {}, [], set(), []

    def visit(name):  # Tarjan's strongly connected components
        index[name] = low[name] = len(index)
        stack.append(name)
        on.add(name)
        for other, _ in graph[name]:
            if other not in index:
                visit(other)
                low[name] = min(low[name], low[other])
            elif other in on:
                low[name] = min(low[name], index[other])
        if low[name] == index[name]:
            group = []
            while True:
                other = stack.pop()
                on.discard(other)
                group.append(other)
                if other == name:
                    break
            if len(group) > 1:
                out.append(sorted(group))

    for name in sorted(graph):
        if name not in index:
            visit(name)
    return sorted(out)


def main():
    """Prints each cycle with its include lines; exit 1 on a cycle with --check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on a cycle")
    args = parser.parse_args()
    root = repo_root().resolve() / INCLUDE
    sources = {}
    for path in sorted(root.rglob("*")):
        if path.suffix in SUFFIXES:
            if path.name in sources:
                print(f"header_cycles: two headers named {path.name}", file=sys.stderr)
                return 2
            sources[path.name] = path.read_text(encoding="utf-8")
    graph = includes(sources)
    found = cycles(graph)
    for group in found:
        print("cycle: " + ", ".join(group))
        for name in group:
            lines = [f"{other}:{n}" for other, n in graph[name] if other in group]
            print(f"  {name} includes {', '.join(lines)}")
    print(f"{len(found)} include cycles in {len(sources)} headers")
    return 1 if args.check and found else 0


if __name__ == "__main__":
    sys.exit(main())
