"""Fails when a binary in a wheel names a path of the machine that built it.
Usage: check_paths.py WHEEL... [--forbid PATH]...
Forbidden: each --forbid, $HOME, $GITHUB_WORKSPACE, $RUNNER_WORKSPACE, $RUNNER_TEMP
and $KMR_RUNTIME_TREE when set.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

ENVIRONMENT = (
    "GITHUB_WORKSPACE",
    "RUNNER_WORKSPACE",
    "RUNNER_TEMP",
    "KMR_RUNTIME_TREE",
)
# Members that are read as binaries: ELF files, objects, archives, images.
SUFFIXES = (".o", ".a", ".bin")
# A path continues through these; a match must end at any other byte.
CONTINUES = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


def forbidden_paths(extra, environ=None):
    """The paths no binary may name: `extra`, the home directory and the runner's."""
    environ = os.environ if environ is None else environ
    values = [*extra, str(Path.home()), *(environ.get(k, "") for k in ENVIRONMENT)]
    # "/" alone would match every path; a path is named without its trailing slash.
    return sorted({v.rstrip("/") for v in values if v.rstrip("/")})


def is_binary(name, head):
    return head.startswith(b"\x7fELF") or ".so" in name or name.endswith(SUFFIXES)


def find_paths(data, paths):
    """(path, context) for each place `data` names one of `paths`."""
    found = []
    for path in paths:
        needle = path.encode()
        at = data.find(needle)
        while at >= 0:
            end = at + len(needle)
            if end >= len(data) or data[end] not in CONTINUES:
                context = data[max(0, at - 20) : end + 40]
                found.append((path, context.decode("ascii", "replace")))
            at = data.find(needle, end)
    return found


def scan(wheel, paths):
    """(member, path, context) over the binaries of `wheel`, and how many were read."""
    hits, count = [], 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            with archive.open(member) as handle:
                data = handle.read()
            if is_binary(member.filename, data[:4]):
                count += 1
                hits += [(member.filename, p, c) for p, c in find_paths(data, paths)]
    return hits, count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--forbid", action="append", default=[], metavar="PATH")
    args = parser.parse_args(argv)
    paths = forbidden_paths(args.forbid)
    failed = False
    for wheel in args.wheels:
        hits, count = scan(wheel, paths)
        for member, path, context in hits:
            print(f"{wheel.name}: {member} names {path}: ...{context!r}...")
        failed |= bool(hits)
        print(f"{wheel.name}: {count} binaries, {len(hits)} naming {len(paths)} paths")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
