"""Reads, stamps and bumps the version in pyproject.toml and komorebi.__version__.
Usage: version.py read | nightly | bump patch|minor|major | write <version>
(nightly prints X.Y.(Z+1).devYYYYmmddHHMMSS).
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = r'(?m)^({}\s*=\s*")([^"]+)(")'  # a line-anchored `name = "..."`
FILES = {
    ROOT / "pyproject.toml": re.compile(LINE.format("version")),
    ROOT / "src/komorebi/__init__.py": re.compile(LINE.format("__version__")),
}


def read():
    """The version in pyproject.toml; the package's copy must agree."""
    found = {path.name: p.search(path.read_text()) for path, p in FILES.items()}
    if not all(found.values()):
        raise SystemExit(f"no version in {[k for k, v in found.items() if not v]}")
    versions = {m.group(2) for m in found.values()}
    if len(versions) != 1:
        raise SystemExit(f"the versions disagree: {sorted(versions)}")
    return versions.pop()


def write(version):
    for path, pattern in FILES.items():
        path.write_text(pattern.sub(rf"\g<1>{version}\g<3>", path.read_text(), 1))


def parts(version):
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    return tuple(int(g) for g in core.groups())


def bump(version, part):
    major, minor, patch = parts(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version, now=None):
    """The PEP 440 dev release of the next patch, stamped with the UTC time."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
    return f"{bump(version, 'patch')}.dev{stamp}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("read", "nightly", "bump", "write"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    current = read()
    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        print(bump(current, args.value or "patch"))
    else:
        if not args.value:
            raise SystemExit("write needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
