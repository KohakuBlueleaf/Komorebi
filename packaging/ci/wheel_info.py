"""Reads the two release wheels in a directory: names, versions, requirements.
Usage: wheel_info.py identify DIR | versions DIR [--tag TAG] | requires WHEEL [EXTRA]...
identify prints KEY=value lines; requires prints the requirements of the wheel and of
the extras named, one per line.
"""

import argparse
import email.parser
import itertools
import re
import sys
import zipfile
from pathlib import Path

try:
    from packaging.requirements import Requirement

    from packaging.version import Version
except ImportError:  # the build machine's pip carries a copy
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def metadata(wheel):
    """The wheel's METADATA as a message."""
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        return email.parser.Parser().parsestr(archive.read(name).decode())


def requirements(wheel, extras=()):
    """The requirements of `wheel` that apply with `extras`, without markers."""
    found = []
    for req in all_requirements(wheel):
        if req.marker is None or any(
            req.marker.evaluate({"extra": e}) for e in ("", *extras)
        ):
            req.marker = None
            found.append(req)
    return found


def all_requirements(wheel):
    return [
        Requirement(line) for line in metadata(wheel).get_all("Requires-Dist") or []
    ]


def identify(directory):
    """The two wheels of `directory` as (wheel, runtime wheel, the requirement of the
    first that names the second): the runtime wheel is the one the other requires."""
    wheels = sorted(Path(directory).glob("*.whl"))
    if len(wheels) != 2:
        raise SystemExit(f"{directory} holds {len(wheels)} wheels, expected 2")
    for wheel, other in itertools.permutations(wheels):
        target = normalized(metadata(other)["Name"])
        names = [r for r in all_requirements(wheel) if normalized(r.name) == target]
        if names:
            return wheel, other, names[0]
    raise SystemExit(f"neither wheel in {directory} requires the other")


def describe(directory):
    """The names, versions and file names of the two wheels, as KEY: value."""
    wheel, runtime, requirement = identify(directory)
    head, base = metadata(wheel), metadata(runtime)
    return {
        "name": head["Name"],
        "version": head["Version"],
        "wheel": wheel.name,
        "runtime_name": base["Name"],
        "runtime_version": base["Version"],
        "runtime_wheel": runtime.name,
        "runtime_requirement": str(requirement.specifier),
        "prerelease": str(Version(head["Version"]).is_prerelease).lower(),
    }


def check_versions(directory, tag=None):
    """The problems: the tag disagrees with the version, or the runtime requirement
    of the main wheel does not admit the runtime wheel."""
    info = describe(directory)
    problems = []
    if tag is not None and tag.removeprefix("v") != info["version"]:
        problems.append(f"tag {tag} does not name version {info['version']}")
    specifier = identify(directory)[2].specifier
    if not specifier.contains(Version(info["runtime_version"]), prereleases=True):
        problems.append(
            f"{info['name']} requires {info['runtime_name']}{specifier}, "
            f"the wheel is {info['runtime_version']}"
        )
    return info, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("identify").add_argument("directory")
    versions = sub.add_parser("versions")
    versions.add_argument("directory")
    versions.add_argument("--tag")
    needs = sub.add_parser("requires")
    needs.add_argument("wheel")
    needs.add_argument("extras", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "identify":
        print("\n".join(f"{k}={v}" for k, v in describe(args.directory).items()))
    elif args.command == "requires":
        for req in requirements(args.wheel, args.extras):
            print(req)
    else:
        info, problems = check_versions(args.directory, args.tag)
        print(f"{info['name']} {info['version']}", end=", ")
        print(f"{info['runtime_name']} {info['runtime_version']}")
        print("\n".join(problems))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
