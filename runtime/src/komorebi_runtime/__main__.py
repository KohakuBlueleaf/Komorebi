"""Commands of the runtime package: `python -m komorebi_runtime doctor` (files present,
libraries load, compiler found), `sfpi` (fetch the compiler the tree names), `root`
(print the runtime root, for TT_METAL_RUNTIME_ROOT).
"""

import argparse
import ctypes
import hashlib
import platform
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import komorebi_runtime as kr


def sfpi_url(release, machine, distro):
    """(url, sha256) of the sfpi archive for `machine` and `distro`."""
    name = f"sfpi_{release['version']}_{machine}_{distro}.txz"
    url = f"{release['repo']}/releases/download/{release['version']}/{name}"
    return url, release[f"{machine}_{distro}_txz_hash"]


def host_distro():
    """The sfpi build for this host: "fedora" on RHEL-likes, else "debian"."""
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return "debian"
    return "fedora" if re.search(r"(?m)^ID(_LIKE)?=.*(fedora|rhel)", text) else "debian"


def fetch_sfpi():
    """Downloads and unpacks the sfpi release the tree names into the cache, unless it
    is there; returns its directory."""
    target = kr.sfpi_dir()
    if kr.sfpi_bin() is not None:
        return target
    url, sha256 = sfpi_url(kr.sfpi_release(), platform.machine(), host_distro())
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "sfpi.txz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response, archive.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise SystemExit(f"{url} does not match its recorded sha256")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        (Path(tmp) / "sfpi").replace(target)
    return target


def doctor():
    """Checks the install; returns (lines, number of problems)."""
    info = kr.manifest()
    lines = [
        f"komorebi-runtime {info['version']} (tt-metal {info['pin']}, "
        f"build {info['build']}, patches {', '.join(info['patches'])})",
        f"package   {kr.package_dir()}",
    ]
    problems = 0
    missing = [
        name for name in info["files"] if not (kr.package_dir() / name).is_file()
    ]
    if missing:
        problems += 1
        lines.append(
            f"files     {len(missing)} of {len(info['files'])} missing: "
            f"{missing[:3]}"
        )
    else:
        lines.append(f"files     {len(info['files'])} present")
    for name in info["libraries"]:
        try:
            ctypes.CDLL(str(kr.lib_dir() / name), mode=ctypes.RTLD_GLOBAL)
            lines.append(f"loads     {name}")
        except OSError as err:
            problems += 1
            lines.append(f"FAILS     {name}: {err}")
    pieces = kr.x280()
    files = [*pieces.objects, pieces.linker_script, pieces.idle_image]
    absent = [p for p in files if not p.is_file()]
    problems += bool(absent)
    lines.append(f"x280      {'missing ' + str(absent) if absent else pieces.include}")
    compiler = kr.sfpi_bin()
    if compiler is None:
        problems += 1
        lines.append(
            f"sfpi      not found in {kr.sfpi_dir()} "
            "(python -m komorebi_runtime sfpi)"
        )
    else:
        lines.append(f"sfpi      {compiler}")
    return lines, problems


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m komorebi_runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check files, library loading and the compiler")
    sub.add_parser("sfpi", help="fetch the RISC-V compiler into the cache")
    sub.add_parser("root", help="print the runtime root (TT_METAL_RUNTIME_ROOT)")
    args = parser.parse_args(argv)
    try:
        if args.command == "sfpi":
            print(fetch_sfpi())
        elif args.command == "root":
            print(kr.root())
        else:
            lines, problems = doctor()
            print("\n".join(lines))
            return 1 if problems else 0
    except kr.RuntimeMissing as err:
        print(f"komorebi_runtime: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
