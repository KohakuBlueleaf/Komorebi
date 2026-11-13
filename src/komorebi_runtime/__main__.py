"""Commands of the runtime package: `python -m komorebi_runtime doctor` (files present,
libraries load, compiler found), `sfpi` (fetch the compiler the tree names), `prime`
(prime a card's X280 memory), `root` (print the runtime root: TT_METAL_RUNTIME_ROOT).
"""

import argparse
import ctypes
import sys

import komorebi_runtime as kr
from komorebi_runtime import prepare


def doctor():
    """Checks the install; returns (lines, number of problems)."""
    info = kr.manifest()
    lines = [
        f"komorebi-runtime {info['version']} (tt-metal {info['pin']}, "
        f"build {info['build']}, patches {', '.join(info['patches'])})",
        f"package   {kr.package_directory()}",
    ]
    problems = 0
    missing = [
        name for name in info["files"] if not (kr.package_directory() / name).is_file()
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
    lines.append(
        "prime     once per card: kmr prime (python -m komorebi_runtime prime)"
    )
    return lines, problems


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m komorebi_runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check files, library loading and the compiler")
    sub.add_parser("sfpi", help="fetch the RISC-V compiler into the cache")
    primer = sub.add_parser("prime", help="prime a card's X280 memory, once per card")
    primer.add_argument("--card", type=int, help="the N of /dev/tenstorrent/N")
    sub.add_parser("root", help="print the runtime root (TT_METAL_RUNTIME_ROOT)")
    args = parser.parse_args(argv)
    try:
        if args.command == "sfpi":
            print(prepare.fetch_sfpi())
        elif args.command == "prime":
            card = prepare.default_card() if args.card is None else args.card
            lines, code = prepare.prime(card, "python -m komorebi_runtime sfpi")
            print("\n".join(lines), file=sys.stderr if code else sys.stdout)
            return code
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
