"""The `kmr` command line: diagnostics and out-of-tree builds."""

import argparse
import platform
import sys
from pathlib import Path

from komorebi import __version__, _env, build, power


def doctor_report(environ=None):
    """What this host offers, as a dict; opens no device and runs no program."""
    tree, source = _env.find_tree(environ)
    return {
        "version": __version__,
        "python": f"{platform.python_version()} ({sys.executable})",
        "package_dir": str(_env.package_dir()),
        "include_dir": _env.include_dir(),
        "tt_metal": tree,
        "source": source,
        "runtime": _env.runtime_version(),
        "build_dir": tree and _env.build_dir(tree),
        "x280_runtime": bool(tree and _env.has_x280_runtime(tree)),
        "x280_compiler": tree and _env.x280_compiler(tree),
    }


def format_report(report):
    """The doctor report as aligned lines."""

    def found(value, missing):
        return str(value) if value else missing

    unset = f"not found ({_env.HOME_VARIABLE} is unset or not a directory)"
    tree = report["tt_metal"]
    if report["source"] == _env.RUNTIME_SOURCE:
        source = (
            f"{_env.RUNTIME_SOURCE} {report['runtime']} ({_env.HOME_VARIABLE} unset)"
        )
    elif report["source"]:
        source = _env.HOME_VARIABLE
    else:
        source = "-"
    # The runtime package carries a built runtime and no build directory.
    home = tree and report["source"] == _env.HOME_VARIABLE
    runtime = "yes" if report["x280_runtime"] else "no"
    rows = [
        ("komorebi", report["version"]),
        ("python", report["python"]),
        ("package", report["package_dir"]),
        ("include", found(report["include_dir"], "not found")),
        ("tt-metal", found(tree, unset)),
        ("from", source),
        ("build", found(report["build_dir"], "not found" if home else "-")),
        ("x280 runtime", runtime if tree else "-"),
        ("x280 compiler", found(report["x280_compiler"], "not found" if tree else "-")),
    ]
    return "\n".join(f"{name:<14}{value}" for name, value in rows)


def run_build(args):
    """Builds what the arguments name; returns the exit code."""
    tree = args.tree or _env.find_tree()[0]
    if tree is None:
        print(
            f"kmr build: pass --tree, set {_env.HOME_VARIABLE} or install "
            f"{_env.RUNTIME_SOURCE}",
            file=sys.stderr,
        )
        return 2
    try:
        if args.kind == "image":
            build.build_image(tree, args.out, args.sources)
        else:
            build.build_cxx(
                args.kind, tree, args.out, args.sources[0], args.includes, args.defines
            )
    except build.BuildError as err:
        print(f"kmr build: {err}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    """Parses `argv` and runs the chosen subcommand; returns the exit code."""
    parser = argparse.ArgumentParser(prog="kmr", description="Komorebi tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report the host without opening a device")

    builder = sub.add_parser(
        "build", help="build a host program, Python module or X280 image"
    )
    builder.add_argument("kind", choices=[*build.TEMPLATES, "image"])
    builder.add_argument("out", type=Path, help="output file")
    builder.add_argument("sources", nargs="+", help="source files (one for C++)")
    builder.add_argument("--tree", type=Path, help="tt-metal tree (default: env)")
    builder.add_argument("-I", dest="includes", action="append", default=[])
    builder.add_argument("-D", dest="defines", action="append", default=[])

    sampler = sub.add_parser(
        "power", help="print card, GPU or CPU power while you watch"
    )
    sampler.add_argument("--card", type=int, help="card index (default: every card)")
    sampler.add_argument("--gpu", type=int, help="GPU index (through nvidia-smi)")
    sampler.add_argument("--cpu", action="store_true", help="the CPU package")
    sampler.add_argument("--seconds", type=float, default=10.0)
    sampler.add_argument("--period", type=float, default=power.DEFAULT_PERIOD_S)

    args = parser.parse_args(argv)
    if args.command == "power":
        return power.run_cli(args, sys.stdout)
    if args.command == "doctor":
        print(format_report(doctor_report()))
        return 0
    return run_build(args)


if __name__ == "__main__":
    sys.exit(main())
