"""The `kmr` command line: diagnostics and out-of-tree builds."""

import argparse
import platform
import sys
from pathlib import Path

from komorebi import __version__, _env
from komorebi.bench import power
from komorebi.builder import flags as build
from komorebi.builder import prebuild
from komorebi.device import mock
from komorebi.session import process
from komorebi.tools import dram_check

try:
    from komorebi_runtime import prepare
except ImportError:  # no runtime package, or one older than its setup steps
    prepare = None

# What `kmr sfpi`, `kmr prime` and `kmr setup` need, and a tree's own way to prime.
NEEDS_RUNTIME = (
    "kmr {}: needs the komorebi-runtime package, 0.81.1.3 or later "
    '(pip install "komorebi-tt[runtime]")'
)
TREE_PRIME = (
    "with a tt-metal tree instead: X280_ALLOW_LIM_ECC_PRIME=1 TT_VISIBLE_DEVICES=N "
    "build/test/tt_metal/x280/x280_prime_test, then tt-smi -r N"
)


def doctor_report(environ=None, root=None, dev=None):
    """What this host offers, as a dict; opens no device and runs no program. `cards`
    is each card's state from the driver's files (sysfs below `root`, nodes in `dev`).
    """
    tree, source = _env.find_tree(environ)
    # tt-metal's cards 0 .. n - 1 (each read at its driver number: card_states)
    cards = list(range(len(process.card_entries(root, dev))))
    return {
        "version": __version__,
        "python": f"{platform.python_version()} ({sys.executable})",
        "package_directory": str(_env.package_directory()),
        "include_directory": _env.include_directory(),
        "tt_metal": tree,
        "source": source,
        "runtime": _env.runtime_version(),
        "build_directory": tree and _env.build_directory(tree),
        "x280_runtime": bool(tree and _env.has_x280_runtime(tree)),
        "x280_compiler": tree and _env.x280_compiler(tree),
        "cards": process.card_states(cards, root, dev),
    }


def card_rows(states):
    """The doctor's rows of the cards: one a card with what its driver files say, and
    a row saying what keeps a card from opening, when something does."""
    if not states:
        return [("cards", "none (no driver entry, no /dev/tenstorrent node)")]
    rows = []
    for card, state in states.items():

        def value(key, unit="", state=state):
            return "missing" if state[key] is None else f"{state[key]}{unit}"

        before, after = state["heartbeat"]
        if None in (before, after):
            beat = "missing"
        else:
            beat = "advancing" if before != after else f"frozen at {after}"
        node = "" if state["node_exists"] else " missing"
        holder = state["holder"]
        facts = (
            f"type {value('type')}, firmware {value('firmware')}, AICLK "
            f"{value('aiclk', ' MHz')}, heartbeat {beat}, node {state['node']}{node}, "
            + ("not held" if holder is None else f"held by process {holder}")
        )
        if not state["entry"]:
            facts = f"no driver entry, node {state['node']}{node}"
        rows.append((f"card {card}", facts))
        said = process.card_problems(state)
        if said:
            reset = process.reset(card, state)
            rows.append(("", f"NOT READY: {'; '.join(said)}: {reset}"))
    return rows


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
        ("package", report["package_directory"]),
        ("include", found(report["include_directory"], "not found")),
        ("tt-metal", found(tree, unset)),
        ("from", source),
        ("build", found(report["build_directory"], "not found" if home else "-")),
        ("x280 runtime", runtime if tree else "-"),
        ("x280 compiler", found(report["x280_compiler"], "not found" if tree else "-")),
        ("card setup", "once per card: kmr setup"),
        *card_rows(report["cards"]),
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
    except build.BuildError as error:
        print(f"kmr build: {error}", file=sys.stderr)
        return 2
    return 0


def run_setup(args):
    """`kmr sfpi`, `kmr prime` or `kmr setup` (both, in that order) through the
    runtime package; returns the exit code."""
    if prepare is None or _env.runtime_version() is None:
        print(NEEDS_RUNTIME.format(args.command), file=sys.stderr)
        if args.command != "sfpi":
            print(TREE_PRIME, file=sys.stderr)
        return 2
    if args.command in ("sfpi", "setup"):
        print(f"compiler  {prepare.fetch_sfpi()}")
    if args.command == "sfpi":
        return 0
    card = prepare.default_card() if args.card is None else args.card
    lines, code = prepare.prime(card, "kmr sfpi")
    print("\n".join(lines), file=sys.stderr if code else sys.stdout)
    return code


def main(argv=None):
    """Parses `argv` and runs the chosen subcommand; returns the exit code."""
    parser = argparse.ArgumentParser(prog="kmr", description="Komorebi tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report the host without opening a device")
    sub.add_parser("sfpi", help="fetch the X280 compiler into the cache")
    for name, text in (
        ("prime", "prime a card's X280 memory, once per card"),
        ("setup", "once per card: the compiler (sfpi), then the card (prime)"),
    ):
        step = sub.add_parser(name, help=text)
        step.add_argument("--card", type=int, help="the N of /dev/tenstorrent/N")

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

    checker = sub.add_parser(
        "dram-check", help="write a card's DRAM with random data and read it back"
    )
    checker.add_argument("--card", type=int, required=True, help="the card's index")
    checker.add_argument("--megabytes", type=float, default=dram_check.MEGABYTES)
    checker.add_argument("--seed", type=int, default=0)
    checker.add_argument(
        "--all", action="store_true", help="as much DRAM as the allocator gives"
    )

    prebuild.add_arguments(
        sub.add_parser(mock.COMMAND, help="build a script's programs with no card")
    )

    args = parser.parse_args(argv)
    if args.command == mock.COMMAND:
        return prebuild.run_cli(args)
    if args.command == "power":
        return power.run_cli(args, sys.stdout)
    if args.command == "dram-check":
        return dram_check.run_cli(args, sys.stdout)
    if args.command == "doctor":
        print(format_report(doctor_report()))
        return 0
    if args.command in ("sfpi", "prime", "setup"):
        return run_setup(args)
    return run_build(args)


if __name__ == "__main__":
    sys.exit(main())
