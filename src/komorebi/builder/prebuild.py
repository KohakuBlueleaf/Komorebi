"""A script's programs built with no card (`kmr prebuild`): the script runs on a mock
P150 or two, each load builds what the card's would into the build cache, and a mock
run stands in for its launch (docs/reference/host.md#building-with-no-card)."""

import argparse
import builtins
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import sysconfig
import tempfile
import traceback
from pathlib import Path

from komorebi.device import remembered
from komorebi.device.mock import (
    ACTIVE,
    COMMAND,
    LOADS,
    MEMORY,
    MOCK,
    SESSIONS,
    active,
    descriptor,
)
from komorebi.errors import KernelSizeError, L1BudgetError

REFUSED = 75  # the script's exit: its load refused by size, the size now kept
STOPPED = 76  # the script's exit: an error stopped it before its end (no prebuild)
EXITED = 77  # the script's exit: non-zero, or its own failed assert, with no load built
CUT = 400  # a stop's message is cut here, but for the compiler's first error line
PASSES = 6  # passes at most: each opens at a size the passes before it did not
LIMIT = (
    "a prebuild follows the path of a run in which every command succeeds (its reads "
    "give zeros): a script that branches on what it read may build other loads than "
    "on the card"
)
# (module, function) of the loads that build: an error raised in one is a failure
LOAD_FRAMES = {
    ("komorebi.session.parts.load", "_load"),
    ("komorebi.session.parts.load", "load_islands"),
}
BESIDE = ".prebuild"  # a file the script writes goes beside it: name + BESIDE
# a source the toolchain reads keeps its name: generated during the run, no result
SOURCES = (".h", ".hpp", ".c", ".cc", ".cpp", ".S", ".ld")

# The script's process: main() of this module as imported (the loads' LOADS).
CHILD = "import sys, komorebi.builder.prebuild as p; sys.exit(p.main())"


def run(script, args=(), cards=(0,), cluster=None, arch="blackhole"):
    """Runs `script` with `args` in a new process against a mock of `cards` on chips
    of `arch` (cluster: another descriptor); its loads build and stop. Returns the
    process's exit code."""
    env = dict(os.environ, **{ACTIVE: "1"})
    env[MOCK] = str(cluster or descriptor(cards, arch))
    env["TT_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    command = [sys.executable, "-c", CHILD, str(script), *args]
    with tempfile.TemporaryDirectory() as folder:
        env[MEMORY] = str(Path(folder) / "kernel_buffer.json")

        def one_pass():
            return subprocess.run(command, env=env, check=False)

        code = converge(one_pass, env[MEMORY])
        remembered.seed(Path(env[MEMORY]))
    return code


def refusals_path(memory=None):
    """The refused loads of a prebuild's passes, one JSON line each, beside its copy
    of the remembered sizes (`memory`, else the process's)."""
    return Path(memory or os.environ[MEMORY]).with_suffix(".refused")


def scripts_kept(memory):
    """The script entries of a prebuild's copy of the remembered sizes ({} if none)."""
    path = Path(memory) if memory else None
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text()).get("scripts", {})


def converge(one_pass, memory, log=lambda line: print(line, flush=True)):
    """Runs passes (`one_pass()`: the script's process) until a pass opens at the size
    the card's next run opens at: none refused by size and the script's entry what it
    was at its start (what a load builds depends on the L1 left). Two loads that
    cannot share one size: said with both, and exit 1."""
    seen, passes = [], 1
    before = scripts_kept(memory)
    code = one_pass().returncode
    while code == REFUSED or scripts_kept(memory) != before:
        if code == REFUSED:
            lines = refusals_path(memory).read_text().splitlines()
            now = json.loads(lines[-1])
            conflict = next((one for one in seen if one["kind"] != now["kind"]), None)
            if conflict is None and any(one["opens"] == now["opens"] for one in seen):
                conflict = seen[-1]
            if conflict is not None:
                grows, shrinks = sorted((conflict, now), key=lambda one: one["kind"])
                log(
                    f"kmr {COMMAND}: the script's loads cannot share one kernel "
                    f"buffer: {grows['load']} (line {grows['line']}) needs "
                    f"{grows['opens']:,} B, {shrinks['load']} (line {shrinks['line']}) "
                    f"fits only at {shrinks['opens']:,} B or less; give them sessions "
                    "of their own"
                )
                return 1
            seen.append(now)
        if passes == PASSES:
            log(f"kmr {COMMAND}: the kernel buffer did not settle in {PASSES} passes")
            return 1
        log(f"kmr {COMMAND}: again, at the size the next run opens at")
        before, passes = scripts_kept(memory), passes + 1
        code = one_pass().returncode
    return code


def refused_load(error, script, kept, before):
    """Appends the refused load to the passes' record: its program, its line in the
    script, grows or shrinks, and the size the next pass opens at (the copy's
    script entry that changed)."""
    frames = traceback.extract_tb(error.__traceback__)
    line = next((f.lineno for f in reversed(frames) if Path(f.filename) == script), 0)
    old = json.loads(before or b"{}").get("scripts", {})
    new = json.loads(kept.read_bytes()).get("scripts", {})
    sizes = [size for key, size in new.items() if old.get(key) != size] or [0]
    opens = max(sizes)
    kind = "grows" if isinstance(error, KernelSizeError) else "shrinks"
    record = {"load": str(error).split(":")[0], "line": line, "kind": kind}
    with refusals_path().open("a") as out:
        out.write(json.dumps(record | {"opens": opens}) + "\n")


def cut(message):
    """`message` to CUT characters, and the compiler's first error line whole if the
    cut left it out (a failed build step's output follows its command)."""
    if len(message) <= CUT:
        return message
    first = next((s for s in message.splitlines() if "error:" in s), None)
    kept = message[:CUT]
    return kept if first is None or first in kept else f"{kept} ... {first.strip()}"


def in_load(error):
    """Whether `error` was raised inside a load (a build), not by the script: a frame
    of LOAD_FRAMES (a module's function, whatever its file's name or place)."""
    return any(
        (frame.f_globals.get("__name__"), frame.f_code.co_name) in LOAD_FRAMES
        for frame, _ in traceback.walk_tb(error.__traceback__)
    )


def add_arguments(parser):
    """The command's arguments (kmr's subcommand)."""
    parser.add_argument("--cards", default="0", help="0, or 0,1 for a mesh of two")
    parser.add_argument("--cluster", type=Path, help="another mock cluster descriptor")
    parser.add_argument(
        "--arch",
        choices=("blackhole", "wormhole_b0"),
        default="blackhole",
        help="the chips of the mock: a P150 card, or one Wormhole chip of a Galaxy",
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)


def run_cli(args):
    cards = [int(card) for card in args.cards.split(",")]
    return run(args.script, args.args, cards, args.cluster, args.arch)


@contextlib.contextmanager
def writes_beside():
    """From the first session on, a file written outside komorebi goes to
    `<file>.prebuild` (its reads gave zeros: a card run's results must not take them),
    and a move of it moves that; a write before any session or of a source (SOURCES)
    keeps its name; a missing file with a `.prebuild` reads it and exists."""
    real = builtins.open
    tests = os.path.exists, os.path.isfile, Path.exists, Path.is_file
    stdlib = sysconfig.get_paths()["stdlib"]
    own = str(Path(__file__).resolve().parents[1])  # the komorebi package

    def caller():
        """The first frame past the libraries' (the hook's callers there): its file."""
        frame = sys._getframe(2)
        while frame and library(frame.f_code.co_filename):
            frame = frame.f_back
        return frame.f_code.co_filename if frame else ""

    def library(name):
        return name.startswith(stdlib) or "site-packages" in name

    def beside(name):
        """`name`'s .prebuild when only it is there (a read takes it), else None."""
        if tests[0](name) or not tests[0](f"{name}{BESIDE}"):
            return None
        return f"{name}{BESIDE}"

    def opened(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and not caller().startswith(own):
            name = os.fspath(file)
            if set(mode) & set("wax"):
                if SESSIONS[0] and not name.endswith(SOURCES):
                    file = f"{name}{BESIDE}"
            else:
                file = beside(name) or file
        return real(file, mode, *args, **kwargs)

    def tested(test):
        """`test` (exists, isfile) as a read finds files: a .prebuild alone counts."""

        def answer(path, *args, **kwargs):
            if test(path, *args, **kwargs):
                return True
            if not isinstance(path, (str, os.PathLike)) or caller().startswith(own):
                return False
            there = beside(os.fspath(path))
            there = there if there is None or isinstance(path, str) else Path(there)
            return there is not None and test(there, *args, **kwargs)

        return answer

    moves = os.replace, os.rename

    def moved(move):
        """`move` (replace, rename) as a write finds files: a source written beside
        (`.prebuild` alone) moves to its target's `.prebuild` (a write under a name of
        its own, then moved: the atomic form, which two makers may use at once)."""

        def answer(source, target, *args, **kwargs):
            names = (source, target)
            if (
                all(isinstance(name, (str, os.PathLike)) for name in names)
                and not caller().startswith(own)
                and not os.path.lexists(source)
                and os.path.lexists(f"{os.fspath(source)}{BESIDE}")
            ):
                names = (f"{os.fspath(name)}{BESIDE}" for name in names)
            return move(*names, *args, **kwargs)

        return answer

    builtins.open = io.open = opened
    os.path.exists, os.path.isfile, Path.exists, Path.is_file = map(tested, tests)
    os.replace, os.rename = map(moved, moves)
    try:
        yield
    finally:
        builtins.open = io.open = real
        os.path.exists, os.path.isfile, Path.exists, Path.is_file = tests
        os.replace, os.rename = moves


def main():
    """The script's process: runs it as __main__ until its first load stops."""
    if not active() or len(sys.argv) < 2:
        sys.exit(f"usage: kmr {COMMAND} [--cards 0|0,1] script.py [args]")
    # as the interpreter runs it: __file__ absolute, sys.argv[0] as typed
    script = Path(sys.argv[1]).absolute()
    sys.argv = sys.argv[1:]
    said = f"kmr {COMMAND}: "
    sys.path.insert(0, str(script.resolve().parent))
    kept = Path(os.environ[MEMORY]) if os.environ.get(MEMORY) else None
    before = kept.read_bytes() if kept and kept.exists() else None
    try:
        with writes_beside():
            runpy.run_path(str(script), run_name="__main__")
    except (KernelSizeError, L1BudgetError) as error:
        if kept is None or not kept.exists() or kept.read_bytes() == before:
            raise
        refused_load(error, script, kept, before)
        print(f"{said}refused by size, as the card's run would be", flush=True)
        return REFUSED
    except Exception as error:  # the script's own checks on zeros, or a build: stops it
        if in_load(error):
            raise  # a build or a load failed: as it would on the card
        frames = traceback.extract_tb(error.__traceback__)
        line = next(
            (f.lineno for f in reversed(frames) if Path(f.filename) == script), "?"
        )
        what = cut(f"{type(error).__name__}: {error}")
        if isinstance(error, AssertionError) and Path(frames[-1].filename) == script:
            # its own assert, on zeros: a check of the script's, as its own exit
            print(
                f"{said}the script's assert at line {line} failed after {LOADS[0]} "
                f"load(s) ({what})",
                file=sys.stderr,
            )
            return 0 if LOADS[0] else EXITED
        print(
            f"{said}the script stopped at line {line} after {LOADS[0]} load(s) "
            f"({what})",
            file=sys.stderr,
        )
        return STOPPED
    except SystemExit as exit:  # the script's own exit
        if exit.code not in (None, 0):
            print(f"{said}the script exited {exit.code}", file=sys.stderr)
            if not LOADS[0]:  # nothing built: its error was not a check on zeros
                print(f"{said}0 load(s) built", file=sys.stderr)
                return EXITED
    print(f"{said}{LOADS[0]} load(s) built; {LIMIT}", file=sys.stderr)
    return 0
