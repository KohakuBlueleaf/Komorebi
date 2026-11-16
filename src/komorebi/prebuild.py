"""A script's programs built with no card (`kmr prebuild`): the script runs on a mock
P150 or two, each load builds what the card's would into the build cache, and a mock
run stands in for its launch (docs/reference/host.md#building-with-no-card)."""

import argparse
import json
import os
import runpy
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from komorebi.errors import KernelSizeError, L1BudgetError

COMMAND = "prebuild"  # the command's and the call's one spelling: kmr prebuild
ACTIVE = "KMR_PREBUILD"  # set in the script's process: its loads build, then mock
MOCK = "TT_METAL_MOCK_CLUSTER_DESC_PATH"
MEMORY = "KMR_PREBUILD_MEMORY"  # the remembered sizes a prebuild reads and keeps
REFUSED = 75  # the script's exit: its load refused by size, the size now kept
PASSES = 6  # passes at most: each opens at a size the passes before it did not
MHZ = 1000  # a mock has no clock: the cycles of a limit, never compiled in
LIMIT = (
    "a mock run's calls return at once and its reads give zeros: a script that "
    "branches on what it read may build other loads than on the card"
)
LOADS = [0]  # the loads this process built

# The script's process: main() of this module as imported (the loads' LOADS).
CHILD = "import sys, komorebi.prebuild as p; sys.exit(p.main())"


class Nothing:
    """What a mock run's call gives: any field, call or item is itself, a number is
    0, it iterates over nothing and is true (a loop waiting for it ends)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

    __index__ = __int__

    def __format__(self, spec):
        try:
            return format(0.0, spec)
        except ValueError:
            return format(0, spec)

    def __arith__(self, other):
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __arith__
    __truediv__ = __rtruediv__ = __floordiv__ = __mod__ = __pow__ = __arith__


NOTHING = Nothing()


class Done(Nothing):
    """What a mock run's wait gives: a done event (its kind "done"; an island run's
    parts, itself), else Nothing."""

    kind = "done"
    parts = property(lambda self: (self,))


DONE = Done()


class Run:
    """The run a prebuild's load returns in place of the card's: every call returns
    at once with Nothing, a wait with a done event; stop and abort end its `parts`."""

    def __init__(self, counted=True, parts=()):
        self.stopped, self.parts = False, list(parts)
        LOADS[0] += int(counted)

    def stop(self, timeout=None):
        self.stopped = True
        for part in self.parts:
            part.stop()

    abort = stop

    def alive(self):
        return not self.stopped

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return (lambda *args, **kwargs: DONE) if name.startswith("wait") else NOTHING


def active():
    """Whether this process is a prebuild's script."""
    return bool(os.environ.get(ACTIVE))


def memory(real):
    """The remembered sizes' file a prebuild reads and writes in place of `real` (a
    copy of it, the prebuild's own: the card's file is never written), else None."""
    if not active() or not os.environ.get(MEMORY):
        return None
    own = Path(os.environ[MEMORY])
    if not own.exists() and real.is_file():
        own.write_bytes(real.read_bytes())
    return own


def descriptor(cards):
    """The mock cluster of `cards` (one or two P150 cards), shipped in the package."""
    if len(cards) not in (1, 2):
        raise ValueError(f"cards {list(cards)}: one card or two")
    name = "p150.yaml" if len(cards) == 1 else "p150x2.yaml"
    return Path(__file__).resolve().parent.parent / "komorebi_native" / "mock" / name


class Dispatch:
    """The X280 dispatch a mock lacks: what a load asks of it before the launch (the
    grid, the core table, the dispatcher's tile and LIM base: runtime arguments)."""

    idle, close_failed, lim_base, l2_x, l2_y = True, "", 0x20000000, 8, 3
    settle_microseconds = 0.0

    def __init__(self, device):
        self.device = device
        self.grid_x, self.grid_y = device.grid()

    def prepare(self, *args, **kwargs):
        return 1

    def core_table(self):
        cores = []
        for index in range(self.grid_x * self.grid_y):
            x, y = self.device.worker_core(index % self.grid_x, index // self.grid_x)
            cores.append(x | y << 16)
        return np.array(cores + [0] * (128 - len(cores)), dtype=np.uint64)

    def work_info(self, cluster):
        return {"virtual_x": self.l2_x, "virtual_y": self.l2_y, "lim_base": 0}

    def close(self):
        return True


def run(script, args=(), cards=(0,), cluster=None):
    """Runs `script` with `args` in a new process against a mock of `cards` (cluster:
    another descriptor); its loads build and stop. Returns the process's exit code."""
    env = dict(os.environ, **{ACTIVE: "1"})
    env[MOCK] = str(cluster or descriptor(cards))
    env["TT_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    command = [sys.executable, "-c", CHILD, str(script), *args]
    with tempfile.TemporaryDirectory() as folder:
        env[MEMORY] = str(Path(folder) / "kernel_buffer.json")

        def one_pass():
            return subprocess.run(command, env=env, check=False)

        code = converge(one_pass, env[MEMORY])
        from komorebi import remembered  # noqa: PLC0415 (remembered imports us)

        remembered.seed(Path(env[MEMORY]))
    return code


def refusals_path(memory=None):
    """The refused loads of a prebuild's passes, one JSON line each, beside its copy
    of the remembered sizes (`memory`, else the process's)."""
    return Path(memory or os.environ[MEMORY]).with_suffix(".refused")


def converge(one_pass, memory, log=lambda line: print(line, flush=True)):
    """Runs passes (`one_pass()`: the script's process) until no load is refused by
    size: each opens at the size kept by the one before, as the card's next run
    would. Two loads that cannot share one size: said with both, and exit 1."""
    seen, code = [], one_pass().returncode
    while code == REFUSED:
        lines = refusals_path(memory).read_text().splitlines()
        now = json.loads(lines[-1])
        conflict = next((one for one in seen if one["kind"] != now["kind"]), None)
        if conflict is None and any(one["opens"] == now["opens"] for one in seen):
            conflict = seen[-1]
        if conflict is not None:
            grows, shrinks = sorted((conflict, now), key=lambda one: one["kind"])
            log(
                f"kmr {COMMAND}: the script's loads cannot share one kernel buffer: "
                f"{grows['load']} (line {grows['line']}) needs {grows['opens']:,} B, "
                f"{shrinks['load']} (line {shrinks['line']}) fits only at "
                f"{shrinks['opens']:,} B or less; give them sessions of their own"
            )
            return 1
        if len(seen) + 1 == PASSES:
            log(f"kmr {COMMAND}: the kernel buffer did not settle in {PASSES} passes")
            return 1
        seen.append(now)
        log(f"kmr {COMMAND}: again, at the size the next run opens at")
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


def in_load(error):
    """Whether `error` was raised inside a load (a build), not by the script."""
    return any(
        frame.name in ("_load", "load_islands")
        and frame.filename.endswith("session.py")
        for frame in traceback.extract_tb(error.__traceback__)
    )


def add_arguments(parser):
    """The command's arguments (kmr's subcommand)."""
    parser.add_argument("--cards", default="0", help="0, or 0,1 for a mesh of two")
    parser.add_argument("--cluster", type=Path, help="another mock cluster descriptor")
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)


def run_cli(args):
    cards = [int(card) for card in args.cards.split(",")]
    return run(args.script, args.args, cards, args.cluster)


def main():
    """The script's process: runs it as __main__ until its first load stops."""
    if not active() or len(sys.argv) < 2:
        sys.exit(f"usage: kmr {COMMAND} [--cards 0|0,1] script.py [args]")
    script = Path(sys.argv[1])
    sys.argv = sys.argv[1:]
    said = f"kmr {COMMAND}: "
    sys.path.insert(0, str(script.resolve().parent))
    kept = Path(os.environ[MEMORY]) if os.environ.get(MEMORY) else None
    before = kept.read_bytes() if kept and kept.exists() else None
    try:
        runpy.run_path(str(script), run_name="__main__")
    except (KernelSizeError, L1BudgetError) as error:
        if kept is None or not kept.exists() or kept.read_bytes() == before:
            raise
        refused_load(error, script, kept, before)
        print(f"{said}refused by size, as the card's run would be", flush=True)
        return REFUSED
    except Exception as error:  # the script's own checks on zeros, or a build
        if in_load(error):
            raise  # a build or a load failed: as it would on the card
        frames = traceback.extract_tb(error.__traceback__)
        line = next(
            (f.lineno for f in reversed(frames) if Path(f.filename) == script), "?"
        )
        print(
            f"{said}the script stopped at line {line} after {LOADS[0]} load(s) "
            f"({type(error).__name__}: {error})"[:400],
            file=sys.stderr,
        )
    except SystemExit as exit:  # the script's own exit
        if exit.code not in (None, 0):
            print(f"{said}the script exited {exit.code}", file=sys.stderr)
    print(f"{said}{LOADS[0]} load(s) built; {LIMIT}", file=sys.stderr)
    return 0
