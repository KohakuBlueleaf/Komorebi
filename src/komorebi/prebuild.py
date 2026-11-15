"""A script's programs built with no card (`kmr prebuild`): the script runs against a
mock device, one P150 or two in a mesh, and each load builds everything the card's
load would (kernels, overlay parts, the host transfer kernel, a mesh's link kernels)
into the build cache, then stops; a card run of the script finds them built
(docs/reference/host.md#building-with-no-card)."""

import argparse
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from komorebi.errors import KernelSizeError, L1BudgetError

COMMAND = "prebuild"  # the command's and the call's one spelling: kmr prebuild
ACTIVE = "KMR_PREBUILD"  # set in the script's process: its loads stop before launch
MOCK = "TT_METAL_MOCK_CLUSTER_DESC_PATH"
MEMORY = "KMR_PREBUILD_MEMORY"  # the remembered sizes a prebuild reads and keeps
REFUSED = 75  # the script's exit: its load refused by size, the size now kept
MHZ = 1000  # a mock has no clock: the cycles of a limit, never compiled in
STOPPED = (
    "stopped at the first load; later loads are not built (a script that loads again "
    "after using a run: prebuild each of its other forms, or let the card build them)"
)


# The script's process: main() of this module as imported (its Stopped the loads').
CHILD = "import sys, komorebi.prebuild as p; sys.exit(p.main())"


class Stopped(BaseException):
    """Raised by a load in a prebuild once it built everything: the script ends."""


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
        code = subprocess.run(command, env=env, check=False).returncode
        if code == REFUSED:  # as the card's next run: open at the size kept
            print(
                f"kmr {COMMAND}: again, at the size the next run opens at", flush=True
            )
            code = subprocess.run(command, env=env, check=False).returncode
    return code


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
    sys.path.insert(0, str(script.resolve().parent))
    kept = Path(os.environ[MEMORY]) if os.environ.get(MEMORY) else None
    before = kept.read_bytes() if kept and kept.exists() else None
    try:
        runpy.run_path(str(script), run_name="__main__")
    except (KernelSizeError, L1BudgetError):
        if kept is None or not kept.exists() or kept.read_bytes() == before:
            raise
        print(f"kmr {COMMAND}: refused by size, as the card's run would be")
        return REFUSED
    except Stopped:
        print(f"kmr {COMMAND}: {STOPPED}", file=sys.stderr)
        return 0
    print(f"kmr {COMMAND}: the script ended without a load", file=sys.stderr)
    return 0
