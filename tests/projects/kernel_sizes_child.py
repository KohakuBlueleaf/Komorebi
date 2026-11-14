"""Runs a driver script (or `--program module:function`) with `Session.load` cut down
to its kernel build on a mock device (no card), and prints the sizes as one
`SIZES {json}` line: kernel_sizes_child.py script.py [arguments].
"""

import importlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np

from komorebi import native, programs
from komorebi import session as km_session
from komorebi.program import Program


class Built(BaseException):
    """Raised after the first build, to stop the driver script."""


class FakeDispatch:
    """What `describe` reads of the X280 dispatcher: the grid, the core table and
    addresses that only reach runtime arguments, never the kernels' text."""

    lim_base = 0x20000000
    l2_x, l2_y = 8, 3

    def __init__(self, device):
        self.device = device
        self.grid_x, self.grid_y = device.grid()

    def core_table(self):
        cores = self.grid_x * self.grid_y
        table = np.zeros(128, dtype=np.uint64)
        for core in range(cores):
            x, y = self.device.worker_core(core % self.grid_x, core // self.grid_x)
            table[core] = x | y << 16
        return table

    def work_info(self, cluster):
        return {"virtual_x": 8, "virtual_y": 3, "lim_base": self.lim_base}


def build_only(self, program, depth=None, trace=0, **_):
    """`Session.load` up to the runtime's kernel layout; prints the sizes and stops."""
    placed = None
    if not isinstance(program, Program) or program.form() == "program":
        grid = self.grid
        placed = programs.plan(program, grid[0] * grid[1])
        program = placed.programs[0]
    owner = placed.threads_owner() if placed is not None else program
    sources = program.sources()
    if placed is not None:
        sources = programs.runtime_kernels()
    self._module = native.load_module(self.tree)
    dispatch = FakeDispatch(self.device)
    backend = km_session._Backend(self, dispatch, 1000, trace)
    rings = {cluster: (0, 0) for cluster in owner.threads}
    _, port_lines = km_session._port_lines(owner, dispatch, rings)
    descriptor = backend.describe(
        program, sources, 1, depth or program.depth, 2.0, 0.0, trace, port_lines, placed
    )
    _, *build = descriptor.build()
    sizes = km_session.kernel_sizes(build)
    row = {
        "program": program.name,
        "text": sizes.text,
        "total": sizes.total,
        "limit": sizes.limit,
        "fits": not build[-1],
    }
    print("SIZES", json.dumps(row), flush=True)
    raise Built


def load_program(spec):
    """Loads the program that `module:function` returns, in a default session."""
    module, function = spec.split(":")
    with km_session.Session() as session:
        session.load(getattr(importlib.import_module(module), function)())


def main():
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    km_session.Session.load = build_only
    try:
        if sys.argv[1] == "--program":
            load_program(sys.argv[2])
            return
        script = Path(sys.argv[1]).resolve()
        sys.argv = sys.argv[1:]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
    except Built:
        pass


if __name__ == "__main__":
    main()
