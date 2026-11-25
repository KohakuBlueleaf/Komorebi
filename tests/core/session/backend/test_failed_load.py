"""A load that fails before its launch gives back the L1 it took, on a card and on a
card split in two islands: the rows a later build counts are the ones before it, and a
program that fits loads after it (a mock session in the tree lane)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import komorebi as km
from komorebi.device import mock

# Large's kernels (40 handlers) are over the given kernel buffer, Small's within it;
# the rows after the refusal are read while its error is held, as a merge's rebuild does
LOADS = """
import json, sys
import numpy as np
import komorebi as km
from komorebi.dsl import program
from komorebi.errors import KernelSizeError


class Small(km.Program):
    @km.handler
    def drain(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def drained(self, source: km.Tensor):
    km.Stream(source.pages(self.rank, 4)).drain()


Large = type(
    "Large", (km.Program,), {f"h{i}": km.handler(drained) for i in range(40)}
)
split = sys.argv[2] == "islands"
opened = {"card": 0, "islands": 2, "graph": [(0, 1)]} if split else {}


def rows(session):
    each = getattr(session, "sessions", None) or [session]
    return [one.l1_rows(program.Unbuilt()) for one in each]


seen = {}
with km.Session(kernel_bytes=122880 if split else 40960, **opened) as session:
    zeros = np.zeros((4096, 32), np.float32)
    copied = {"copy": True} if split else {}
    source = session.tensor(zeros, "bfloat16", "tile", **copied)
    seen["before"] = rows(session)
    try:
        session.load(Large(session=session, log=lambda line: None, source=source))
    except KernelSizeError as error:
        seen["refused"] = str(error)
        seen["after"] = rows(session)
    small = Small(session=session, log=lambda line: None, source=source)
    session.load(small).stop()
    seen["small"] = small.sizes.total
    seen["loaded"] = rows(session)
with open(sys.argv[1], "w") as out:
    json.dump(seen, out)
"""


def loads(tmp_path, form):
    """What LOADS saw on a mock card ("card") or on one split in two ("islands")."""
    env = dict(os.environ, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor((0,), "blackhole"))
    env[mock.MEMORY] = str(tmp_path / "kernel_buffer.json")
    env["TT_VISIBLE_DEVICES"] = "0"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(km.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    out = tmp_path / "seen.json"
    done = subprocess.run(
        [sys.executable, "-c", LOADS, str(out), form],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(out.read_text())


def bounds_aside(rows):
    """A core's rows by name, the page bounds list aside (it grows by an entry of 32 B
    for each DRAM tensor of the session, the state area the load makes among them)."""
    rows = dict(rows)
    return rows.pop("page bounds list"), rows


@pytest.mark.tree
def test_a_load_refused_for_its_kernels_gives_back_its_l1(tmp_path):
    seen = loads(tmp_path, "card")
    assert "of the 40960 B kernel buffer" in seen["refused"]
    (before,), (after,) = seen["before"], seen["after"]
    bounds, before = bounds_aside(before)
    bounds_after, after = bounds_aside(after)
    assert bounds_after - bounds == 32  # the state area; nothing else differs
    assert after == before
    assert 0 < seen["small"] <= 40960  # and a program that fits loads after it


@pytest.mark.tree
def test_a_load_over_islands_refused_for_its_kernels_gives_back_its_l1(tmp_path):
    seen = loads(tmp_path, "islands")
    assert "of the 122880 B kernel buffer" in seen["refused"]
    assert len(seen["after"]) == len(seen["loaded"]) == 2
    for after, loaded in zip(seen["after"], seen["loaded"], strict=True):
        # what any load leaves: the session's own tables, made at its first load
        # (rows "made at the first load" before it), and none of the refused load's
        assert bounds_aside(after)[1] == bounds_aside(loaded)[1]
    assert 0 < seen["small"] <= 122880
