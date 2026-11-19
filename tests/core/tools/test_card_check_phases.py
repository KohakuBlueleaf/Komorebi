"""The prebuild gate in two phases: two lines that share one NEW kernel, prebuilt two at
a time, both end with nothing compiled at the second pass (a file's time says compiled,
so a checking run that overlapped the other line's compile was blamed for it)."""

import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from komorebi.device import native
from komorebi.tools import card_check

SCRIPT = """
import numpy as np

import komorebi as km

BODY = "copy_init(tile_buffer);\\ncopy_tile(tile_buffer, tile_index, 0);  // {token}"


class Scaled(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for tile in km.Stream(a.pages(0, 1)):
            scaled = km.cxx(BODY, takes=[tile], gives=km.Tile)
            results.push(scaled)


with km.Session(log=print) as session:
    source = session.tensor(np.zeros((32, 32), np.float32), "bfloat16", "tile")
    target = session.tensor(np.zeros((32, 32), np.float32), "bfloat16", "tile")
    session.load(Scaled(a=source, out=target))
"""


def cache_of_its_own(tmp_path, monkeypatch):
    """A KMR_CACHE of the test's own linking the shared one's whole entries (its host
    module is built first; no sizes store, no ".*" build folder, no empty module-*),
    and a TT_METAL_CACHE of its own: another test's kernel then is not its compile.
    """
    native.module_path()
    folder = tmp_path / "cache"
    folder.mkdir()
    for entry in native.cache_directory().iterdir():
        if entry.name.startswith((".", "kernel_buffer")):
            continue
        if (
            entry.name.startswith("module-")
            and not (entry / native.MODULE_FILE).is_file()
        ):
            continue
        (folder / entry.name).symlink_to(entry)
    monkeypatch.setenv("KMR_CACHE", str(folder))
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "ttcache"))
    root = str(Path(sys.modules["komorebi"].__file__).parents[1])
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])
    )


@pytest.mark.tree
def test_two_lines_sharing_one_new_kernel_both_compile_nothing_at_the_second_pass(
    tmp_path, monkeypatch
):
    cache_of_its_own(tmp_path, monkeypatch)
    text = SCRIPT.replace("{token}", uuid.uuid4().hex)  # a kernel no cache has
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(text)
    checks = [card_check.Check("1", name, ()) for name in ("a.py", "b.py")]
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    results = list(card_check.prebuilt(checks, tmp_path, log_dir, jobs=2))
    assert len(results) == 2
    for check, compiled, _, code, _, _first, why in results:
        assert (compiled, why, code) == (0, "", 0), (check.script, compiled, why)
    assert sum(result[5] for result in results) > 0  # the building runs compiled it


def test_no_checking_run_starts_before_every_building_run_has_ended(
    tmp_path, monkeypatch
):
    events, seen = [], {}
    lock = threading.Lock()

    def fake_run(check, root):
        """A prebuild that takes as long as its script's name says (a, b, c: 0.1 s,
        0.6 s, 0.2 s), and compiles on its first run only."""
        with lock:
            seen[check.script] = run = seen.get(check.script, 0) + 1
        events.append((time.monotonic(), "start", run, check.script))
        time.sleep({"a.py": 0.1, "b.py": 0.6, "c.py": 0.2}[check.script])
        events.append((time.monotonic(), "end", run, check.script))
        compiled = 3 if run == 1 else 0
        text = f"built: 0 found, {compiled} compiled (kernels and parts)\n"
        return text + "kmr prebuild: 1 load(s) built\n", 0, 0.1

    monkeypatch.setattr(card_check, "prebuild_run", fake_run)
    checks = [card_check.Check("1", name, ()) for name in ("a.py", "b.py", "c.py")]
    results = list(card_check.prebuilt(checks, tmp_path, tmp_path, jobs=3))
    assert [(r[1], r[-1]) for r in results] == [(0, "")] * 3
    ended = max(t for t, what, run, _ in events if what == "end" and run == 1)
    started = min(t for t, what, run, _ in events if what == "start" and run == 2)
    assert started >= ended  # the interleaved gate starts a.py's second at 0.1 s
    assert sum(r[5] for r in results) == 9  # the building runs' compiles, not counted


def test_a_check_that_failed_to_build_has_no_checking_run(tmp_path, monkeypatch):
    runs = []

    def fake_run(check, root):
        runs.append(check.script)
        if check.script == "bad.py":
            return "error: no\n", 1, 0.1
        return "built: 0 found, 0 compiled\nkmr prebuild: 1 load(s) built\n", 0, 0.1

    monkeypatch.setattr(card_check, "prebuild_run", fake_run)
    checks = [card_check.Check("1", name, ()) for name in ("ok.py", "bad.py")]
    results = list(card_check.prebuilt(checks, tmp_path, tmp_path, jobs=2))
    assert sorted(runs) == ["bad.py", "ok.py", "ok.py"]
    assert sorted((r[0].script, bool(r[-1])) for r in results) == [
        ("bad.py", True),
        ("ok.py", False),
    ]
