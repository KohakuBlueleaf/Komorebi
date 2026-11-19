"""One kernel build at a time per build cache: two processes that build the same new
kernel at the same instant compile it once (the other finds it), every load builds
under the lock, and a build waiting for the lock says so."""

import fcntl
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from komorebi.device import mock
from komorebi.session import kernels

# A mock card's process that builds the host transfer kernel with a define of its own
# (new for each token) once a file exists, as a load does: under the build lock.
BUILD = """
import contextlib, sys, time
from pathlib import Path
import komorebi as km
from komorebi import abi
from komorebi.device import native
from komorebi.session import kernels
lock = getattr(kernels, "build_lock", contextlib.nullcontext)
with km.Session(log=lambda line: None) as session:
    module = native.load_module(session.tree)
    launch = module.Launch(session.device)
    include = abi.headers().abi.parents[2]
    source = include.parent / "kernels" / "host_transfer.cpp"
    cores = [(0, 0, 0, 0)]
    launch.kernel(
        str(source), cores, 0, [], [("KM_UNIQUE", sys.argv[1])], [str(include)],
        [(0, 0, [0] * 16)], noc=0,
    )
    launch.cb(cores, 0, "uint32", 16, 2 * 4096 + 128)
    Path(sys.argv[2] + ".ready").touch()
    while sys.argv[3] != "-" and not Path(sys.argv[3]).exists():
        pass
    with lock():
        started = time.time()
        launch.build()
    found, compiled = kernels.built_files(launch.recipes(), started)
    elves = sorted({row[1] for _, rows in launch.recipes() for row in rows})
    print("BUILT", found, compiled, *elves)
"""
# A prebuild of a script, with each take of the build lock recorded to a file (written
# through a descriptor opened first: the prebuild moves files opened by name aside).
LOADS = """
import contextlib, os, sys
import komorebi.builder.prebuild as p
from komorebi.session import kernels
real = kernels.build_lock
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_APPEND)

@contextlib.contextmanager
def recorded(*args, **kwargs):
    os.write(fd, b"enter\\n")
    with real(*args, **kwargs):
        yield
    os.write(fd, b"exit\\n")

kernels.build_lock = recorded
sys.argv = [sys.argv[0], sys.argv[1]]
sys.exit(p.main())
"""
PROGRAM = """
import numpy as np

import komorebi as km

BODY = "copy_init(tile_buffer);\\ncopy_tile(tile_buffer, tile_index, 0);"


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


def environment(folder, name):
    """A child's environment: a mock of card 0, a store of its own."""
    env = dict(os.environ, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor((0,), "blackhole"))
    env[mock.MEMORY] = str(folder / f"{name}.json")
    env["TT_VISIBLE_DEVICES"] = "0"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    root = str(Path(sys.modules["komorebi"].__file__).parents[1])
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    return env


def child(folder, name, code, *args):
    return subprocess.Popen(
        [sys.executable, "-c", code, *args],
        env=environment(folder, name),
        cwd=folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def built_by(run):
    """(found, compiled, ELF paths) from a build child's BUILT line."""
    output, _ = run.communicate(timeout=900)
    assert run.returncode == 0, output[-2000:]
    words = [line for line in output.splitlines() if line.startswith("BUILT")][-1]
    found, compiled, *elves = words.split()[1:]
    return int(found), int(compiled), elves


def build_at_once(folder, token, count):
    """`count` processes building the kernel of `token` the moment all have started."""
    go = folder / "go"
    runs = [
        child(folder, f"c{n}", BUILD, token, str(folder / f"c{n}"), str(go))
        for n in range(count)
    ]
    while not all((folder / f"c{n}.ready").exists() for n in range(count)):
        time.sleep(0.01)
    go.touch()
    done = [built_by(run) for run in runs]
    go.unlink()
    return done


@pytest.mark.tree
def test_two_processes_building_one_new_kernel_compile_it_once(tmp_path):
    first, second = build_at_once(tmp_path, f"u{uuid.uuid4().hex}", 2)
    assert min(first[1], second[1]) == 0, (first[:2], second[:2])  # one found them all
    assert max(first[1], second[1]) > 0, (first[:2], second[:2])  # the other built them
    for elf in first[2] + second[2]:  # both can load what is there
        assert Path(elf).read_bytes()[:4] == b"\x7fELF"


@pytest.mark.tree
def test_a_pass_after_another_process_s_build_compiles_nothing(tmp_path):
    token = f"u{uuid.uuid4().hex}"
    (first,) = build_at_once(tmp_path, token, 1)
    assert first[1] > 0
    (second,) = build_at_once(tmp_path, token, 1)
    assert second[1] == 0, second[:2]


@pytest.mark.tree
def test_a_load_builds_its_kernels_under_the_lock(tmp_path):
    script, record = tmp_path / "card.py", tmp_path / "locks"
    script.write_text(PROGRAM)
    run = child(tmp_path, "load", LOADS, str(script), str(record))
    output, _ = run.communicate(timeout=900)
    assert run.returncode == 0, output[-2000:]
    taken = record.read_text().split()
    assert taken and taken == ["enter", "exit"] * (len(taken) // 2), taken


def test_a_build_waits_for_the_lock_and_says_so_after_a_while(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path))
    monkeypatch.setattr(kernels, "BUILD_WAIT", 0.2)
    held = (tmp_path / kernels.BUILD_LOCK).open("a")
    fcntl.flock(held, fcntl.LOCK_EX)
    said, entered = [], threading.Event()

    def build():
        with kernels.build_lock(said.append):
            entered.set()

    waiting = threading.Thread(target=build)
    waiting.start()
    assert not entered.wait(0.6)  # still waiting: it said so once
    assert said == ["waiting for another process's kernel build"]
    held.close()  # the holder is done
    assert entered.wait(30)
    waiting.join(30)
    with kernels.build_lock(said.append):  # free again: no line
        pass
    assert len(said) == 1


def test_the_lock_is_in_the_tt_metal_cache_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "cache"))
    with kernels.build_lock():
        assert (tmp_path / "cache" / kernels.BUILD_LOCK).is_file()
