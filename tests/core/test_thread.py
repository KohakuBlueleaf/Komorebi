"""The X280 thread's host side: argument block, region table and ring line, the result
block, the host fallback, the stop of a thread that does not return, and the ring
calls against a fake kernel on the CPU."""

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from komorebi import thread as kt

KMX = kt.native.kmx_dir()
K = kt.layout()


def a_thread(**kw):
    regions = {"counts": np.arange(10, dtype=np.uint32), "out": 5000}
    return kt.Thread(sources=["t.c"], regions=regions, words=[7, 9], **kw)


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_ring_calls_against_a_fake_kernel(tmp_path):
    source = Path(__file__).resolve().parent / "kmx_ring_check.c"
    includes = [f"-I{KMX}", f"-I{KMX.parents[1] / 'include'}"]
    exe = tmp_path / "ring_check"
    cmd = ["cc", "-std=c11", "-Wall", "-Werror", *includes, str(source), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    assert subprocess.run([exe], capture_output=True, text=True).stdout == "ok\n"


def test_ring_line_names_the_placement():
    ring = kt.RingPlacement(0x1000, 0x2040, 0x3000, 0x4000, 16, 64, cached=True)
    block = kt.args_block(a_thread(), base=0, check_value=1, ring=ring)
    line = struct.unpack_from("<4Q3I", block, K["KMX_LIM_RING"])
    assert line == (0x1000, 0x2040, 0x3000, 0x4000, 16, 64, K["KMX_RING_CACHED"])


def test_argument_block_holds_the_region_table_and_words():
    block = kt.args_block(a_thread(deadline=2.0), base=0x100000, check_value=0xABC)
    assert len(block) == K["KMX_LIM_BYTES"]
    head = struct.unpack_from("<QIIQIIQQ", block)
    assert head == (
        0,
        K["KMX_MAGIC"],
        K["KMX_VERSION"],
        100_000_000,
        2,
        2,
        0x100000,
        0xABC,
    )
    table = struct.unpack_from("<4Q", block, K["KMX_ARGS_REGIONS"])
    assert table == (0x100000 + kt.PAGE, 40, 0x100000 + 2 * kt.PAGE, 5000)
    assert struct.unpack_from("<3I", block, K["KMX_ARGS_WORDS"]) == (7, 9, 0)
    assert block[K["KMX_LIM_STOP"] :] == bytes(K["KMX_LIM_BYTES"] - K["KMX_LIM_STOP"])


def test_result_block_is_read_with_names():
    words = [K["KMX_STATE_DONE"], K["KMX_ERR_NONE"], 5, 2, 3, 100, 500] + [0] * 9
    raw = struct.pack("<64I", *words, 11, 12, *([0] * 46))
    r = kt.parse_result(raw)
    assert (r.state, r.error, r.code, r.words, r.where) == (
        "done",
        "none",
        5,
        [11, 12],
        3,
    )
    assert (r.where_us, r.us) == (2.0, 10.0)


def test_host_fallback_gets_the_initial_regions_and_its_writes_are_kept():
    def double(regions, words):
        regions["out"][:10] = regions["counts"] * words[0]
        return 3, [int(regions["out"].sum())]

    run = kt.ThreadRun(a_thread(host_fallback=double), None, None, on_card=False)
    run.start()
    r = run.wait(1.0)
    assert (r.code, r.words, r.state) == (3, [315], "done")
    out = run.read("out").view(np.uint32)
    assert len(out) == 1250 and list(out[:4]) == [0, 7, 14, 21] and out[10:].sum() == 0


class FakePort:
    def __init__(self):
        self.calls = []

    def work_start(self, c, image, block):
        self.calls.append(("start", c))

    def work_wait(self, c, timeout):
        self.calls.append(("wait", c))
        return {"ready": False}

    def work_write(self, c, offset, data):
        self.calls.append(("write", offset, struct.unpack_from("<I", data)[0]))

    def work_read(self, c, offset, n):
        return struct.pack("<64I", K["KMX_STATE_RUNNING"], *([0] * 63))

    def work_stop(self, c, timeout):
        self.calls.append(("stop", c))
        return {"cleaned": True}


def test_a_thread_past_its_timeout_is_asked_to_stop_then_stopped():
    port = FakePort()
    run = kt.ThreadRun(a_thread(), port, None, image="t.bin")
    run.start()
    r = run.wait(0.1)
    stop_line = ("write", K["KMX_LIM_STOP"], K["KMX_STOP_WORD"])
    assert port.calls == [
        ("start", 1),
        ("wait", 1),
        stop_line,
        ("wait", 1),
        ("stop", 1),
    ]
    assert not r.ran and r.state == "running" and not run.running
