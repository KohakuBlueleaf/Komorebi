"""The X280 thread's host side: argument block, region table and ring line, the result
block, the host fallback, the stop of a thread that does not return, and the ring
calls against a fake kernel on the CPU."""

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fake_port import FakeBackend
from fake_port import FakePort as FakeKernel

from komorebi import thread as kt
from komorebi.errors import ThreadFailed
from komorebi.program import Program
from komorebi.run import Run

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


def test_the_threads_wake_is_the_kernels():
    assert K["KMX_EV_WAKE"] == kt.abi.headers()["KMD_EV_WAKE"]


def test_ring_line_names_the_placement():
    ring = kt.RingPlacement(0x1000, 0x2040, 0x3000, 0x4000, 16, 64, cached=True)
    block = kt.args_block(a_thread(), base=0, check_value=1, ring=ring)
    line = struct.unpack_from("<4Q3I", block, K["KMX_LIM_RING"])
    assert line == (0x1000, 0x2040, 0x3000, 0x4000, 16, 64, K["KMX_RING_CACHED"])


def test_a_work_ring_has_commands_in_the_kernels_lim_and_events_in_the_threads():
    ring, end = kt.work_ring(0x8100000, (8, 3), 0x8120000, (8, 9))
    d = kt.abi.headers()
    assert ring.cmd == 0x8100000 + d["KMD_LIM_BYTES"] and ring.remote == 8 | 3 << 16
    assert ring.read == ring.cmd + 64 * kt.RING_DEPTH
    area = 0x8120000 + K["KMX_LIM_BYTES"]
    assert (ring.consumed, ring.events) == (area, area + d["KMD_WORK_EVENTS"])
    assert end == (8 | 9 << 16, area)
    block = kt.args_block(a_thread(), base=0, check_value=1, ring=ring)
    assert len(block) == K["KMX_LIM_BYTES"] + kt.ring_area_bytes(ring)
    flags, tile = struct.unpack_from("<2I", block, K["KMX_LIM_RING"] + 40)
    assert (flags, tile) == (K["KMX_RING_REMOTE"], 8 | 3 << 16)


def test_each_work_cluster_has_its_own_ring_after_the_ones_before():
    d = kt.abi.headers()
    ring3, end3 = kt.work_ring(0x8100000, (8, 3), 0x8120000, (8, 7), cluster=3)
    assert ring3.cmd == 0x8100000 + d["KMD_LIM_BYTES"] + 2 * kt.ring_bytes()
    args = kt.kernel_work_args({3: end3, 1: (8 | 9 << 16, 0x500)})
    assert args["work_cmds"] == d["KMD_LIM_BYTES"]
    assert args["work_noc"] == [8 | 9 << 16, 0, 8 | 7 << 16]
    assert args["work_noc_addr"] == [0x500, 0, end3[1]]
    assert kt.kernel_work_args({}) == {}


def test_the_argument_block_tells_the_thread_its_cluster():
    block = kt.args_block(a_thread(cluster=3), base=0, check_value=1)
    assert struct.unpack_from("<I", block, 56)[0] == 3


def test_threads_of_one_image_are_peers_led_by_the_lowest_cluster():
    groups = kt.peer_groups({3: "a.bin", 1: "a.bin", 2: "b.bin"})
    assert groups == {1: [1, 3], 3: [1, 3], 2: [2]}
    peers = kt.Peers(0x1234, 1, {1: 8 | 9 << 16, 3: 8 | 7 << 16}, {1: 6, 3: 7})
    ring, _ = kt.work_ring(0x8100000, (8, 3), 0x8120000, (8, 7), cluster=3)
    block = kt.args_block(a_thread(cluster=3), 0, 1, ring=ring, peers=peers)
    line = struct.unpack_from("<12I", block, K["KMX_LIM_PEERS"])
    assert line[:4] == (0x1234, 1, 0b1010, len(block))  # par_all lines after the block
    assert line[4:] == (0, 8 | 9 << 16, 0, 8 | 7 << 16, 0, 6, 0, 7)
    assert len(block) % 64 == 0


def test_argument_block_holds_the_region_table_and_words():
    block = kt.args_block(a_thread(deadline=2.0), base=0x100000, check_value=0xABC)
    assert len(block) == K["KMX_LIM_BYTES"]
    head = struct.unpack_from("<QIIQIIQQQ", block)
    assert head == (
        0,
        K["KMX_MAGIC"],
        K["KMX_VERSION"],
        100_000_000,
        2,
        2,
        0x100000,
        0xABC,
        0,
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
    def double(regions, words, run):
        regions["out"][:10] = regions["counts"] * words[0]
        return 3, [int(regions["out"].sum())]

    run = kt.ThreadRun(a_thread(host_fallback=double), None, None, on_card=False)
    run.start()
    r = run.wait(1.0)
    assert (r.code, r.words, r.state) == (3, [315], "done")
    out = run.read("out").view(np.uint32)
    assert len(out) == 1250 and list(out[:4]) == [0, 7, 14, 21] and out[10:].sum() == 0


def test_host_fallback_submits_commands_through_the_run_and_gets_result_words():
    kernel = FakeKernel()
    run = Run(FakeBackend(kernel), (12, 10))

    def leader_says(regions, words, run):
        handle = run.submit(1, (0, 8), result=True)
        kernel.results[handle] = 0xBEEF
        return 0, [run.wait(handle).result]

    fallback = a_thread(host_fallback=leader_says)
    thread = kt.ThreadRun(fallback, None, None, on_card=False, run=run)
    assert thread.wait(1.0).words == [0xBEEF]
    assert kernel.ring[1][1] >> 8 == kt.abi.headers()["KM_F_RESULT"]


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


def test_prefetch_keep_is_a_flag_in_the_argument_block():
    block = kt.args_block(a_thread(prefetch="keep"), base=0, check_value=1)
    assert struct.unpack_from("<Q", block, 48)[0] == K["KMX_FLAG_KEEP_PREFETCH"]


def test_a_thread_stop_stays_inside_the_time_it_is_given():
    port = FakePort()
    given = []
    port.work_wait = lambda c, timeout: given.append(("wait", timeout)) or {}
    port.work_stop = lambda c, timeout: given.append(("stop", timeout)) or {}
    run = kt.ThreadRun(a_thread(), port, None, image="t.bin")
    run.start()
    run.stop(0.6)
    (_, wait), (_, stop) = given
    assert wait == pytest.approx(0.2) and 0.5 < stop <= 0.6 and not run.running
    given.clear()
    run.running = True
    run.stop(30.0)  # the thread's own wait stays at most 1 s
    assert given[0] == ("wait", 1.0) and 29.0 < given[1][1] <= 30.0


def test_one_call_puts_the_same_thread_on_several_clusters():
    prog = Program("p", kernels="k")
    made = prog.thread(clusters=(1, 2, 3), sources=["t.c"], regions={"r": 64})
    assert [t.cluster for t in made] == [1, 2, 3] and [t.bank for t in made] == [
        6,
        7,
        7,
    ]
    with pytest.raises(ValueError, match="already has a thread on cluster 2"):
        prog.thread(cluster=2, sources=["t.c"], regions={"r": 64})


def test_a_shared_region_lives_in_one_bank():
    prog = Program("p", kernels="k")
    prog.region("pool", 4096)
    prog.thread(clusters=(2, 3), sources=["t.c"], regions={"r": 64}, shared=["pool"])
    prog.check_threads()
    prog.thread(cluster=1, sources=["t.c"], regions={"r": 64}, shared=["pool"])
    with pytest.raises(ValueError, match="banks \\[6, 7\\]"):
        prog.check_threads()


class TwoClusters(FakePort):
    """Cluster 2's thread fails at once; cluster 3's would run on."""

    def work_wait(self, c, timeout):
        self.calls.append(("wait", c))
        return {"ready": c == 2}

    def work_read(self, c, offset, n):
        state = K["KMX_STATE_FAILED"] if c == 2 else K["KMX_STATE_RUNNING"]
        return struct.pack("<64I", state, K["KMX_ERR_LATE"] if c == 2 else 0, *[0] * 62)


def test_one_threads_failure_stops_the_others_and_names_it():
    port, run = TwoClusters(), Run(FakeBackend(FakeKernel()), (12, 10))
    for c in (2, 3):
        run.threads[c] = kt.ThreadRun(a_thread(cluster=c), port, None, image="t.bin")
        run.threads[c].start()
    with pytest.raises(ThreadFailed) as err:
        run.thread_results(timeout=5.0)
    assert err.value.cluster == 2 and err.value.results[2].error == "late"
    assert not err.value.results[3].ran and ("stop", 3) in port.calls
    assert not any(t.running for t in run.threads.values())


class Stuck(FakePort):
    """A core that takes no stop flag: the runtime cannot stop the cluster."""

    def work_stop(self, c, timeout):
        self.calls.append(("stop", c))
        return {"cleaned": False, "failed": "X280 cores still running"}


def test_a_thread_that_does_not_read_the_stop_flag_fails_and_says_why():
    port, run = Stuck(), Run(FakeBackend(FakeKernel()), (12, 10))
    run.thread = kt.ThreadRun(a_thread(), port, None, image="t.bin")
    run.thread.start()
    with pytest.raises(ThreadFailed, match="stuck outside the library") as err:
        run.thread_result(timeout=0.1)
    assert "card needs a reset" in str(err.value) and "still running" in str(err.value)
    assert err.value.results[1].stuck and not run.thread.running


def test_a_thread_that_returns_after_the_stop_word_is_not_stuck():
    port = FakePort()
    answers = iter([{"ready": False}, {"ready": True}])
    port.work_wait = lambda c, timeout: next(answers)
    run = kt.ThreadRun(a_thread(), port, None, image="t.bin")
    run.start()
    r = run.wait(0.1)
    assert not r.ran and not r.stuck and r.detail == ""
