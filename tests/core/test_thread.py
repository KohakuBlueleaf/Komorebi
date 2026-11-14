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

from komorebi import thread as thread_module
from komorebi.errors import ThreadFailed
from komorebi.program import Program
from komorebi.run import Run

KMX_DIRECTORY = thread_module.native.kmx_directory()
CONSTANTS = thread_module.layout()


def a_thread(**options):
    regions = {"counts": np.arange(10, dtype=np.uint32), "out": 5000}
    return thread_module.Thread(
        sources=["t.c"], regions=regions, words=[7, 9], **options
    )


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_ring_calls_against_a_fake_kernel(tmp_path):
    source = Path(__file__).resolve().parent / "kmx_ring_check.c"
    includes = [f"-I{KMX_DIRECTORY}", f"-I{KMX_DIRECTORY.parents[1] / 'include'}"]
    exe = tmp_path / "ring_check"
    command_line = [
        "cc",
        "-std=c11",
        "-Wall",
        "-Werror",
        *includes,
        str(source),
        "-o",
        str(exe),
    ]
    subprocess.run(command_line, check=True)
    assert subprocess.run([exe], capture_output=True, text=True).stdout == "ok\n"


def test_the_threads_wake_is_the_kernels():
    headers = thread_module.abi.headers()
    assert CONSTANTS["KMX_EVENT_WAKE"] == headers["KMD_EVENT_WAKE"]
    assert CONSTANTS["KMX_REJECT_BUSY"] == headers["KMD_REJECT_BUSY"]


def test_each_programs_record_reaches_the_ring_line_for_kmx_load():
    switch = {"records": {0: 0x1000, 1: 0x1240, 2: 0x1480}, "refused": None}
    records = thread_module.load_records(switch)
    assert records == (0x1000, 0x240, 3)
    ring = thread_module.RingPlacement(0x1000, 0x2040, 0x3000, 0x4000, 16, 64)
    block = thread_module.args_block(
        a_thread(), base=0, check_value=1, ring=ring, records=records
    )
    assert struct.unpack_from("<3I", block, CONSTANTS["KMX_LIM_RING"] + 52) == records
    assert thread_module.load_records(None) is None
    assert thread_module.load_records({**switch, "refused": "kernels differ"}) is None
    with pytest.raises(ValueError, match="not one step apart"):
        thread_module.load_records(
            {"records": {0: 0x1000, 1: 0x1240, 2: 0x1500}, "refused": None}
        )


def test_the_lines_the_kernel_hands_a_thread_follow_its_events_on_both_sides():
    headers = thread_module.abi.headers()
    ring, _ = thread_module.work_ring(0x8100000, (8, 3), 0x8120000, (8, 9))
    assert CONSTANTS["KMX_RING_IN_DEPTH"] == headers["KMD_WORK_IN_DEPTH"]
    assert CONSTANTS["KMX_HANDLER_ANSWER"] == headers["KMD_HANDLER_ANSWER"]
    lines = headers["KMD_WORK_EVENTS"] + 16 * ring.event_depth
    assert (
        thread_module.ring_area_bytes(ring)
        == lines + 64 * CONSTANTS["KMX_RING_IN_DEPTH"]
    )


def test_ring_line_names_the_placement():
    ring = thread_module.RingPlacement(
        0x1000, 0x2040, 0x3000, 0x4000, 16, 64, cached=True
    )
    block = thread_module.args_block(a_thread(), base=0, check_value=1, ring=ring)
    line = struct.unpack_from("<4Q3I", block, CONSTANTS["KMX_LIM_RING"])
    assert line == (
        0x1000,
        0x2040,
        0x3000,
        0x4000,
        16,
        64,
        CONSTANTS["KMX_RING_CACHED"],
    )


def test_a_work_ring_has_commands_in_the_kernels_lim_and_events_in_the_threads():
    ring, end = thread_module.work_ring(0x8100000, (8, 3), 0x8120000, (8, 9))
    abi_headers = thread_module.abi.headers()
    assert (
        ring.cmd == 0x8100000 + thread_module.rings_at() and ring.remote == 8 | 3 << 16
    )
    assert ring.read == ring.cmd + 64 * thread_module.RING_DEPTH
    area = 0x8120000 + CONSTANTS["KMX_LIM_BYTES"]
    assert (ring.consumed, ring.events) == (area, area + abi_headers["KMD_WORK_EVENTS"])
    assert end == (8 | 9 << 16, area)
    block = thread_module.args_block(a_thread(), base=0, check_value=1, ring=ring)
    assert len(block) == CONSTANTS["KMX_LIM_BYTES"] + thread_module.ring_area_bytes(
        ring
    )
    flags, tile = struct.unpack_from("<2I", block, CONSTANTS["KMX_LIM_RING"] + 40)
    assert (flags, tile) == (CONSTANTS["KMX_RING_REMOTE"], 8 | 3 << 16)


def test_each_work_cluster_has_its_own_ring_after_the_ones_before():
    ring3, end3 = thread_module.work_ring(
        0x8100000, (8, 3), 0x8120000, (8, 7), cluster=3
    )
    assert (
        ring3.cmd
        == 0x8100000 + thread_module.rings_at() + 2 * thread_module.ring_bytes()
    )
    args = thread_module.kernel_work_args({3: end3, 1: (8 | 9 << 16, 0x500)})
    assert args["work_cmds"] == thread_module.rings_at()
    assert args["work_noc"] == [8 | 9 << 16, 0, 8 | 7 << 16]
    assert args["work_noc_addr"] == [0x500, 0, end3[1]]
    assert thread_module.kernel_work_args({}) == {}


def test_the_argument_block_tells_the_thread_its_cluster():
    block = thread_module.args_block(a_thread(cluster=3), base=0, check_value=1)
    assert struct.unpack_from("<I", block, 56)[0] == 3


def test_threads_of_one_image_are_peers_led_by_the_lowest_cluster():
    groups = thread_module.peer_groups({3: "a.bin", 1: "a.bin", 2: "b.bin"})
    assert groups == {1: [1, 3], 3: [1, 3], 2: [2]}
    peers = thread_module.Peers(
        0x1234, 1, {1: 8 | 9 << 16, 3: 8 | 7 << 16}, {1: 6, 3: 7}
    )
    ring, _ = thread_module.work_ring(0x8100000, (8, 3), 0x8120000, (8, 7), cluster=3)
    block = thread_module.args_block(a_thread(cluster=3), 0, 1, ring=ring, peers=peers)
    line = struct.unpack_from("<12I", block, CONSTANTS["KMX_LIM_PEERS"])
    assert line[:4] == (
        0x1234,
        1,
        0b1010,
        len(block),
    )  # parallel_all lines after the block
    assert line[4:] == (0, 8 | 9 << 16, 0, 8 | 7 << 16, 0, 6, 0, 7)
    assert len(block) % 64 == 0


def test_argument_block_holds_the_region_table_and_words():
    block = thread_module.args_block(
        a_thread(deadline=2.0), base=0x100000, check_value=0xABC
    )
    assert len(block) == CONSTANTS["KMX_LIM_BYTES"]
    head = struct.unpack_from("<QIIQIIQQQ", block)
    assert head == (
        0,
        CONSTANTS["KMX_MAGIC"],
        CONSTANTS["KMX_VERSION"],
        100_000_000,
        2,
        2,
        0x100000,
        0xABC,
        0,
    )
    table = struct.unpack_from("<4Q", block, CONSTANTS["KMX_ARGS_REGIONS"])
    assert table == (
        0x100000 + thread_module.PAGE,
        40,
        0x100000 + 2 * thread_module.PAGE,
        5000,
    )
    assert struct.unpack_from("<3I", block, CONSTANTS["KMX_ARGS_WORDS"]) == (7, 9, 0)
    assert block[CONSTANTS["KMX_LIM_STOP"] :] == bytes(
        CONSTANTS["KMX_LIM_BYTES"] - CONSTANTS["KMX_LIM_STOP"]
    )


def test_result_block_is_read_with_names():
    words = [
        CONSTANTS["KMX_STATE_DONE"],
        CONSTANTS["KMX_ERROR_NONE"],
        5,
        2,
        3,
        100,
        500,
    ] + [0] * 9
    raw = struct.pack("<64I", *words, 11, 12, *([0] * 46))
    result = thread_module.parse_result(raw)
    assert (result.state, result.error, result.code, result.words, result.where) == (
        "done",
        "none",
        5,
        [11, 12],
        3,
    )
    assert (result.where_microseconds, result.microseconds) == (2.0, 10.0)


def test_host_fallback_gets_the_initial_regions_and_its_writes_are_kept():
    def double(regions, words, run):
        regions["out"][:10] = regions["counts"] * words[0]
        return 3, [int(regions["out"].sum())]

    run = thread_module.ThreadRun(
        a_thread(host_fallback=double), None, None, on_card=False
    )
    run.start()
    result = run.wait(1.0)
    assert (result.code, result.words, result.state) == (3, [315], "done")
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
    thread = thread_module.ThreadRun(fallback, None, None, on_card=False, run=run)
    assert thread.wait(1.0).words == [0xBEEF]
    assert kernel.ring[1][1] >> 8 == thread_module.abi.headers()["KM_FLAG_RESULT"]


class FakePort:
    def __init__(self):
        self.calls = []

    def work_start(self, cluster, image, block):
        self.calls.append(("start", cluster))

    def work_wait(self, cluster, timeout):
        self.calls.append(("wait", cluster))
        return {"ready": False}

    def work_write(self, cluster, offset, data):
        self.calls.append(("write", offset, struct.unpack_from("<I", data)[0]))

    def work_read(self, cluster, offset, length):
        return struct.pack("<64I", CONSTANTS["KMX_STATE_RUNNING"], *([0] * 63))

    def work_stop(self, cluster, timeout):
        self.calls.append(("stop", cluster))
        return {"cleaned": True}


def test_a_thread_past_its_timeout_is_asked_to_stop_then_stopped():
    port = FakePort()
    run = thread_module.ThreadRun(a_thread(), port, None, image="t.bin")
    run.start()
    result = run.wait(0.1)
    stop_line = ("write", CONSTANTS["KMX_LIM_STOP"], CONSTANTS["KMX_STOP_WORD"])
    assert port.calls == [
        ("start", 1),
        ("wait", 1),
        stop_line,
        ("wait", 1),
        ("stop", 1),
    ]
    assert not result.ran and result.state == "running" and not run.running


def test_prefetch_keep_is_a_flag_in_the_argument_block():
    block = thread_module.args_block(a_thread(prefetch="keep"), base=0, check_value=1)
    assert struct.unpack_from("<Q", block, 48)[0] == CONSTANTS["KMX_FLAG_KEEP_PREFETCH"]


def test_a_thread_stop_stays_inside_the_time_it_is_given():
    port = FakePort()
    given = []
    port.work_wait = lambda cluster, timeout: given.append(("wait", timeout)) or {}
    port.work_stop = lambda cluster, timeout: given.append(("stop", timeout)) or {}
    run = thread_module.ThreadRun(a_thread(), port, None, image="t.bin")
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
    assert [thread.cluster for thread in made] == [1, 2, 3] and [
        thread.bank for thread in made
    ] == [
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

    def work_wait(self, cluster, timeout):
        self.calls.append(("wait", cluster))
        return {"ready": cluster == 2}

    def work_read(self, cluster, offset, length):
        state = (
            CONSTANTS["KMX_STATE_FAILED"]
            if cluster == 2
            else CONSTANTS["KMX_STATE_RUNNING"]
        )
        return struct.pack(
            "<64I", state, CONSTANTS["KMX_ERROR_LATE"] if cluster == 2 else 0, *[0] * 62
        )


def test_one_threads_failure_stops_the_others_and_names_it():
    port, run = TwoClusters(), Run(FakeBackend(FakeKernel()), (12, 10))
    for cluster in (2, 3):
        run.threads[cluster] = thread_module.ThreadRun(
            a_thread(cluster=cluster), port, None, image="t.bin"
        )
        run.threads[cluster].start()
    with pytest.raises(ThreadFailed) as raised:
        run.thread_results(timeout=5.0)
    assert raised.value.cluster == 2 and raised.value.results[2].error == "late"
    assert not raised.value.results[3].ran and ("stop", 3) in port.calls
    assert not any(thread_run.running for thread_run in run.threads.values())


class Stuck(FakePort):
    """A core that takes no stop flag: the runtime cannot stop the cluster."""

    def work_stop(self, cluster, timeout):
        self.calls.append(("stop", cluster))
        return {"cleaned": False, "failed": "X280 cores still running"}


def test_a_thread_that_does_not_read_the_stop_flag_fails_and_says_why():
    port, run = Stuck(), Run(FakeBackend(FakeKernel()), (12, 10))
    run.thread = thread_module.ThreadRun(a_thread(), port, None, image="t.bin")
    run.thread.start()
    with pytest.raises(ThreadFailed, match="stuck outside the library") as raised:
        run.thread_result(timeout=0.1)
    assert "card needs a reset" in str(raised.value) and "still running" in str(
        raised.value
    )
    assert raised.value.results[1].stuck and not run.thread.running


def test_a_thread_that_returns_after_the_stop_word_is_not_stuck():
    port = FakePort()
    answers = iter([{"ready": False}, {"ready": True}])
    port.work_wait = lambda cluster, timeout: next(answers)
    run = thread_module.ThreadRun(a_thread(), port, None, image="t.bin")
    run.start()
    result = run.wait(0.1)
    assert not result.ran and not result.stuck and result.detail == ""


def test_a_threads_handlers_come_from_a_dict_or_a_header_and_reach_its_c(tmp_path):
    header = tmp_path / "h.h"
    header.write_text(
        "constexpr uint32_t kHandlerSum = 5;\nconstexpr uint32_t kHandlerBigStep = 7;\n"
    )
    thread = a_thread(handlers=header)
    assert thread.handlers == {"sum": 5, "big_step": 7}
    assert thread.handler("big_step") == 7 and thread.handler(9) == 9
    assert thread.handler_defines() == {"kHandlerSum": "5u", "kHandlerBigStep": "7u"}
    assert a_thread().handler_defines() == {}
    with pytest.raises(KeyError):
        thread.handler("missing")
    with pytest.raises(ValueError, match="not in 1"):
        a_thread(handlers={"stop": 240})
