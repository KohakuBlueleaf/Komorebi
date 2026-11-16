"""Transfers through the host channel on a card (KMR_CARD=1): pages pushed and pulled
exact without a run and inside one, and a bound the kernel refuses before it writes."""

from pathlib import Path

import numpy as np
import pytest

import komorebi as km
from komorebi import ChannelError
from komorebi.device import channel as channel_module
from komorebi.session.session import read_tensors

pytestmark = pytest.mark.card
KERNELS = Path(__file__).resolve().parents[3] / "examples" / "01_mapcollect" / "kernels"
PAGES, PAGE = 64, 4096  # 256 KiB: above the size from which a launch pays


@pytest.fixture
def session(session):
    if session.channel is None:
        pytest.skip("no host channel on this machine")
    return session


def pages(seed):
    return np.random.default_rng(seed).integers(0, 256, (PAGES, PAGE), dtype=np.uint8)


def test_pages_go_both_ways_exact_without_a_run(session):
    data = pages(1)
    tensor = session.tensor(data, "uint8", "row")  # pulled by the transfer kernel
    (written,) = read_tensors(session.device, session._module, [tensor])
    assert np.array_equal(written, data)
    (pushed,) = session.read(tensor)
    assert np.array_equal(pushed, data)
    assert np.array_equal(session.read(tensor, first=5, pages=30)[0], data[5:35])
    tensor.free()


def test_the_kernel_refuses_a_transfer_past_the_region_before_it_writes(session):
    tensor = session.tensor(pages(2), "uint8", "row")
    region = session._region
    wrong = channel_module.Transfer(
        tensor.address, PAGE, 0, PAGES, region.words, region.words, region.bytes, 7,
        0, 4 * PAGE,
    )  # fmt: skip
    words = channel_module.words_view(session.channel, region, 1)
    words[:] = 0
    want = channel_module.sequence_words(7, PAGES, PAGE, 1)
    with pytest.raises(ChannelError, match=r"status \[1\]"):
        session._launch_transfer(wrong, 1, True, words, want, 0.2)
    assert not words.any()
    tensor.free()


def test_pages_go_both_ways_exact_inside_a_run(session):
    data, other = pages(3), pages(4)
    tensor = session.tensor(data, "uint8", "row")
    program = km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "mapcollect.h",
    )
    run = session.load(program)
    run.wait(run.submit(0, (0, 8)))
    assert np.array_equal(session.read(tensor)[0], data)
    session.write(tensor, other[:10], first=20)
    assert np.array_equal(session.read(tensor, first=18, pages=14)[0][2:12], other[:10])
    run.stop()
    assert run.result()["error"] == 0
    (back,) = read_tensors(session.device, session._module, [tensor])
    assert np.array_equal(back[20:30], other[:10])
    assert np.array_equal(back[:20], data[:20])
