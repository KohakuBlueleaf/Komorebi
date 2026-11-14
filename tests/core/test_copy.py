"""A copy between DRAM pages: its bounds and page placement in C on the CPU, and what
the host hands a thread for kmx_fetch and kmx_store (no card)."""

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from komorebi import abi
from komorebi import channel as channel_module
from komorebi import thread as thread_module

CONSTANTS = thread_module.layout()


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_bounds_and_page_placement_in_c(tmp_path):
    source = Path(__file__).resolve().parent / "km_copy_check.c"
    include = thread_module.native.kmx_directory().parents[1] / "include"
    exe = tmp_path / "copy_check"
    subprocess.run(
        ["cc", "-std=c11", "-Wall", "-Werror", f"-I{include}", str(source), "-o", exe],
        check=True,
    )
    assert subprocess.run([exe], capture_output=True, text=True).stdout == "ok\n"


def test_the_thread_gets_the_cores_copy_buffer_in_its_ring_line():
    thread = thread_module.Thread(sources=["t.c"], regions={"in": 4096}, cluster=2)
    ring = thread_module.RingPlacement(0x1000, 0x2040, 0x3000, 0x4000, 16, 64)
    block = thread_module.args_block(
        thread, base=0, check_value=1, ring=ring, buffer=0x40010
    )
    assert struct.unpack_from("<I", block, CONSTANTS["KMX_LIM_RING"] + 48) == (0x40010,)
    headers = abi.headers()
    assert headers["KM_TRANSFER_CHUNK_BYTES"] == channel_module.CHUNK_DEFAULT
    assert headers["KM_HANDLER_COPY"] > headers["KM_HANDLER_USER_MAX"]
