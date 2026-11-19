"""A program's NoC per data-movement kernel: checked, passed to the kernel's build."""

import pytest

from komorebi.errors import ProgramError
from komorebi.program.description import Program
from komorebi.session import kernels as session_kernels


def kernels(tmp_path):
    for kernel in ("reader", "compute", "writer"):
        (tmp_path / f"{kernel}.cpp").write_text("")
    return tmp_path


def test_each_kernel_takes_its_own_noc(tmp_path):
    for noc in ({"reader": 2, "writer": 0}, {"compute": 1}, {"reader": 1}):
        with pytest.raises(ProgramError):
            Program("p", kernels=kernels(tmp_path), noc=noc)


def test_a_swap_reaches_the_build_and_the_default_leaves_it(tmp_path):
    swapped = Program("p", kernels=kernels(tmp_path), noc={"reader": 1, "writer": 0})
    assert session_kernels.kernel_build(swapped, "reader", [], "blackhole") == {
        "defines": [],
        "noc": 1,
    }
    assert session_kernels.kernel_build(swapped, "compute", [], "blackhole") == {
        "defines": [("KM_DST_WINDOW", "1")]
    }
    assert session_kernels.kernel_build(swapped, "compute", [], "wormhole_b0") == {
        "defines": [("KM_DST_WINDOW", "0")]
    }
    plain = Program("q", kernels=tmp_path)
    assert session_kernels.kernel_build(plain, "writer", [], "blackhole") == {
        "defines": []
    }
