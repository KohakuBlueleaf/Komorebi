"""A load states its writer's margin to the writer CPU's code memory (Wormhole's
NCRISC: 16 KiB, chips.Chip.writer_code) and warns under load.WRITER_MARGIN."""

import types
import warnings

import pytest

from komorebi.program.description import KernelSizes
from komorebi.session.parts import load


def logged(arch, ncrisc):
    """The lines _log_kernel_buffer gives a load whose writer's text is `ncrisc`."""
    lines = []
    session = types.SimpleNamespace(arch=arch, log=lines.append)
    text = {"BRISC": 9000, "NCRISC": ncrisc, "TRISC0": 5000}
    program = types.SimpleNamespace(name="p", sizes=KernelSizes(text, 30000, 70656))
    load.LoadMixin._log_kernel_buffer(session, program)
    return lines


def test_wormhole_states_the_writer_margin():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = logged("wormhole_b0", 12540)
    assert lines[-1] == "writer code: 12,540 of 16,384 B, margin 3,844 B"


def test_wormhole_warns_under_one_kib_of_margin():
    with pytest.warns(UserWarning, match=r"p: writer code: 16,140 of 16,384 B, margin"):
        lines = logged("wormhole_b0", 16140)
    assert lines[-1].endswith("margin 244 B")


def test_blackhole_has_no_writer_code_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = logged("blackhole", 30000)
    assert not any(line.startswith("writer code") for line in lines)
