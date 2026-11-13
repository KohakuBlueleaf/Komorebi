"""X280 ports on the host: the default deal of cores over X280 cores, the rejections,
the ports line and area the threads read, and the control-block lines (no card)."""

import struct

import pytest

from komorebi import abi
from komorebi import thread as thread_module
from komorebi.program import Program

CONSTANTS = thread_module.layout()
HEADERS = abi.headers()


def test_by_default_cores_are_dealt_in_grid_order_cluster_by_cluster():
    plan = thread_module.port_plan(None, 120, {3, 1, 2})
    assert plan[0] == (1, 0) and plan[9] == (1, 0) and plan[10] == (1, 1)
    assert plan[119] == (3, 3) and len(plan) == 120
    per_cpu = {}
    for who in plan.values():
        per_cpu[who] = per_cpu.get(who, 0) + 1
    assert sorted(set(per_cpu.values())) == [10]
    one = thread_module.port_plan(None, 120, {2})
    assert set(one.values()) == {(2, x280_core) for x280_core in range(4)} and one[
        119
    ] == (2, 3)


@pytest.mark.parametrize(
    ("serve", "match"),
    [
        ({120: (1, 0)}, "the grid has cores 0 to 119"),
        ({0: (2, 0)}, "no thread on cluster 2"),
        ({0: (1, 4)}, "X280 core 4 is not 0 to 3"),
    ],
)
def test_a_plan_out_of_the_grid_or_the_threads_is_rejected(serve, match):
    with pytest.raises(ValueError, match=match):
        thread_module.port_plan(serve, 120, {1})
    with pytest.raises(ValueError, match="need an X280 thread"):
        thread_module.port_plan(None, 120, set())


def test_ports_are_declared_once_on_noc_0_or_1_with_some_core():
    prog = Program("p", kernels="k")
    with pytest.raises(ValueError, match="0 or 1"):
        prog.ports(noc=2)
    with pytest.raises(ValueError, match="names no core"):
        prog.ports(serve={})
    prog.ports(serve={"3": ("1", 2)})
    assert prog.x280_ports == {"serve": {3: (1, 2)}, "noc": 0}
    with pytest.raises(ValueError, match="already declares"):
        prog.ports()


def test_the_ports_line_and_area_name_the_tiles_and_x280_cores():
    thread = thread_module.Thread(sources=["t.c"], regions={"r": 64})
    ports = thread_module.Ports(
        0x1000, 3, [0x10001, 0x20002, 0x30003], {0: (1, 2), 2: (3, 0)}, 1
    )
    block = thread_module.args_block(thread, base=0, check_value=1, ports=ports)
    line = struct.unpack_from("<5I", block, CONSTANTS["KMX_LIM_PORTS"])
    area_offset = CONSTANTS["KMX_LIM_BYTES"] + CONSTANTS["KMX_ALL_BYTES"]
    assert (
        line == (area_offset, 0x1000, 3, CONSTANTS["KMX_PORTS_NOC1"], 0)
        and thread_module.ports_at(None) == area_offset
    )
    area = thread_module.ports_area(ports)
    assert (
        len(area) == CONSTANTS["KMX_PORTS_BYTES"]
        and area[CONSTANTS["KMX_PORTS_BYTES"] - 1] == 0
    )
    assert struct.unpack_from("<3I", area, CONSTANTS["KMX_PORTS_TILES"]) == tuple(
        ports.tiles
    )
    owners = area[CONSTANTS["KMX_PORTS_OWNER"] : CONSTANTS["KMX_PORTS_OWNER"] + 3]
    assert list(owners) == [1 << 2 | 2, 0, 3 << 2 | 0]
    box = thread_module.port_box(0x8000000, None, 2)
    assert box == 0x8000000 + area_offset + CONSTANTS["KMX_PORTS_BOX"] + 128


def test_the_port_lines_end_the_control_block_and_fit_the_library():
    assert (
        HEADERS["KM_CONTROL_X280"] + HEADERS["KM_X280_BYTES"]
        == HEADERS["KM_CONTROL_BYTES"]
    )
    assert HEADERS["KM_X280_TO"] + 16 <= HEADERS["KM_X280_BYTES"]
    assert CONSTANTS["KMX_PORT_WINDOW"] + CONSTANTS["KMX_PORTS_CORES"] <= 192
    assert CONSTANTS["KMX_PORTS_CORES"] == HEADERS["KM_CORES_MAX"]


def test_a_pull_asks_with_its_own_word():
    assert HEADERS["KM_PULL_ASK"] == 0x50554C4C  # "PULL"
