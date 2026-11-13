"""X280 ports on the host: the default deal of cores over X280 cores, the refusals, the
ports line and area the threads read, and the control-block lines (no card)."""

import struct

import pytest

from komorebi import abi
from komorebi import thread as kt
from komorebi.program import Program

K = kt.layout()
H = abi.headers()


def test_by_default_cores_are_dealt_in_grid_order_cluster_by_cluster():
    plan = kt.port_plan(None, 120, {3, 1, 2})
    assert plan[0] == (1, 0) and plan[9] == (1, 0) and plan[10] == (1, 1)
    assert plan[119] == (3, 3) and len(plan) == 120
    per_cpu = {}
    for who in plan.values():
        per_cpu[who] = per_cpu.get(who, 0) + 1
    assert sorted(set(per_cpu.values())) == [10]
    one = kt.port_plan(None, 120, {2})
    assert set(one.values()) == {(2, h) for h in range(4)} and one[119] == (2, 3)


@pytest.mark.parametrize(
    ("serve", "match"),
    [
        ({120: (1, 0)}, "the grid has cores 0 to 119"),
        ({0: (2, 0)}, "no thread on cluster 2"),
        ({0: (1, 4)}, "X280 core 4 is not 0 to 3"),
    ],
)
def test_a_plan_out_of_the_grid_or_the_threads_is_refused(serve, match):
    with pytest.raises(ValueError, match=match):
        kt.port_plan(serve, 120, {1})
    with pytest.raises(ValueError, match="need an X280 thread"):
        kt.port_plan(None, 120, set())


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
    thread = kt.Thread(sources=["t.c"], regions={"r": 64})
    ports = kt.Ports(0x1000, 3, [0x10001, 0x20002, 0x30003], {0: (1, 2), 2: (3, 0)}, 1)
    block = kt.args_block(thread, base=0, check_value=1, ports=ports)
    line = struct.unpack_from("<5I", block, K["KMX_LIM_PORTS"])
    at = K["KMX_LIM_BYTES"] + K["KMX_ALL_BYTES"]
    assert line == (at, 0x1000, 3, K["KMX_PORTS_NOC1"], 0) and kt.ports_at(None) == at
    area = kt.ports_area(ports)
    assert len(area) == K["KMX_PORTS_BYTES"] and area[K["KMX_PORTS_BYTES"] - 1] == 0
    assert struct.unpack_from("<3I", area, K["KMX_PORTS_TILES"]) == tuple(ports.tiles)
    owners = area[K["KMX_PORTS_OWNER"] : K["KMX_PORTS_OWNER"] + 3]
    assert list(owners) == [1 << 2 | 2, 0, 3 << 2 | 0]
    box = kt.port_box(0x8000000, None, 2)
    assert box == 0x8000000 + at + K["KMX_PORTS_BOX"] + 128


def test_the_port_lines_end_the_control_block_and_fit_the_library():
    assert H["KM_CTL_X280"] + H["KM_X280_BYTES"] == H["KM_CTL_BYTES"]
    assert H["KM_X280_TO"] + 16 <= H["KM_X280_BYTES"]
    assert K["KMX_PORT_WINDOW"] + K["KMX_PORTS_CORES"] <= 192
    assert K["KMX_PORTS_CORES"] == H["KM_CORES_MAX"]
