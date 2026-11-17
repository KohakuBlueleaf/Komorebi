"""tanh, erf and log in a tile function: each is lowered with its bound printed, and
an error tighter than the bound is refused."""

import pytest

from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam


def stream(fn):
    x, y = TensorParam("x", False, F32), TensorParam("y", True, F32)
    ir = KernelIR("f", [x, y], [], {})
    with trace.Tracer(ir):
        xs, ys = trace.TensorProxy(x), trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(xs)):
            trace.store(ys.tiles[t], fn(trace.load(xs.tiles[t])))
    return unit_program(ir)


@pytest.mark.parametrize("name", ["tanh", "erf", "log"])
def test_function_lowered_with_its_bound(name):
    unit = stream(getattr(trace, name))
    done = {p.name: p.done for p in unit.passes}
    assert done["templates"][0].startswith(f"{name} (line")
    assert "accurate, relative error" in done["templates"][0]
    assert emit(unit).count(f"{unit.name}_init")


def test_error_tighter_than_the_bound_is_refused():
    with pytest.raises(PlanError, match="no template meets"):
        stream(lambda x: trace.log(x, error=1e-9))
