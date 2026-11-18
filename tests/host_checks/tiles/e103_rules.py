"""Padding refusals (no session): the build's and a submit's (Handler.args) check of a
32 x 40 tensor (held as 32 x 64), and the marks each submit leaves."""

import e103_classes as c
from stand_in import Pages

from komorebi import formats
from komorebi.device.tensor import Tensor, unwritten
from komorebi.dsl import compiler, padding
from komorebi.dsl.program import place_handlers
from komorebi.errors import ProgramError


def tensor(shape, written=True):  # session.tensor's (zero-padded) or session.empty's
    t = Tensor(Pages(4096), formats.padded_shape(shape, "tile"), "float32", "tile")
    if written:
        t.logical = formats.logical_shape(shape, "tile")
    else:
        unwritten(t, formats.logical_shape(shape, "tile"))
    return t


def build(cls, tensors):
    compiled = compiler.compile_class(cls, {}, tensors)
    place_handlers(compiled, {m.name: [m] for m in compiler.handler_methods(cls)})
    for m in compiler.handler_methods(cls):
        padding.checked(m.name, m.padding, tensors, built=True)


def refused(label, fn, words):
    try:
        fn()
    except ProgramError as error:
        assert all(w in str(error) for w in words), str(error)
        print(f"{label}: refused: {str(error)[:110]}")
        return
    raise SystemExit(f"{label}: not refused")


def exp_two():
    x, y, out = tensor((32, 40)), tensor((32, 40), False), tensor((32, 40), False)
    build(c.Two, {"x": x, "y": y, "out": out})
    c.Two.first.args(x, y)
    assert y.padding == "dirty" and y.padding_by.startswith("first")
    c.Two.second.args(y, out)


refused("exp then sum, two handlers", exp_two, ["reduce_sum", "written by first"])
refused(
    "exp then sum, one handler",
    lambda: build(c.One, {"x": tensor((32, 40)), "out": tensor((32, 40), False)}),
    ["reduce_sum"],
)
refused(
    "row max",
    lambda: build(c.Max, {"x": tensor((32, 40)), "out": tensor((32, 40), False)}),
    ["reduce_max", "-inf"],
)

x, y, out = tensor((32, 40)), tensor((32, 40), False), tensor((32, 40), False)
build(c.Two, {"x": x, "y": y, "out": out})
c.Two.gelu_of.args(x, y)
c.Two.second.args(y, out)
assert y.padding == "clean"
print("GELU then sum: submitted, y clean")

x, w, h, w2, y = (
    tensor((32, 40)),
    tensor((40, 40)),
    tensor((32, 40), False),
    tensor((40, 40)),
    tensor((32, 40), False),
)
build(c.Linear, {"x": x, "w": w, "y": h})
c.Linear.layer.args(x, w, h)
c.Linear.layer.args(h, w2, y)
assert h.padding == y.padding == "clean"
print("linear after linear, K = 40: submitted, both clean")


def product_dirty():
    x, y = tensor((32, 40)), tensor((32, 40), False)
    build(c.Two, {"x": x, "y": y, "out": tensor((32, 40), False)})
    c.Two.first.args(x, y)
    build(c.Linear, {"x": y, "w": tensor((40, 40)), "y": tensor((32, 40), False)})
    c.Linear.layer.args(y, tensor((40, 40)), tensor((32, 40), False))


refused("a product of exp's output", product_dirty, ["a product", "written by first"])

m = Tensor(Pages(4096), (32, 64), "float32", "tile")
x, out = tensor((32, 40)), tensor((32, 40), False)
build(c.Masked, {"x": x, "m": m, "out": out})
c.Masked.run.args(x, m, out)
print("exp then sum times a mask row: submitted")

whole = [Tensor(Pages(4096), (32, 64), "float32", "tile") for _ in range(3)]
build(c.Two, dict(zip(("x", "y", "out"), whole)))
c.Two.first.args(*whole[:2])
c.Two.second.args(*whole[1:])
assert whole[0].logical is None
print("whole tiles: exp then sum submitted (skipped)")
print("OK")
