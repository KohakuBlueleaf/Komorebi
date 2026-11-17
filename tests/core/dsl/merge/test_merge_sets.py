"""km.merge of classes with a handler's listed constant sets: each listed set (or the
defaults) a handler of the merged program at the numbers the class's own build gives
it, a call's set picked by its constants, an unlisted set refused, km.ANY refused with
its reason, two classes' constant handlers of one name refused (no card)."""

import pytest
from test_class_first_use import FakeRun
from test_class_handler_constants import TENSORS, Gate, quiet

import komorebi as km
from komorebi.errors import ProgramError


@km.tile_function
def thrice(a):
    return a + a + a


class Plain(km.Program):
    @km.handler
    def plain(self, a: km.Tensor, out: km.Tensor, tiles: int):
        results = km.Stream(out.pages(0, tiles), write=True, block=4)
        for x in km.Stream(a.pages(0, tiles), block=4):
            results.push(thrice(x))


class Sized(km.Program):
    @km.handler
    def gate(
        self, a: km.Tensor, out: km.Tensor, tiles: int, size: km.constant[int] = 4
    ):
        results = km.Stream(out.pages(0, tiles), write=True, block=size)
        for x in km.Stream(a.pages(0, tiles), block=size):
            results.push(thrice(x))


def made(cls, **given):
    return cls(l1=1 << 20, log=quiet, **TENSORS, **given)


def merged(*programs):
    return km.merge(*programs, l1=1 << 20, log=quiet)


def test_each_listed_set_is_a_handler_of_the_merged_program():
    gate = made(Gate, handlers={"gate": [{"size": 4}, {"size": 8}]})
    assert gate.handlers == {"gate_size4": 1, "gate_size8": 3, "other": 2}
    program = merged(made(Plain), gate)
    assert program.handlers == {
        "plain": 1,
        "gate_size4": 2,
        "other": 3,
        "gate_size8": 4,
    }
    first = merged(gate, made(Plain))  # the first class keeps its own numbers
    assert first.handlers == {"gate_size4": 1, "other": 2, "gate_size8": 3, "plain": 4}
    run = FakeRun([program])
    a, out = TENSORS["a"], TENSORS["out"]
    four, eight = (run._built_for(Gate.gate, (a, out, 3, n)) for n in (4, 8))
    assert (int(four), int(eight)) == (2, 4)
    assert four.method.name == "gate_size4" and four.args(a, out, 3) == [4096, 4096, 3]
    assert run._handler(Gate.other) == (3, 0) and run._handler(Plain.plain) == (1, 0)
    why = r"gate with size=2: no handler was built for it \(built: size=4; size=8\)"
    with pytest.raises(ProgramError, match=why):
        run._built_for(Gate.gate, (a, out, 3, 2))


def test_a_handler_s_defaults_when_no_set_is_listed():
    program = merged(made(Plain), made(Gate))
    assert program.handlers == {"plain": 1, "gate": 2, "other": 3}
    assert [(v, m.number) for v, m in program.constant_handlers["gate"]] == [
        ({"size": 8}, 2)
    ]


def test_the_refusals():
    with pytest.raises(ProgramError, match=r"gate: made with km.ANY, whose sets are"):
        merged(made(Plain), made(Gate, handlers={"gate": [{"size": 4}, km.ANY]}))
    sized = Sized(l1=1 << 20, log=quiet, **TENSORS)
    why = r"gate has constant parameters in gate and sized; a set is picked by"
    with pytest.raises(ProgramError, match=why):
        merged(made(Gate), sized)
