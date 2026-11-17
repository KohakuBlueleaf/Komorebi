"""IslandRun.addon of hand-written code (no card): every island checked and built
first, island 0's number on every island, then each island's writes; a refusal on a
later island writes nothing anywhere."""

from types import SimpleNamespace

import pytest

from komorebi.islands.run import IslandRun
from komorebi.program.addon import code as added


def island_run(monkeypatch, taken=()):
    graph = SimpleNamespace(ranks=236, islands=[0, 1])
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None, backend=None)] * 2)
    run.runs = ("island 0", "island 1")
    run.session = SimpleNamespace()
    said = []

    def prepared(one, name, parts, handler, number, *rest):
        if number is None:
            number = 5
        if (one, number) in taken:
            raise added.AddError(f"{name}: number {number} is an overlay already")
        said.append(("prepared", one, number, handler))
        return SimpleNamespace(number=number), handler

    def committed(one, ready, timeout):
        said.append(("committed", one, ready[0].number))
        return ready[0].number

    monkeypatch.setattr(added, "prepared", prepared)
    monkeypatch.setattr(added, "committed", committed)
    return run, said


def test_code_is_added_on_every_island_with_island_0_s_number(monkeypatch):
    run, said = island_run(monkeypatch)
    assert run.addon("scale", {"compute": "scale.cpp"}, handler=True) == 5
    assert said == [
        ("prepared", "island 0", 5, True),
        ("prepared", "island 1", 5, True),
        ("committed", "island 0", 5),
        ("committed", "island 1", 5),
    ]


def test_a_refusal_on_a_later_island_writes_nothing(monkeypatch):
    run, said = island_run(monkeypatch, taken={("island 1", 5)})
    run._error = lambda error, island: type(error)(f"island {island}: {error}")
    with pytest.raises(added.AddError, match=r"island 1: scale: number 5"):
        run.addon("scale", {"compute": "scale.cpp"})
    assert [step for step, *_ in said] == ["prepared"]


def test_code_needs_its_parts_and_a_split_card_refuses_it(monkeypatch):
    run, _ = island_run(monkeypatch)
    with pytest.raises(TypeError, match=r"parts, as Program.overlay's"):
        run.addon("scale")
    run.session = None
    with pytest.raises(added.AddError, match=r"a card split into islands"):
        run.addon("scale", {"compute": "scale.cpp"})
