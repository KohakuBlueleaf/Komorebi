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


def test_an_island_s_entry_command_goes_to_its_ranks_in_flat_ranks(monkeypatch):
    """Island 1 of two cards (ranks 119..237, core 119 its link core): the entry goes
    to its rank cores as flat ranks, its part only; one island's is unchanged."""
    sent = []
    room = added.Room(None, 0, 1, [added.Group(0, 119, {}, {}, set())], set())
    log = []
    backend = SimpleNamespace(room=room, card=SimpleNamespace(log=log.append))
    monkeypatch.setattr(added, "_place", lambda run, made, builds: [{}])
    monkeypatch.setattr(added, "_mirror", lambda *call: None)
    made = SimpleNamespace(name="scale", number=5, message=False)
    for first, ranks, cores in ((119, 119, 120), (0, 120, 120)):
        run = SimpleNamespace(
            backend=backend, island_first=first, island_ranks=ranks, cores=cores,
            _flag_part=4, wait=lambda handle, timeout: SimpleNamespace(kind="done"),
            _queue_command=lambda *call: sent.append(call[1:2] + call[3:]),
        )  # fmt: skip
        room.added.clear()
        assert added.committed(run, (made, True, [{}], 0.0, True), 1.0) == 5
    assert sent == [((119, 119), False, 4), ((0, 119), False, 0)]
