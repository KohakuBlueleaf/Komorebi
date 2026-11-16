"""A load of several programs counts its slots before its kernels are built: a
switching run one a program (and one more for a class named ahead with an own
overlay), else each group's overlays; past 4 its kernels get KM_OVERLAY_SLOTS (no card).
"""

from types import SimpleNamespace

from komorebi import overlay, programs
from komorebi.session import placed_slots, slots_most

ALL = {"reader": "r.cpp", "writer": "w.cpp", "compute": "c.cpp"}


def program(count=None, ahead=()):
    made = SimpleNamespace(overlay_slots=overlay.Slots(count), overlays={})
    made.added_ahead_programs = list(ahead)
    return made


def plan(made, groups, switch):
    return SimpleNamespace(
        programs=made,
        switch=switch,
        groups=[SimpleNamespace(programs=group) for group in groups],
    )


def test_a_switching_run_takes_one_a_program_and_one_for_an_own_overlay_ahead():
    five = [program() for _ in range(5)]
    assert placed_slots(plan(five, [range(5)], True)) == dict.fromkeys(ALL, 5)
    assert slots_most(None, plan(five, [range(5)], True)) == 5
    three = [program() for _ in range(3)]
    assert slots_most(None, plan(three, [range(3)], True)) == 4  # no define
    ahead = SimpleNamespace(overlays={"o": SimpleNamespace(sources={"reader": "o"})})
    four = [program(ahead=[ahead])] + [program() for _ in range(3)]
    counts = placed_slots(plan(four, [range(4)], True))
    assert counts == {"reader": 5, "writer": 4, "compute": 4}
    declared = [program(count={"reader": 2})] + [program() for _ in range(5)]
    assert placed_slots(plan(declared, [range(6)], True))["reader"] == 2


def test_groups_count_their_own_overlays(monkeypatch):
    def parts(each, switch=False, alone=True):
        return [SimpleNamespace(sources=ALL)]

    def own(each):
        return [SimpleNamespace(sources={"compute": "x"})] * each.own

    monkeypatch.setattr(programs, "program_overlays", parts)
    monkeypatch.setattr(programs, "own_overlays", own)
    made = [program(), program()]
    made[0].own, made[1].own = 5, 1
    counts = placed_slots(plan(made, [[0], [1]], False))
    assert counts == {"reader": 1, "writer": 1, "compute": 6}
    assert overlay.slots_define(slots_most(None, plan(made, [[0], [1]], False))) == [
        ("KM_OVERLAY_SLOTS", "6")
    ]
