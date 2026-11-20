"""A switching run builds a program's own overlays, on every group, with the recipes of
the group that holds the program (its buffers' format tables and Dst mode), as its part
is built; on its own group, and for a program no group holds, the group's (no card)."""

from types import SimpleNamespace

from komorebi.program import overlay
from komorebi.program import several as programs
from komorebi.program.addon import program as added_program


def own(program):
    return [
        SimpleNamespace(name=f"{program.name} {number}", number=number)
        for number in program.numbers
    ]


def placed(*held):
    made = [
        SimpleNamespace(name="index", numbers=[1]),
        SimpleNamespace(name="compress", numbers=[2, 3]),
        SimpleNamespace(name="attend", numbers=[]),
    ]
    return SimpleNamespace(
        programs=made, groups=[SimpleNamespace(programs=each) for each in held]
    )


def recipes(group):
    return {
        target: f"{part} of group {group}" for target, part in overlay.TARGETS.items()
    }


def built_with(monkeypatch, each, recipe="the group's", part="unpack"):
    """The recipe a part of `each` is built with, when the group's is `recipe`."""
    if getattr(each, "build_part", None) is None:
        return recipe
    seen = []

    def cached(recipe, made, kernel, part, cache, size):
        seen.append(recipe)
        return "linked", None, False

    monkeypatch.setattr(overlay, "build_part_cached", cached)
    assert each.build_part(recipe, "compute", part, "cache", None) == "linked"
    return seen[0]


def test_a_switched_program_s_overlays_are_built_with_its_own_group_s_recipes(
    monkeypatch,
):
    monkeypatch.setattr(programs, "own_overlays", own)
    run = placed((0,), (1,), (2,), ())
    by_group = [recipes(group) for group in range(4)]
    for number in range(4):
        made = added_program.home_overlays(run, number, by_group)
        assert [each.name for each in made] == ["index 1", "compress 2", "compress 3"]
        assert [each.number for each in made] == [1, 2, 3]
        index, compress = made[0], made[1]
        want = "the group's" if number == 0 else "unpack of group 0"
        assert built_with(monkeypatch, index) == want
        want = "the group's" if number == 1 else "unpack of group 1"
        assert built_with(monkeypatch, compress) == want
        assert built_with(monkeypatch, made[2], part="pack") in (
            "the group's",
            "pack of group 1",
        )


def test_a_program_no_group_holds_keeps_each_group_s_recipes(monkeypatch):
    monkeypatch.setattr(programs, "own_overlays", own)
    run = placed((0,), (2,))  # compress held by no group here (another island's)
    by_group = [recipes(group) for group in range(2)]
    made = added_program.home_overlays(run, 1, by_group)
    assert built_with(monkeypatch, made[0]) == "unpack of group 0"
    assert [built_with(monkeypatch, each) for each in made[1:]] == ["the group's"] * 2
