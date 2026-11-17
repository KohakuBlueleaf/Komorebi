"""Added code built ahead: Program.ahead keeps run.addon's overlay, or a class's
handler at a set its first use compiles (the same name and parts the first use adds),
and the load builds each part for each group into the parts' cache, whose key holds
no overlay number (no card)."""

import dataclasses

import pytest
from test_class_first_use import FakeRun, fresh, made
from test_class_handler_constants import TENSORS, Gate

import komorebi as km
from komorebi.errors import ProgramError
from komorebi.program import overlay
from komorebi.program.addon import code as added


def test_a_set_built_ahead_is_the_one_its_first_use_adds():
    program = made(handlers={"gate": km.ANY})
    fresh(program, 4)
    program.ahead(Gate.gate, size=4)
    (kept,) = program.added_ahead
    run = FakeRun([program])
    run._built_for(Gate.gate, (TENSORS["a"], TENSORS["out"], 3, 4))
    ((name, parts, defines),) = run.adds
    sources = {k: str(v) for k, v in kept.sources.items()}
    assert (kept.name, sources) == (name, {k: str(v) for k, v in parts.items()})
    assert kept.defines == dict(defines or {})


def test_a_hand_overlay_and_the_refusals(tmp_path):
    source = tmp_path / "extra.cpp"
    source.write_text("// a part\n")
    program = made()
    program.ahead("extra", {"reader": source}, defines={"X": 1})
    assert [(m.name, m.defines) for m in program.added_ahead] == [("extra", {"X": 1})]
    with pytest.raises(ProgramError, match=r"built ahead only with km.ANY"):
        made(handlers={"gate": [{"size": 4}]}).ahead(Gate.gate, size=4)
    with pytest.raises(ProgramError, match=r"width is not a constant parameter"):
        made(handlers={"gate": km.ANY}).ahead(Gate.gate, width=4)
    with pytest.raises(ProgramError, match=r"name the handler"):
        program.ahead("extra", {"reader": source}, size=4)


@dataclasses.dataclass(frozen=True)
class Recipe:
    target: str
    elf: object

    def provided(self):
        return ""


def test_the_load_builds_each_part_for_each_group(tmp_path, monkeypatch):
    source = tmp_path / "extra.cpp"
    source.write_text("// a part\n")
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"\x7fELF")
    program = made().ahead("extra", {"reader": source, "writer": source})
    (kept,) = program.added_ahead
    recipe = Recipe("brisc", elf)
    assert overlay.built_key(recipe, kept, "reader") == overlay.built_key(
        recipe, dataclasses.replace(kept, number=7), "reader"
    )  # the add's number is not in the key
    built = []

    def build(recipe, made, kernel, part, cache, slot_bytes):
        built.append((recipe, made.name, part, slot_bytes))
        return type("Linked", (), {"image": b""})(), elf, len(built) > 2

    monkeypatch.setattr(overlay, "build_part_cached", build)
    groups = [
        added.Group(first, 60, {"reader": first, "writer": -first}, slots, set())
        for first, slots in (
            (0, {"reader": (2, 4096), "writer": (2, 4096)}),
            (60, {"reader": (1, 2048), "writer": (1, 2048)}),
        )
    ]
    room = type("Room", (), {"groups": groups})()
    lines = []
    session = type("Session", (), {"log": lines.append, "tree": None})()
    monkeypatch.setattr(added.insn_check, "enabled", lambda: False)
    added.build_ahead(session, room, [program])
    assert built == [
        (0, "extra", "reader", 4096), (0, "extra", "writer", 4096),
        (60, "extra", "reader", 2048), (-60, "extra", "writer", 2048),
    ]  # fmt: skip
    assert lines[0].startswith("ahead extra: 4 parts for 2 groups in")
    assert lines[0].endswith("(2 built before)")
    added.build_ahead(session, None, [program])  # no room reserved: none built
    assert len(built) == 4
