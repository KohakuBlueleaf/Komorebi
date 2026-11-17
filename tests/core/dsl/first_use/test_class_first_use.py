"""A handler's constant set compiled on its first use (km.ANY): the opt-in's forms,
the set compiled as overlay parts on the load's buffers and added as a handler, found
added and found built; the keys holding the constants; each refusal (no card)."""

import json

import pytest
from class_first_use_fixture import FakeRun, fresh, made
from class_handler_constants_fixture import TENSORS, Gate, quiet

import komorebi as km
from komorebi.dsl import first_use
from komorebi.dsl.first_use import constant_handler
from komorebi.errors import L1BudgetError, ProgramError
from komorebi.program import overlay


def test_the_opt_in_s_forms():
    alone = made(handlers={"gate": km.ANY})
    assert alone.first_use.opted == {"gate": km.ANY}
    assert alone.constant_handlers["gate"][0][0] == {"size": 8}  # its default built
    assert alone.handlers == {"gate": 1, "other": 2}
    listed = made(handlers={"gate": [{"size": 4}, km.ANY]})
    assert [values for values, _ in listed.constant_handlers["gate"]] == [{"size": 4}]
    assert repr(km.ANY(size=16)) == "km.ANY(size=16)" and repr(km.ANY) == "km.ANY"
    for program in (alone, listed):
        assert program.defines["KM_ADDED_HANDLERS"] == "1"
        assert program.overlay_slots.bytes == first_use.SLOT_BYTES
    assert not hasattr(made(handlers={"gate": [{"size": 4}]}), "first_use")


def test_a_reserve_sizes_the_load_s_buffers_for_its_set():
    plain = made(handlers={"gate": km.ANY})
    sized = made(handlers={"gate": km.ANY(size=16)})
    pages = {b.index_a: b.pages for b in plain.first_use.buffers.values()}
    larger = {b.index_a: b.pages for b in sized.first_use.buffers.values()}
    assert pages == {16: (16, 16), 0: (32, 32), 2: (4, 4)}  # size 8; other's
    assert larger == {16: (32, 32), 0: (64, 64), 2: (4, 4)}
    assert plain.built.make != sized.built.make  # the reserve is in the make's key
    one = first_use.folder_of(plain, Gate.gate, {"size": 4})
    assert one != first_use.folder_of(sized, Gate.gate, {"size": 4})  # and the set's


def test_a_set_not_built_is_compiled_added_and_found():
    program = made(handlers={"gate": [{"size": 8}, km.ANY(size=16)]})
    folder = fresh(program, 4)
    run = FakeRun([program])
    a, out = TENSORS["a"], TENSORS["out"]
    built = run._built_for(Gate.gate, (a, out, 3, 4))
    assert int(built) == 20 and built.method.name == "gate_size4"
    assert built.method.overlay and built.args(a, out, 3) == Gate.gate.args(a, out, 3)
    ((name, parts, defines),) = run.adds
    assert name == f"{program.first_use.key}.gate_size4"
    assert parts["reader"] == parts["writer"] == folder / "gate_size4/movement.cpp"
    assert parts["compute"] == folder / "gate_size4/compute.cpp"
    assert defines["KM_ADDED_HANDLERS"] == "1"
    text = parts["reader"].read_text()
    assert "uint32_t km_overlay(km::MoverRuntime& km_core" in text
    assert "kRead0Block = 4;" in (folder / "gate.h").read_text()
    assert (
        "constexpr km::BufferPair kRead0 = {0, 1};" in (folder / "gate.h").read_text()
    )
    kept = json.loads((folder / first_use.KEPT).read_text())
    assert kept["name"] == "gate_size4" and kept["words"][2] == ["tiles", "int"]
    assert program.first_use.made[("gate", (("size", 4),))][3] is False  # compiled
    # the next submit finds it added; the listed set is the loaded handler
    assert int(run._built_for(Gate.gate, (a, out, 3, 4))) == 20 and len(run.adds) == 1
    assert int(run._built_for(Gate.gate, (a, out, 3, 8))) == 1
    # a later program of the same make (a later process) finds it built
    later = made(handlers={"gate": [{"size": 8}, km.ANY(size=16)]})
    other = FakeRun([later])
    assert int(other._built_for(Gate.gate, (a, out, 3, 4))) == 20
    assert later.first_use.made[("gate", (("size", 4),))][3] is True  # found built
    assert other.adds[0][1] == parts


def test_a_folder_whose_files_changed_is_compiled_again():
    sets = {"gate": [{"size": 8}, km.ANY(size=16)]}
    program = made(handlers=sets)
    folder = fresh(program, 4)
    first_use.compiled(program, Gate.gate, {"size": 4})
    stamp = ("gate", (("size", 4),))
    assert program.first_use.made[stamp][3] is False  # compiled
    first_use.compiled(made(handlers=sets), Gate.gate, {"size": 4})
    (folder / "gate.h").write_text("// stale\n")
    later = made(handlers=sets)
    first_use.compiled(later, Gate.gate, {"size": 4})
    assert later.first_use.made[stamp][3] is False  # its files were not its text
    assert "stale" not in (folder / "gate.h").read_text()
    found = made(handlers=sets)
    first_use.compiled(found, Gate.gate, {"size": 4})
    assert found.first_use.made[stamp][3] is True


def test_the_folder_key_holds_the_constants():
    program = made(handlers={"gate": km.ANY(size=16)})
    four = first_use.folder_of(program, Gate.gate, {"size": 4})
    assert four == first_use.folder_of(program, Gate.gate, {"size": 4})
    assert four != first_use.folder_of(program, Gate.gate, {"size": 2})
    assert (
        four.name.startswith("gate-use") and four.parent == program.built.folder.parent
    )
    key = first_use.folder_key("gate-0", "gate", {"size": 4}, [b"class"])
    assert key != first_use.folder_key("gate-0", "gate", {"size": 4}, [b"klass"])
    assert key != first_use.folder_key("gate-1", "gate", {"size": 4}, [b"class"])
    assert key != first_use.folder_key("gate-0", "gate", {"size": 2}, [b"class"])


def test_a_set_the_load_s_buffers_do_not_hold_is_refused():
    program = made(handlers={"gate": km.ANY})  # buffers for size 8
    fresh(program, 16)
    why = (
        r"Gate.gate_size16: buffer kWrite0 \(index 16\): it would hold 32 \+ 32 "
        r"pages, the load's holds 16 \+ 16; the L1 layout is the load's: list the set "
        r"at the make, or make it with km.ANY\(...\) naming the largest set"
    )
    with pytest.raises(ProgramError, match=why):
        constant_handler(program, Gate.gate, {"size": 16}, run=FakeRun([program]))
    fresh(program, 3)
    why = r"buffer kWrite0 \(index 16\): it would hold 6 \+ 6 pages, the load's holds"
    with pytest.raises(ProgramError, match=why):  # 16 pages are not 6-page blocks
        constant_handler(program, Gate.gate, {"size": 3}, run=FakeRun([program]))


def test_the_refusals():
    program = made(handlers={"gate": km.ANY})
    why = (
        r"Gate.gate with size=4: no handler was built for it, and a first use "
        r"\(km.ANY\) is reached from run.submit only, not from this submit"
    )
    with pytest.raises(ProgramError, match=why):
        constant_handler(program, Gate.gate, {"size": 4})  # no run: another path
    with pytest.raises(ProgramError, match="gives km.ANY 2 times; give it once"):
        made(handlers={"gate": [km.ANY, {"size": 4}, km.ANY]})
    why = r"km.ANY\(width=...\) names no constant parameter of it \(size\)"
    with pytest.raises(ProgramError, match=why):
        made(handlers={"gate": km.ANY(width=16)})
    with pytest.raises(ProgramError, match="Gate.other: handlers= lists sets of a"):
        made(handlers={"other": km.ANY})
    why = r"the buffers sized for km.ANY's sets take 3,162,112 B, and 1,048,576 B"
    with pytest.raises(ProgramError, match=why):
        made(handlers={"gate": km.ANY(size=128)})
    # without the opt-in the refusal is as it was
    listed = made(handlers={"gate": [{"size": 4}]})
    why = (
        r"Gate.gate with size=8: no handler was built for it \(built: size=4\); list "
        r"it at the make, handlers=\{'gate': \[\{...\}\]\} \(a set is not compiled on "
        r"first use\)"
    )
    with pytest.raises(ProgramError, match=why):
        constant_handler(listed, Gate.gate, {}, run=FakeRun([listed]))


def test_a_field_s_shape_is_the_load_s():
    field = km.L1[km.fp32, 1, 2, 32, 32]
    loaded = first_use.Kept({}, {}, {}, None, {}, {}, {}, {"sums": field}, "", {})

    class Result:  # what a compile of the set gave
        fields = {"sums": km.L1[km.fp32, 1, 4, 32, 32]}
        buffers = []

    why = r"Gate.gate_size2: field sums would be \(1, 4, 32, 32\), and the load's is"
    with pytest.raises(ProgramError, match=why):
        first_use.load_checked(loaded, Result, None, "Gate.gate_size2")


class FakeSession:
    """What a make with session= reads of one: its L1, the load's table's rows (the
    buffers alone here), what it keeps."""

    def __init__(self, l1):
        self.free = l1

    def l1(self):
        return 0, self.free

    def l1_rows(self, program, trace=0):
        held = sum(
            b.page_bytes * (sum(b.pages) if isinstance(b.pages, tuple) else b.pages)
            for b in program.buffers
        )
        return [("buffers", held)]

    def l1_held(self):
        return []


def test_a_program_that_cannot_hold_the_added_slots_is_refused_at_its_making():
    buffers = 2 * 16 * 2048 + 2 * 32 * 2048 + 2 * 4 * 2048  # write, read, other's
    most = overlay.SERVICE_SLOT_MOST

    def with_l1(l1, handlers):
        return Gate(session=FakeSession(l1), log=quiet, handlers=handlers, **TENSORS)

    with_l1(buffers + most + 32 * 1024, {"gate": [{"size": 8}]})  # fits, no slots
    why = r"gate \(km.ANY: the added handlers' slots\): its buffers and slots do not"
    with pytest.raises(L1BudgetError, match=why):
        with_l1(buffers + most + 32 * 1024, {"gate": km.ANY})
    with_l1(buffers + 5 * 16 * 1024, {"gate": km.ANY})  # each part's 16 KiB
