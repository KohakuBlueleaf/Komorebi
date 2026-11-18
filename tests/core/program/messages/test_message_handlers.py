"""A message names its handler: word 0's packing (lap mark, handler, program), the
host's declaration of several dynamic handlers, and of handlers as overlay parts."""

import pytest

import komorebi as km
from komorebi import abi
from komorebi.errors import ProgramError
from komorebi.program.description import Dynamic, message_word, messages_header
from komorebi.program.overlay import Overlay


def device_mark(position, shift):
    """messages_mark of km_messages.hpp, step by step: the lap folded once mod 65535."""
    lap = position >> shift
    folded = (lap & 0xFFFF) + (lap >> 16)
    folded = folded - 0xFFFF if folded >= 0xFFFF else folded
    return folded + 1


def test_the_device_fold_is_the_lap_mod_65535_for_every_position_below_2_to_32():
    for shift in (5, 7, 16):
        for lap in (0, 1, 65534, 65535, 65536, 131070, (1 << (32 - shift)) - 1):
            position = lap << shift
            assert device_mark(position, shift) == lap % 0xFFFF + 1
            assert message_word(position, 1 << shift, 0) == lap % 0xFFFF + 1


def test_a_slot_never_holds_the_mark_its_reader_waits_for():
    queue = 32
    for position in (queue, 65535 * queue, 65536 * queue, 70000 * queue + 5):
        mark = message_word(position, queue, 9) & 0xFFFF
        assert mark not in (0, message_word(position - queue, queue, 9) & 0xFFFF)


def test_handler_and_program_sit_above_the_mark():
    constants, _ = messages_header()
    assert constants["kMessagesMarkMask"] == 0xFFFF
    word = message_word(3 * 64 + 1, 64, 0xAB, 0x5C)
    assert word == 4 | 0xAB << 16 | 0x5C << 24
    assert word & constants["kMessagesMarkMask"] == 4
    assert constants["kMessagesOutDestination"] >= abi.headers()["KM_CORES_MAX"] - 1
    assert constants["kMessagesEnd"] >> 24 != 0  # never a destination with a handler


def test_several_dynamic_handlers_pack_one_a_byte():
    dynamic = Dynamic(handler=5, handlers=(5, 9, 200))
    assert dynamic.word() == 5 | 9 << 8 | 200 << 16
    assert dynamic.args(0, 0)[0] == dynamic.word()
    assert Dynamic(handler=7).handlers == (7,) and Dynamic(handler=7).word() == 7


@pytest.mark.parametrize(
    "handlers", [(1, 2, 3, 4, 5), (1, 1), (2, 3), (1, 240), (1, 0)]
)
def test_bad_dynamic_handler_sets_are_rejected(handlers):
    with pytest.raises(ValueError, match="dynamic handlers"):
        Dynamic(handler=1, handlers=handlers)


def test_a_program_declares_its_dynamic_handlers_by_name():
    names = {"lookup": 1, "value": 2, "report": 3, "plain": 4}
    program = km.Program("handlers", ".", handlers=names)
    program.dynamic("lookup", "value", "report", queue_messages=1 << 12)
    assert program.messages.handlers == (1, 2, 3)
    assert program.messages.handler == 1 and program.messages.queue_messages == 1 << 12
    with pytest.raises(ValueError, match="already"):
        program.dynamic("plain")
    one = km.Program("one", ".", handlers=names).dynamic("value")
    assert one.messages.handlers == (2,) and one.messages.word() == 2
    with pytest.raises(KeyError):
        km.Program("missing", ".", handlers=names).dynamic("lookup", "nothing")


def parts_folder(tmp_path, names):
    for name in names:
        (tmp_path / f"{name}.cpp").write_text("// a handler's part\n")
    return tmp_path


def test_handlers_as_parts_are_overlays_under_the_runtimes_kernels(tmp_path):
    names = {"lookup": 1, "value": 2, "report": 3}
    folder = parts_folder(tmp_path, names)
    program = km.Program("parts", None, handlers=names)
    program.dynamic("value", "lookup", "report", parts=folder)
    assert [(o.name, o.number, o.message) for o in program.overlays.values()] == [
        ("value", 1, True),
        ("lookup", 2, True),
        ("report", 3, True),
    ]
    assert program.overlays["lookup"].sources == {"compute": folder / "lookup.cpp"}
    assert {
        f"{path.parent.name}/{path.name}" for path in program.sources().values()
    } == {
        "messages/reader.cpp",
        "messages/compute.cpp",
        "messages/writer.cpp",
    }
    assert program.messages.defines() == [
        ("KM_MESSAGE_PARTS", "1"),
        ("KM_MESSAGE_HANDLERS", "2,1,3"),
    ]
    assert program.overlay_slots.of("compute", program.overlays.values()) == (3, None)


def test_parts_allow_sixteen_handlers_and_a_run_names_one_of_the_first_four():
    most = messages_header()[0]["kMessagesHandlersMax"]
    dynamic = Dynamic(handler=1, handlers=tuple(range(1, most + 1)), parts=True)
    assert dynamic.word() == 1 | 2 << 8 | 3 << 16 | 4 << 24
    with pytest.raises(ValueError, match="dynamic handlers"):
        Dynamic(handler=1, handlers=tuple(range(1, most + 2)), parts=True)


def test_parts_and_a_programs_own_overlays_are_refused_together(tmp_path):
    names = {"lookup": 1, "value": 2}
    folder = parts_folder(tmp_path, ["lookup", "value", "reader"])
    program = km.Program("both", None, handlers=names)
    program.overlay("own", {"reader": folder / "reader.cpp"}, number=5)
    with pytest.raises(ProgramError, match="overlays of its own"):
        program.dynamic("lookup", "value", parts=folder)
    program = km.Program("parts", None, handlers=names)
    program.dynamic("lookup", "value", parts=folder)
    with pytest.raises(ProgramError, match="no overlays of its own"):
        program.overlay("own", {"reader": folder / "reader.cpp"}, number=5)


def test_a_missing_part_is_refused(tmp_path):
    names = {"lookup": 1, "value": 2}
    folder = parts_folder(tmp_path, ["lookup"])
    with pytest.raises(FileNotFoundError, match="value"):
        km.Program("p", None, handlers=names).dynamic("lookup", "value", parts=folder)
    with pytest.raises(ValueError, match="one compute part"):
        Overlay("lookup", 1, {"reader": folder / "lookup.cpp"}, message=True)
