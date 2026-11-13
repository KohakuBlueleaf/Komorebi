"""A message names its handler: word 0's packing (lap mark, handler, program) and the
host's declaration of several dynamic handlers."""

import pytest

import komorebi as km
from komorebi import abi
from komorebi.program import Dynamic, message_word, messages_header


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
