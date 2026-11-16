"""The words of a call that states a constant set keep the set through a slice, a copy
and an added plain list, and words of two sets are refused naming the handler."""

import pytest

from komorebi.dsl.front import Words
from komorebi.errors import ProgramError


def test_a_slice_a_copy_and_an_added_list_keep_the_set():
    words = Words([1, 2, 3, 4], {"block": 4}, "Gate.gate")
    for kept in (words[1:], words.copy(), words[:2] + [9], [0] + words[2:]):
        assert isinstance(kept, Words)
        assert (kept.constants, kept.handler) == ({"block": 4}, "Gate.gate")
    assert words[1:] == [2, 3, 4] and words[0] == 1


def test_words_of_two_sets_joined_or_repeated_are_refused_by_handler():
    four = Words([1, 2], {"block": 4}, "Gate.gate")
    eight = Words([1, 2], {"block": 8}, "Gate.gate")
    with pytest.raises(ProgramError, match=r"Gate\.gate's words for block=4 joined"):
        four + eight
    with pytest.raises(ProgramError, match=r"Gate\.gate's words for block=4 repeated"):
        four * 2
    assert Words([1], {}, "Gate.gate") * 2 == [1, 1]
