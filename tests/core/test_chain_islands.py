"""The chain over islands on the host: each mark and credit inside one island, and the
refusal of one that would cross, naming the two calls and the two islands."""

import pytest

from komorebi import chain as kc


def island_of(rank):
    """Two islands of four ranks each."""
    return rank // 4


def calls(sum_size=4, shift=0):
    """Scale x into y and y into z on eight ranks, a tile each; sum z in groups of
    `sum_size`, each its own tiles; `shift` moves the second scale's reads on."""
    own = kc.Range(0, sum_size, sum_size)
    return [
        kc.Call(1, (0,), (0, 8), 1, 1, "y", name="scale x"),
        kc.Call(1, (1,), (0, 8), 1, 1, "z", "y", ranges=(kc.Range(shift, 1, 1),)),
        kc.Call(2, (2,), (0, 8), sum_size, 1, "t", "z", ranges=(own,), sends=True),
    ]


def test_marks_and_credits_inside_each_island_are_taken():
    kc.build(calls(), island_of)


def test_a_mark_from_another_island_is_refused():
    with pytest.raises(kc.ChainRefused) as error:
        kc.build(calls(shift=1), island_of)
    text = str(error.value)
    assert "call 1: core 3 on island 0" in text
    assert "call 0 (scale x)'s root 4 on island 1" in text
    assert "the far mark) is not built" in text


def test_a_credit_across_islands_is_refused():
    with pytest.raises(kc.ChainRefused) as error:
        kc.build(calls(sum_size=8), island_of)
    text = str(error.value)
    assert "call 2: group 0 (cores 0 to 7) spans islands 0 and 1" in text
    assert "a credit from one island to another (the far mark) is not built" in text


def test_a_direct_input_from_another_island_is_refused():
    steps = calls()[:2]  # four cores 1 to 4, a share of z's 8 tiles each, sent direct
    steps.append(
        kc.Call(2, (2,), (1, 4), 4, 1, "t", "z", k=(2, 4), direct=kc.Direct(1, 4096))
    )
    with pytest.raises(kc.ChainRefused) as error:
        kc.build(steps, island_of)
    text = str(error.value)
    assert "core 3 on island 0 waits for a mark from call 1's root 4" in text


def test_one_island_is_not_checked():
    kc.build(calls(shift=1, sum_size=8))  # no islands: the same calls are taken
