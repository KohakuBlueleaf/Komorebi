"""The chain over islands on the host: the calls are taken whatever islands a mark, a
credit or a direct input crosses (ranks are flat; nothing is refused for an island)."""

from komorebi import chain as kc


def calls(sum_size=4, shift=0):
    """Scale x into y and y into z on eight ranks, a tile each; sum z in groups of
    `sum_size`, each its own tiles; `shift` moves the second scale's reads on."""
    own = kc.Range(0, sum_size, sum_size)
    return [
        kc.Call(1, (0,), (0, 8), 1, 1, "y", name="scale x"),
        kc.Call(1, (1,), (0, 8), 1, 1, "z", "y", ranges=(kc.Range(shift, 1, 1),)),
        kc.Call(2, (2,), (0, 8), sum_size, 1, "t", "z", ranges=(own,), sends=True),
    ]


def test_a_mark_from_the_next_rank_is_taken():
    kc.build(calls(shift=1))  # rank 3 waits for rank 4's mark: two islands of four


def test_a_tree_over_every_rank_is_taken():
    kc.build(calls(sum_size=8))  # its credits go between ranks 0 to 7


def test_a_direct_input_from_the_next_rank_is_taken():
    steps = calls()[:2]  # four cores 1 to 4, a share of z's 8 tiles each, sent direct
    steps.append(
        kc.Call(2, (2,), (1, 4), 4, 1, "t", "z", k=(2, 4), direct=kc.Direct(1, 4096))
    )
    report, _ = kc.build(steps)
    assert report[2][4] == (4096, 1)  # root 4 sends its tile to core 3
