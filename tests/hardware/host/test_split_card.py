"""A split card (KMR_CARD=1): card 0 as two islands, one command of a small class over
every rank, so its group crosses the islands' edge; every page exact. It closes the
shared device and opens the card as islands."""

import numpy as np
import pytest
import shared_card

import komorebi as km

pytestmark = pytest.mark.card


class Put(km.Program):
    """put: rank r of the group adds `value` to word 0 of page r."""

    @km.handler
    def put(self, out: km.Tensor, value: int):
        word = out.pages(self.rank, 1).read()
        word += value
        out.pages(self.rank, 1).write(word)


def test_a_command_over_both_islands_of_a_split_card_is_exact():
    shared_card.close_host()
    with km.Session(cards=[0], islands=2, log=print) as session:
        ranks = session.ranks
        out = session.tensor(np.zeros((ranks, 16), np.uint32), "uint32")
        run = session.load(Put(out=out, log=print), deadline=2.0)
        first = session.graph.first(1)
        assert 0 < first < ranks
        run.wait(run.submit(Put.put, (0, ranks), args=(out, 7)), timeout=10.0)
        stopped = run.stop()
        got = out.read()
    assert [block["error_name"] for block in stopped.islands] == ["none", "none"]
    assert (got[:, 0] == 7).all() and not got[:, 1:].any()
