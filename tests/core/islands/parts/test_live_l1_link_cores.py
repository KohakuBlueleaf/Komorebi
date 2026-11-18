"""A live read or write of an island's L1 tensor whose pages reach a link core is
refused on the host, by name, before any command is sent (no card)."""

from types import SimpleNamespace

import pytest

from komorebi.errors import ChannelError
from komorebi.session.parts.transfer import TransferMixin


class Island(TransferMixin):
    """One island of two cards (119 ranks, core 119 its link core), its run loaded:
    what _transfer_l1 reads, and the commands it sends."""

    def __init__(self, island):
        self.graph = SimpleNamespace(count=lambda i: 119, first=lambda i: 119 * i)
        self.island = (self.graph, island)
        self.grid = (12, 10)
        self.sent = []
        self._region = SimpleNamespace(words=0, bytes=0)
        self.channel = None
        self.run = SimpleNamespace(
            holds=lambda: [[True] * 120],
            transfer=lambda *call: self.sent.append(call),
            backend=SimpleNamespace(transfer=SimpleNamespace(address=0)),
        )


def tensor():
    return SimpleNamespace(
        address=0x10000, aligned_page_bytes=64, shard_grid=(12, 10), pages=120
    )


@pytest.mark.parametrize("push, what", [(True, "read"), (False, "write")])
def test_a_live_l1_transfer_past_the_ranks_is_refused_by_name(push, what):
    island = Island(1)
    ranks = island.graph.count(1)
    with pytest.raises(ChannelError) as refused:
        island._transfer_l1(tensor(), 0, 120, "run", push)
    text = str(refused.value)
    assert f"page {ranks} is on core {ranks} of island 1, a link core" in text
    assert f"a live {what} of an L1 tensor reaches ranks only" in text
    assert "(group=)" in text and "after run.stop()" in text
    assert island.sent == []  # nothing sent


def test_a_live_l1_transfer_of_the_ranks_alone_is_not_refused():
    island = Island(0)
    held = island._l1_ranks(tensor(), 0, island.graph.count(0))
    island._check_link_cores(tensor(), held, True)  # no refusal
