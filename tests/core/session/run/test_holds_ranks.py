"""An island's holds: a load over a flat group marks only the island's ranks, never
its link cores or a dispatcher's core after them, and a session's write picks its
transfer group among the ranks (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import Session, abi
from komorebi.device import channel as ch
from komorebi.session.run import Run

H = abi.headers()
REGION = ch.check_region(4 << 20)
CORES = 80  # a Wormhole grid, 8 by 10: ranks, then link cores, then the dispatcher


def island_run(first, ranks, held_lengths):
    """An island's Run with no card: the columns _part, _held and holds() read; one
    holds array a program, of the lengths given (a class added live: the grid's)."""
    run = Run.__new__(Run)
    run._link_base = H["KM_GROUP_LINK_BASE"]
    run._cluster_base = H.constants["KM_GROUP_CLUSTER_BASE"]
    run.cores, run.island_first, run.island_ranks = CORES, first, ranks
    run.island_cores, run.island_span = (0, CORES), CORES
    run._holds = [np.zeros(length, bool) for length in held_lengths]
    return run


@pytest.mark.parametrize(
    "first, ranks", [(0, 78), (78, 77)], ids=["island 0", "island 1"]
)
def test_a_flat_group_over_islands_holds_only_the_islands_ranks(first, ranks):
    run = island_run(first, ranks, (ranks, CORES, CORES))  # islands of unequal ranks
    run._held(1, 0, 78 + 77)  # every flat rank of the two islands
    assert np.flatnonzero(run._holds[1]).tolist() == list(range(ranks))
    assert not run._holds[0].any() and not run._holds[2].any()
    run._held(2, first, ranks)  # the island's own ranks switched to another
    assert not run._holds[1].any()
    assert np.flatnonzero(run._holds[2]).tolist() == list(range(ranks))
    low, high = run._part(np.array([0, first]), np.array([78 + 77, ranks + 5]))
    assert high.max() == ranks  # a flat group's part ends at the island's last rank
    link = run._part(run._link_base, 1)
    assert (int(link[0]), int(link[1])) == (ranks, ranks + 1)  # its link core


class TransferRun:
    """An island's live run: its holds as given; transfer() records its group and
    writes each rank's sequence word, as the cores do."""

    def __init__(self, session, holds, ranks):
        self.session, self.cores, self.island_ranks = session, CORES, ranks
        self._holds, self.sent = holds, []
        self.backend = SimpleNamespace(
            transfer=SimpleNamespace(chunk=16 << 10, address=0x50000)
        )

    def holds(self):
        return self._holds

    def transfer(self, handler, group, args, timeout):
        self.sent.append(group)
        t = ch.Transfer(
            *[args[H[f"KM_TRANSFER_{n.upper()}"]] for n in ch.Transfer._fields]
        )
        want = ch.sequence_words(t.nonce, t.pages, t.page, group[1])
        ch.words_view(self.session.channel, REGION, group[1])[:, :4] = want


@pytest.mark.usefixtures("blackhole_card")
@pytest.mark.parametrize("pages", [1, 2, 72])
def test_a_write_after_a_switch_picks_its_transfer_group_among_the_ranks(pages):
    session = Session(channel=REGION.bytes)
    session.channel = np.zeros(REGION.bytes, np.uint8)
    session._dram_banks = 8
    marked = np.zeros(CORES, bool)
    marked[78:] = True  # a class marked on the cores past the ranks (as it was)
    switched = np.zeros(CORES, bool)
    switched[:78] = True  # the ranks switched to the class added after it
    session.run = TransferRun(session, [np.zeros(78, bool), marked, switched], 78)
    x = SimpleNamespace(
        in_dram=True, address=0x1000, aligned_page_bytes=2048, page_bytes=2048,
        pages=pages,
    )  # fmt: skip
    session._transfer(x, 0, pages, "run", False)
    first, count, field = session.run.sent[0]
    assert field == 2 and first + count <= 78
    made = session.channel_transfer(x, push=True)
    assert made.entry.field == 2 and sum(made.entry.group) <= 78
