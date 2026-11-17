"""An island chain of more than kChainRecordsMax records (IslandRun._submit_chain, no
card): a ticket a piece of 256 records, a part on each island naming that island's
records from the piece's first, each piece after the chain before it, the pieces
before the last waited for."""

from types import SimpleNamespace

from komorebi.chain import calls as kc
from komorebi.islands.run import IslandRun

RECORDS = (0x40000, 0x80000)  # each island's records in its own DRAM


class Part:
    """An island's Run: the chain commands queued."""

    def __init__(self, island, said):
        self.island, self.said, self.handles = island, said, 0

    def submit(self, handler, group, args, **options):
        self.handles += 1
        self.said.append((self.island, "submit", args[7], options.get("part")))
        return self.handles


def island_run():
    graph = SimpleNamespace(
        ranks=236, islands=[0, 1], first=lambda i: 118 * i, count=lambda i: 118,
        split=lambda group: [(0, 0, 118), (1, 0, 118)],
    )  # fmt: skip
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None, backend=None)] * 2)
    said = []
    run.runs = (Part(0, said), Part(1, said))
    run._chain_after = lambda ticket: said.append(("after", ticket))
    run.wait = lambda ticket: said.append(("wait", ticket))
    return run, said


def chain_of(records):
    chains = [
        kc.Chain(1, (0, 236), 0, 0, None, SimpleNamespace(address=a), {}, [0] * records)
        for a in RECORDS
    ]
    return kc.IslandChain(chains, (0, 236))


def test_a_chain_past_256_records_is_a_ticket_a_piece_the_first_waited_for():
    run, said = island_run()
    piece = 256 * 64  # a piece's records' bytes
    first = run.submit(chain_of(300))
    assert said == [
        ("after", 0),
        (0, "submit", RECORDS[0], True),
        (1, "submit", RECORDS[1], True),
        ("wait", first - 1),
        ("after", first - 1),
        (0, "submit", RECORDS[0] + piece, True),
        (1, "submit", RECORDS[1] + piece, True),
    ]
    said.clear()
    second = run.submit(chain_of(256))  # one piece, after the chain before it
    assert second == first + 1 and said[0] == ("after", first) and len(said) == 3
