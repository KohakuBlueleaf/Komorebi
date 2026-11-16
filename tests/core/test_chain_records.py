"""Chain records at one bank address on every island: the region taken on each
island's DRAM, handed out in whole lines, given back joined; none when the islands'
places differ."""

from komorebi import chain_records


class Tensor:
    def __init__(self, address):
        self.address, self.freed = address, False

    def free(self):
        self.freed = True


class Session:
    def __init__(self, address):
        self.address, self.made = address, []

    def empty(self, shape, dtype):
        assert dtype == "uint32" and shape[1] * 4 == 4096
        self.made.append(Tensor(self.address))
        return self.made[-1]


def test_one_address_on_every_island_in_whole_lines():
    region = chain_records.Region([Session(0x40000), Session(0x40000)], 4096)
    assert region.address == 0x40000
    first, second = region.take(100), region.take(64 * 4)
    assert (first, second) == (0x40000, 0x40000 + 128)
    region.give(first, 100)
    assert region.take(128) == 0x40000  # the hole, again
    region.give(0x40000, 128)
    region.give(second, 256)
    assert region.take(4096) == 0x40000  # joined whole
    assert region.take(64) is None


def test_none_when_the_islands_differ():
    sessions = [Session(0x40000), Session(0x80000)]
    region = chain_records.Region(sessions, 4096)
    assert region.address is None and region.take(64) is None
    assert all(session.made[0].freed for session in sessions)
