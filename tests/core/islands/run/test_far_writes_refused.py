"""A handler's write into a tensor another island holds (no card): the build keeps the
writes that run at every call; on Wormhole a command whose ranks on an island write
such a tensor at every call while that island holds none of it is refused by name;
the stop's error raised in another keeps it as its context."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.errors import CommandTimeout, KomorebiError
from komorebi.islands import local as island_local
from komorebi.session.parts.run_stop import StopMixin

RANKS = 78  # each island's ranks


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def row(pages):
    return Tensor(Pages(2048, pages), (32, 32 * pages), "bfloat16", "tile")


@km.tile_function
def doubled(a):
    return a * 2.0


class Twice(km.Program):
    @km.handler
    def double(self, x: km.Tensor, y: km.Tensor):
        c = self.rank
        out = km.Stream(y.tiles[0:1, c : c + 1], write=True, block=1)
        for a in km.Stream(x.tiles[0:1, c : c + 1], block=1):
            out.push(doubled(a))

    @km.handler
    def first(self, x: km.Tensor, y: km.Tensor):
        if self.rank == 0:
            out = km.Stream(y.tiles[0:1, 0:1], write=True, block=1)
            for a in km.Stream(x.tiles[0:1, 0:1], block=1):
                out.push(doubled(a))


def held(*parts, copy=False):
    return SimpleNamespace(placement=SimpleNamespace(copy=copy, parts=list(parts)))


ON_0 = held((0, 2 * RANKS), (0, 0))  # every page on island 0
SPLIT = held((0, RANKS), (RANKS, RANKS))


@pytest.fixture(scope="module")
def twice():
    return Twice(l1=1 << 20, log=lambda line: None, x=row(8), y=row(8))


def check(program, method, y, island=1, group=(0, 2 * RANKS), arch="wormhole_b0"):
    ranks = (island * RANKS, (island + 1) * RANKS)
    island_local.check_far_writes(
        program, method, (SPLIT, y), island, group, ranks, arch
    )


def test_the_build_keeps_the_writes_at_every_call(twice):
    (write,) = twice.writes_every_call["double"]
    assert write[0] == "y" and write[2].startswith("out = km.Stream(y.tiles")
    assert "first" not in twice.writes_every_call  # its write is under an if


def test_a_write_into_a_tensor_the_island_holds_none_of_is_refused(twice):
    with pytest.raises(KomorebiError) as refused:
        check(twice, Twice.double, ON_0)
    text = str(refused.value)
    assert text.startswith("Twice.double, line ")
    assert "y's pages are held by island(s) 0" in text
    assert "holds ranks 78 to 155 of island 1, which holds none of them" in text
    assert "place y by hint so each rank writes its own island's pages" in text


def test_what_the_card_can_write_is_not_refused(twice):
    check(twice, Twice.double, ON_0, island=0)  # island 0 holds y
    check(twice, Twice.double, ON_0, group=(0, RANKS))  # no rank of island 1
    check(twice, Twice.double, SPLIT)  # island 1 holds its part
    check(twice, Twice.double, held((0, 2 * RANKS), (0, 0), copy=True))
    check(twice, Twice.first, ON_0)  # under an if: the card decides
    check(twice, Twice.double, ON_0, arch="blackhole")  # far DRAM puts cross
    check(twice, Twice.double, ON_0, arch=None)  # no card's device: a mock


class Stopping:
    _raise_after_stop = StopMixin._raise_after_stop
    _remember = StopMixin._remember
    _log = StopMixin._log
    backend = SimpleNamespace()


def test_the_stop_s_error_keeps_the_error_it_was_raised_in():
    with pytest.raises(CommandTimeout) as raised:
        try:
            raise ValueError("the handler's command was given up")
        except ValueError:
            Stopping()._raise_after_stop(CommandTimeout(6, 0, 6, (0, 79), []), None)
    assert isinstance(raised.value.__context__, ValueError)
    assert not raised.value.__suppress_context__
