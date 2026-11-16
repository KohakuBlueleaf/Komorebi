"""km.direct on the call after one whose roots write their block by a write stream:
refused by run.chain, naming the stream's line (its blocks go to DRAM by
movement.move, so the roots would send nothing); by .write it is sent (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, Handlers, quiet, run_of, tensors

import komorebi as km
from komorebi import chain as kc

GIVEN = tensors()
X, Y, Z, S, T = (GIVEN[name] for name in "xyzst")


class Streamed(km.Program):
    """add writes s by a write stream, pair by .write; sum may take s direct."""

    add = km.handler(Handlers.add.function)
    sum = km.handler(direct=("s",))(Handlers.sum.function)

    @km.handler
    def pair(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * 2
        left = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first, 1)):
            left += tile
        out.pages(first, 1).write(left)
        right = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first + 1, 1)):
            right += tile
        out.pages(first + 1, 1).write(right)


def run():
    program = Streamed(l1=1 << 20, log=quiet, a=X, b=X, out=Y, s=S, t=T, chain=True)
    return run_of(program)


def test_a_write_stream_before_a_direct_input_is_refused_by_its_line():
    line = Handlers.add.function.__code__.co_firstlineno + 3  # results = km.Stream
    why = (
        r"call 1 \(Streamed.sum\): its direct input out \(0x40000\) is written by "
        rf"call 0 \(Streamed.add\) through a write stream \(line {line}\), whose "
        r"blocks go to DRAM, not to the next cores; write the block with "
        r"\.pages\(\.\.\.\)\.write"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        run().chain(
            [Streamed.add(X, Y, S, WIDTH), Streamed.sum(km.direct(S), T, TILES)],
            cores=(0, CORES),
        )


def test_the_same_stream_through_dram_and_a_block_by_write_sent_direct_are_taken():
    taken = run()
    plain = taken.chain(
        [Streamed.add(X, Y, S, WIDTH), Streamed.sum(S, T, TILES)], cores=(0, CORES)
    )
    assert plain.calls[1].direct is None
    sent = taken.chain(
        [Streamed.pair(X, S), Streamed.sum(km.direct(S), T, TILES)], cores=(0, CORES)
    )
    assert sent.calls[1].direct is not None
