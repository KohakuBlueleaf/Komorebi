"""A chain's call with several outputs: each written as block g of its group by the
root, before the root's mark; a call after reads any of them, waiting for every block
of the call before when its input's marks do not cover what it reads (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, made, quiet, run_of, tensor

import komorebi as km
from komorebi import chain as kc


class Pairs(km.Program):
    """pair: a core's WIDTH tiles copied, and their sum as one tile of a second
    output; whole: reads its input's own block and every core's sum tile; own: its
    own sum tile only."""

    @km.handler
    def pair(self, a: km.Tensor, out: km.Tensor, sq: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(first, width)):
            total += tile
        sq.pages(self.rank, 1).write(total)

    @km.handler
    def whole(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int, n: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)
        km.Stream(b.pages(0, n)).drain()

    @km.handler
    def own(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)
        km.Stream(b.pages(self.rank, 1)).drain()


def given():
    names = ("x", "y", "s", "z", "w")
    return {n: tensor(TILES, 0x10000 * (i + 1)) for i, n in enumerate(names)}


def program():
    t = given()
    return t, Pairs(
        l1=1 << 20, log=quiet, a=t["x"], b=t["s"], out=t["y"], sq=t["s"], chain=True
    )


def test_a_call_writes_two_outputs_each_its_block_a_group():
    t, made_program = program()
    run = run_of(made_program)
    chain = run.chain(
        [
            made_program.pair(t["x"], t["y"], t["s"], WIDTH),
            made_program.own(t["y"], t["s"], t["z"], WIDTH),
        ],
        cores=(0, CORES),
    )
    pair, own = chain.calls
    assert (pair.output, pair.width) == ("out (0x20000)", WIDTH)
    assert pair.outputs == (("sq (0x30000)", 1),)
    assert own.input == pair.output and own.ranges == (kc.Range(0, WIDTH, WIDTH),)
    assert own.reads == (kc.Read("sq (0x30000)", None, (kc.Range(0, 1, 1),), True),)


def test_a_read_its_input_does_not_cover_waits_for_every_block():
    t, made_program = program()
    run = run_of(made_program)
    chain = run.chain(
        [
            made_program.pair(t["x"], t["y"], t["s"], WIDTH),
            made_program.whole(t["y"], t["s"], t["z"], WIDTH, CORES),
        ],
        cores=(0, CORES),
    )
    _, whole = chain.calls
    assert whole.k is None and whole.ranges == (kc.Range(0, 0, TILES),)
    assert whole.reads == (
        kc.Read("sq (0x30000)", None, (kc.Range(0, 0, CORES),), True),
    )


def test_a_second_output_read_further_back_is_marked_past():
    t, made_program = program()
    run = run_of(made_program)
    calls = [
        made_program.pair(t["x"], t["y"], t["s"], WIDTH),
        made_program.own(t["y"], t["x"], t["z"], WIDTH),
        made_program.own(t["z"], t["s"], t["w"], WIDTH),
    ]
    with pytest.raises(kc.ChainRefused) as error:
        run.chain(calls, cores=(0, CORES))
    assert "written by call 0" in str(error.value) and "km.past" in str(error.value)
    calls[2] = made_program.own(t["z"], km.past(t["s"]), t["w"], WIDTH)
    third = run.chain(calls, cores=(0, CORES)).calls[2]
    assert third.reads[0].earlier


def test_the_framework_proves_each_output_by_its_own_width():
    call = kc.Call(None, (), (0, 4), 1, 2, "y", outputs=(("s", 1),))
    reader = kc.Call(None, (), (0, 4), 1, 2, "z", "y", ranges=(kc.Range(0, 2, 2),))
    late = kc.Call(None, (), (0, 4), 1, 1, "w", "z", ranges=(kc.Range(0, 1, 1),))
    late = late._replace(reads=(kc.Read("s", None, (kc.Range(0, 0, 4),), True),))
    with pytest.raises(kc.ChainRefused) as error:
        kc.prove([call, reader, late])
    assert "reads s tile 1, written by call 0 (block 1)" in str(error.value)
    ok = late._replace(reads=(kc.Read("s", None, (kc.Range(0, 1, 1),), True),))
    assert kc.prove([call, reader, ok])


def test_a_direct_input_from_a_call_of_several_outputs_is_refused():
    call = kc.Call(None, (), (0, 4), 1, 2, "y", outputs=(("s", 1),))
    direct = kc.Call(
        None, (), (0, 4), 1, 2, "z", "y", k=(0, 8), direct=kc.Direct(8, 2048)
    )
    with pytest.raises(kc.ChainRefused) as error:
        kc.check_direct([call, direct])
    assert "writes several outputs" in str(error.value)


def test_two_outputs_given_one_tensor_are_refused():
    t, made_program = program()
    run = run_of(made_program)
    with pytest.raises(kc.ChainRefused) as error:
        run.chain([made_program.pair(t["x"], t["y"], t["y"], WIDTH)], cores=(0, CORES))
    assert "the same tensor as another output" in str(error.value)


assert made  # the fixture's own maker is not used: Pairs takes its own tensors
