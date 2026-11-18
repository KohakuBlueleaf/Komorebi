"""A body on mover B that writes a tensor states it: `writes=[keys]` makes it written
by the chain's call (no prefetch of it, no outside read of it); the next call's read
of it waits for every block of the call, a later call's is refused (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, quiet, run_of, tensor

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.errors import ProgramError

APPEND = "append_row(keys, position);"


class Cache(km.Program):
    """append: a core's tiles copied, and a body on mover B writing the cache's row;
    copy: a core's tiles, copied."""

    @km.handler
    def append(self, a: km.Tensor, keys: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)
        if self.rank == self.leader:
            km.cxx(APPEND, takes=[keys, width], writes=[keys], on="mover_b")

    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)


def given():
    return {n: tensor(TILES, 0x10000 * (i + 1)) for i, n in enumerate("xykz")}


def made(t):
    return Cache(l1=1 << 20, log=quiet, a=t["x"], keys=t["k"], out=t["y"], chain=True)


def test_a_body_write_is_the_calls_and_nothing_else_of_the_chain_reads_it():
    t = given()
    program = made(t)
    chain = run_of(program).chain(
        [
            program.append(t["x"], t["k"], t["y"], WIDTH),
            program.copy(t["y"], t["z"], WIDTH),
        ],
        cores=(0, CORES),
    )
    append, copy = chain.calls
    assert append.output == "out (0x20000)" and append.outputs == ()
    assert [read.tensor for read in append.reads] == ["a (0x10000)"]  # keys: none
    assert copy.input == append.output


def test_the_next_call_s_read_of_what_a_body_wrote_waits_for_every_block():
    t = given()
    program = made(t)
    chain = run_of(program).chain(
        [
            program.append(t["x"], t["k"], t["y"], WIDTH),
            program.copy(t["k"], t["z"], WIDTH),
        ],
        cores=(0, CORES),
    )
    append, copy = chain.calls
    assert copy.input == append.output == "out (0x20000)"
    assert copy.ranges == (kc.Range(0, 0, WIDTH * CORES),) and copy.reads == ()
    assert kc.prove(chain.calls)


def test_a_read_of_what_a_body_wrote_further_back_is_refused():
    t = given()
    program = made(t)
    with pytest.raises(kc.ChainRefused) as error:
        run_of(program).chain(
            [
                program.append(t["x"], t["k"], t["y"], WIDTH),
                program.copy(t["y"], t["z"], WIDTH),
                program.copy(t["k"], t["x"], WIDTH),
            ],
            cores=(0, CORES),
        )
    message = str(error.value)
    assert "which a body of call 0 (Cache.append) writes anywhere (writes=, line" in (
        message
    )
    assert "only the next call reads it" in message


class NotTaken(km.Program):
    @km.handler
    def append(self, a: km.Tensor, keys: km.Tensor):
        km.cxx(APPEND, takes=[a], writes=[keys], on="mover_b")


def test_writes_names_a_tensor_the_body_takes():
    t = given()
    with pytest.raises(ProgramError) as error:
        NotTaken(l1=1 << 20, log=quiet, a=t["x"], keys=t["k"])
    assert "keys is in writes= and not in takes=" in str(error.value)
