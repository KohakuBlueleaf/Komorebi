"""An inline body that takes a tensor states what it reads: `reads=[t]` makes the
tensor an outside input of the chain's call, refused when a call of the chain writes
it; a body taking a tensor it states nothing of is refused in a chain, and reads=
that names no taken tensor at the build (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, quiet, run_of, tensor

import komorebi as km
from komorebi import chain as kc
from komorebi.errors import ProgramError

ROTARY = "rotate(compute_rows(table, position));"


class Rotary(km.Program):
    """rotate: a body on mover A reads the table; copy: a core's tiles, copied."""

    @km.handler
    def rotate(self, a: km.Tensor, table: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)
        km.cxx(ROTARY, takes=[table, width], reads=[table], on="mover_a")

    @km.handler
    def unstated(self, a: km.Tensor, table: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)
        km.cxx(ROTARY, takes=[table, width], on="mover_a")

    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)


def given():
    return {n: tensor(TILES, 0x10000 * (i + 1)) for i, n in enumerate("xytz")}


def made(t):
    return Rotary(l1=1 << 20, log=quiet, a=t["x"], table=t["t"], out=t["y"], chain=True)


def test_a_body_read_is_an_outside_input_of_the_call():
    t = given()
    program = made(t)
    chain = run_of(program).chain(
        [
            program.copy(t["x"], t["y"], WIDTH),
            program.rotate(t["y"], t["t"], t["z"], WIDTH),
        ],
        cores=(0, CORES),
    )
    rotate = chain.calls[1]
    assert rotate.input == chain.calls[0].output
    assert rotate.reads == (kc.Read("table (0x30000)", None, (), False),)


def test_a_body_read_of_a_tensor_the_chain_writes_is_refused():
    t = given()
    program = made(t)
    with pytest.raises(kc.ChainRefused) as error:
        run_of(program).chain(
            [
                program.copy(t["x"], t["t"], WIDTH),
                program.rotate(t["x"], t["t"], t["z"], WIDTH),
            ],
            cores=(0, CORES),
        )
    message = str(error.value)
    assert "body reads" in message and "an outside input of the chain" in message
    assert "call 0 (Rotary.copy) writes it" in message


def test_a_body_taking_a_tensor_it_states_nothing_of_is_refused_in_a_chain():
    t = given()
    program = made(t)
    with pytest.raises(kc.ChainRefused) as error:
        run_of(program).chain(
            [program.unstated(t["x"], t["t"], t["y"], WIDTH)], cores=(0, CORES)
        )
    assert "state it at the body: reads=[table]" in str(error.value)


class NotTaken(km.Program):
    @km.handler
    def rotate(self, a: km.Tensor, table: km.Tensor):
        km.cxx(ROTARY, takes=[a], reads=[table], on="mover_a")


class NotTensor(km.Program):
    @km.handler
    def rotate(self, a: km.Tensor, n: int):
        km.cxx(ROTARY, takes=[a, n], reads=[n], on="mover_a")


class NotList(km.Program):
    @km.handler
    def rotate(self, a: km.Tensor):
        km.cxx(ROTARY, takes=[a], reads=a, on="mover_a")


@pytest.mark.parametrize(
    "cls, says",
    [
        (NotTaken, "table is in reads= and not in takes="),
        (NotTensor, "n in reads= is not a tensor parameter"),
        (NotList, "reads= is a list of the tensors the body reads"),
    ],
)
def test_reads_names_tensors_the_body_takes(cls, says):
    t = given()
    with pytest.raises(ProgramError) as error:
        cls(l1=1 << 20, log=quiet, a=t["x"], table=t["t"])
    assert says in str(error.value) and "line " in str(error.value)
