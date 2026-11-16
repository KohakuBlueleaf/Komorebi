"""A handler's constant parameters stated in its words (`args(...)`): submit_many,
Graph.command, a stream's submit and a chain's call run the handler built for the
values the words carry; plain words mean the defaults; each refusal (no card)."""

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.device.tensor import Tensor
from komorebi.engines.graph import Graph
from komorebi.engines.stream import Stream
from komorebi.errors import ProgramError
from komorebi.session.run import Run

W = kc.words()
ADDRESS = 0x10_0000


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def grid(rows, columns):
    """A bfloat16 tensor of tiles, rows x columns of them."""
    return Tensor(Pages(2048, rows * columns), (rows * 32, columns * 32), "bfloat16",
                  "tile")  # fmt: skip


class Block:
    def __init__(self, data):
        self.address, self.data, self.freed = ADDRESS, bytes(data), False

    def free(self):
        self.freed = True


class BlockBackend(FakeBackend):
    """A fake backend that keeps a chain's records."""

    def __init__(self, port):
        super().__init__(port)
        self.blocks = []

    def block(self, data, place=None):
        self.blocks.append(Block(data if place is None else place(ADDRESS)))
        return self.blocks[-1]

    def fence(self):
        pass


def quiet(line):
    pass


A, OUT = grid(1, 8), grid(1, 8)


@km.tile_function
def twice(a):
    return a + a


class Picked(km.Program):
    @km.handler
    def gate(
        self, a: km.Tensor, out: km.Tensor, tiles: int, size: km.constant[int] = 8
    ):
        results = km.Stream(out.pages(0, tiles), write=True, block=size)
        for x in km.Stream(a.pages(0, tiles), block=size):
            results.push(twice(x))

    @km.handler
    def other(self, a: km.Tensor):
        km.Stream(a.pages(0, 1)).drain()


def made():
    sets = {"gate": [{"size": 4}, {"size": 8}]}
    return Picked(l1=1 << 20, log=quiet, a=A, out=OUT, handlers=sets)


def run_of(program):
    port = FakePort()
    return Run(BlockBackend(port), (12, 10), program=program), port


def test_the_words_carry_the_stated_values_and_are_a_list_as_before():
    program = made()
    four = program.gate.args(A, OUT, 3, 4)
    assert four == [A.address, OUT.address, 3] and isinstance(four, list)
    assert four.constants == {"size": 4}
    assert Picked.gate.args(A, OUT, 3).constants == {}  # left out: the default
    assert Picked.gate.args(A, OUT, 3, size=4).constants == {"size": 4}
    assert four[:2].constants == {"size": 4}  # a slice keeps its set
    assert (four + [0]).constants == {"size": 4}  # so does a plain list added
    assert Picked.other.args(A).constants == {}


def test_submit_many_runs_the_handler_built_for_its_rows():
    program = made()
    run, port = run_of(program)
    four = Picked.gate.args(A, OUT, 3, 4)
    handles = list(run.submit_many(Picked.gate, [0, 4], [4, 4], [four, four]))
    handles += list(run.submit_many(program.gate, 0, 4, [Picked.gate.args(A, OUT, 3)]))
    handles += list(run.submit_many(Picked.gate, 0, 4, np.array([list(four)])))
    run.wait(handles)
    numbers = program.handlers
    assert [one[3] for one in port.submitted] == [
        numbers["gate_size4"],
        numbers["gate_size4"],
        numbers["gate_size8"],
        numbers["gate_size8"],  # plain words: the default, 8
    ]


def test_submit_many_refuses_rows_of_two_sets_and_a_set_not_built():
    program = made()
    run, _ = run_of(program)
    four, eight = (Picked.gate.args(A, OUT, 3, size) for size in (4, 8))
    why = (
        r"Picked.gate: rows 0 and 1 of the words state size=4 and size=8; "
        r"submit_many runs one handler, so submit each set apart"
    )
    with pytest.raises(ProgramError, match=why):
        run.submit_many(Picked.gate, [0, 4], [4, 4], [four, eight])
    why = r"Picked.gate with size=2: no handler was built for it \(built: size=4; "
    with pytest.raises(ProgramError, match=why):
        run.submit_many(Picked.gate, 0, 4, [Picked.gate.args(A, OUT, 3, 2)])


def test_a_graph_and_a_stream_take_the_handler_built_for_the_words():
    program = made()
    four = program.gate.args(A, OUT, 3, 4)
    numbers = program.handlers
    graph = Graph()
    graph.command(program.gate, group=(0, 4), args=four)
    graph.command(program.gate, group=(0, 4), args=list(four))  # the default
    assert [t.rec[1] & 0xFF for t in graph.templates()] == [
        numbers["gate_size4"],
        numbers["gate_size8"],
    ]
    stream = Stream()
    stream.submit(program.gate, (0, 4), args=four)
    stream.enqueue(program.gate, 4, args=program.gate.args(A, OUT, 3))
    assert [t.rec[1] & 0xFF for t in stream.templates()] == [
        numbers["gate_size4"],
        numbers["gate_size8"],
    ]


def test_a_graph_and_a_stream_refuse_the_class_form_of_such_a_handler():
    made()
    four = Picked.gate.args(A, OUT, 3, 4)
    why = (
        r"Picked.gate: a handler with constant parameters is named through its "
        r"program in a graph \(program.gate\): a graph has no run"
    )
    with pytest.raises(ProgramError, match=why):
        Graph().command(Picked.gate, group=(0, 4), args=four)
    with pytest.raises(ProgramError, match="named through its program in a stream"):
        Stream().submit(Picked.gate, (0, 4), args=four)


def test_a_chains_call_runs_the_handler_built_for_its_words():
    program = made()
    run, _ = run_of(program)
    own = kc.Range(0, 1, 1)
    calls = [
        kc.Call(program.gate, program.gate.args(A, OUT, 3, 4), (0, 8), 1, 1, "y"),
        kc.Call(Picked.gate, Picked.gate.args(A, OUT, 3), (0, 8), 1, 1, "z", "y",
                ranges=(own,)),  # fmt: skip
    ]
    chain = run.chain(calls, mode=3)
    records = np.frombuffer(run.backend.blocks[-1].data, np.uint32).reshape(2, 16)
    handlers = [int(word) >> W["kChainHandlerShift"] for word in records[:, 11]]
    assert handlers == [program.handlers["gate_size4"], program.handlers["gate_size8"]]
    assert int(chain.handler) == program.handlers["gate_size4"]
    assert list(records[0, :3]) == [A.address, OUT.address, 3]
    calls[1] = calls[1]._replace(args=Picked.gate.args(A, OUT, 3, 2))
    with pytest.raises(ProgramError, match="size=2: no handler was built"):
        run.chain(calls, mode=3)
