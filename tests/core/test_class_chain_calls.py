"""A chain stated with a class's handlers (`Steps.scale(x, y)` in run.chain's list):
each call's description derived from its handler's lines, equal to example 29's
written one; the records name their handlers; each line a chain cannot describe
refused, saying which and why (no card)."""

import numpy as np
import pytest
from chain_fixture import (
    CORES,
    TILES,
    WIDTH,
    Handlers,
    Steps,
    host_list,
    made,
    run_of,
    tensors,
)

from komorebi import chain as kc

W = kc.words()


def chained(calls, **options):
    program = made(Steps, tensors())
    run = run_of(program)
    return program, run, run.chain(calls, cores=(0, CORES), **options)


def test_each_call_is_described_from_its_handlers_lines():
    given = tensors()
    program, run, chain = chained(host_list(given))
    scale_x, scale_y, add, total = chain.calls
    own = kc.Range(0, WIDTH, WIDTH)  # group g: tiles g x width on, its own block
    assert (scale_x.group, scale_x.size, scale_x.width) == ((0, CORES), 1, WIDTH)
    assert scale_x.input is None and scale_x.reads[0].earlier is False  # x: outside
    assert scale_y.input == scale_x.output and scale_y.ranges == (own,)
    assert add.input == scale_y.output and add.ranges == (own,)
    assert add.reads == (kc.Read(scale_x.output, None, (own,), True),)  # km.past(y)
    assert (total.size, total.width, total.k, total.sends) == (
        CORES,
        1,
        (0, TILES),
        True,
    )
    assert total.input == add.output
    words = [list(call.args[:4]) for call in chain.calls]
    assert words[0][:2] == [given["x"].address, given["y"].address]
    assert words[3][:3] == [given["s"].address, given["t"].address, TILES]


def test_the_records_name_their_handlers_and_the_command_is_the_walks():
    program, run, chain = chained(host_list(tensors()))
    records = np.frombuffer(run.backend.blocks[-1].data, np.uint32).reshape(4, 16)
    handlers = [int(word) >> W["kChainHandlerShift"] for word in records[:, 11]]
    numbers = program.handlers
    assert handlers == [
        numbers["scale"],
        numbers["scale"],
        numbers["add"],
        numbers["sum"],
    ]
    assert int(chain.handler) == numbers["km_chain"] == program.chain_handler
    assert chain.group == (0, CORES)


def test_cores_states_a_calls_group():
    given = tensors()
    calls = host_list(given)[:1]
    calls.append(Steps.scale(given["y"], given["z"], 1, WIDTH, cores=(0, 4)))
    program, run, chain = chained(calls)
    assert chain.calls[1].group == (0, 4) and chain.calls[0].group == (0, CORES)


@pytest.mark.parametrize(
    ("call", "why"),
    [
        (
            lambda g: Steps.looped(g["x"], 2),
            r"Steps.looped, line \d+: what it reads or writes changes with the "
            r"loop's i",
        ),
        (
            lambda g: Steps.scattered(g["x"], g["y"], WIDTH),
            r"Steps.scattered, line \d+: group 0 writes out pages 1 to 2; a chain's "
            r"call writes block g of its output",
        ),
        (
            lambda g: Steps.owner_sum(g["s"], g["t"], TILES),
            r"Steps.owner_sum, line \d+: a chain's sums go up a tree",
        ),
        (
            lambda g: Steps.bodied(g["x"], 1),
            r"Steps.bodied, line \d+: an inline body takes tensor a: what it reads or "
            r"writes is the body's own C\+\+",
        ),
    ],
)
def test_a_line_a_chain_cannot_describe_is_refused_and_still_runs(call, why):
    given = tensors()
    program = made(Steps, given)
    run = run_of(program)
    with pytest.raises(kc.ChainRefused, match=why + r".*; it still runs as a command"):
        run.chain([call(given)], cores=(0, CORES))
    run.submit(Steps.looped, (0, CORES), args=(given["x"], 2))  # a command as before


def test_a_program_made_without_the_walk_is_refused():
    given = tensors()
    run = run_of(made(Handlers, given))
    why = (
        r"Handlers.scale: its program was made without the chain's walk; make it "
        r"with chain=True \(or give the class a km.chain method\)"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        run.chain([Handlers.scale(given["x"], given["y"], 1, WIDTH)])


def test_a_written_call_among_a_classs_calls_is_refused():
    given = tensors()
    written = kc.Call("scale", (0,), (0, CORES), 1, WIDTH, "y", name="scale x")
    calls = [written, *host_list(given)[1:2]]
    with pytest.raises(
        kc.ChainRefused, match=r"call 0 \(scale x\): a chain.Call among"
    ):
        chained(calls)
