"""km.at_submit("name") at a chain's call: the handler's int word is the chain's
per-submit word (kc.Call's word), given at each run.submit(chain, name=n) with the
records unchanged; a word the call's reads depend on, two a call, and a tensor's
place are refused (no card)."""

import numpy as np
import pytest
from chain_fixture import CORES, TILES, WIDTH, Steps, made, run_of, tensors

import komorebi as km
from komorebi import abi
from komorebi.chain import calls as kc

GIVEN = tensors()


def records(run):
    return np.frombuffer(run.backend.blocks[-1].data, np.uint32).reshape(-1, 16)


def test_the_word_is_given_at_submit_and_the_records_hold_its_place():
    program = made(Steps, GIVEN)
    run = run_of(program)
    x, y = GIVEN["x"], GIVEN["y"]
    chain = run.chain(
        [program.scale(x, y, km.at_submit("factor"), WIDTH)], cores=(0, CORES)
    )
    (call,) = chain.calls
    assert call.word == ("factor", 2) and call.args[2] == 0
    assert chain.word == "factor"
    w = kc.words()
    info = records(run)[0, w["kChainInfo"]]
    token = (info >> w["kChainTokenShift"]) & w["kChainTokenMask"]
    assert token == abi.headers().positions[2][2]
    assert chain.args({"factor": 7})[0] == 7
    with pytest.raises(ValueError, match="per-submit word is factor"):
        chain.args({})


def test_the_chain_method_passes_it_to_each_call_that_takes_it():
    program = made(Steps, GIVEN)
    run = run_of(program)
    x, y, z, s, t = (GIVEN[name] for name in "xyzst")
    chain = run.chain(
        program.steps(x, y, z, s, t, km.at_submit("factor"), WIDTH, TILES),
        cores=(0, CORES),
    )
    assert [call.word for call in chain.calls] == [("factor", 2), None, None, None]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            lambda x, y: (x, y, 5, km.at_submit("width")),
            r"Steps\.scale, line \d+: what it reads or writes changes with width, "
            r"given at each submit",
        ),
        (
            lambda x, y: (x, y, km.at_submit("a"), km.at_submit("b")),
            r"Steps\.scale: km\.at_submit at factor, width: one word a call",
        ),
        (
            lambda x, y: (x, km.at_submit("out"), 5, WIDTH),
            r"Steps\.scale: out is a tensor; km\.at_submit gives an int word",
        ),
    ],
)
def test_a_word_it_cannot_be_is_refused(arguments, message):
    program = made(Steps, GIVEN)
    call = program.scale(*arguments(GIVEN["x"], GIVEN["y"]))
    with pytest.raises(kc.ChainRefused, match=message):
        run_of(program).chain([call], cores=(0, CORES))
