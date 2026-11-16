"""km.past(t) at a chain's call: an input written by a call further back than the one
before, proven complete where each core reads it or refused naming the call, the
tensor and the core; an unmarked one refused; a mark out of place refused (no card)."""

import pytest
from chain_fixture import CORES, THREE, WIDTH, Steps, bits, made, run_of, tensors

import komorebi as km
from komorebi.chain import calls as kc


def chain_of(calls):
    run = run_of(made(Steps, tensors_now))
    return run.chain(calls, cores=(0, CORES))


tensors_now = tensors()
X, Y, Z, S = (tensors_now[name] for name in "xyzs")


def test_a_marked_input_from_further_back_is_proven():
    chain = chain_of(
        [
            Steps.scale(X, Y, bits(2), WIDTH),
            Steps.scale(Y, Z, THREE, WIDTH),
            Steps.add(Z, km.past(Y), S, WIDTH),
        ]
    )
    assert chain.calls[2].reads[0].earlier is True


def test_an_unmarked_input_from_further_back_is_refused():
    why = (
        r"call 2 \(Steps.add\): reads out \(0x20000\), written by call 0 "
        r"\(Steps.scale\) further back; mark it km.past\(out \(0x20000\)\) at the call"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [
                Steps.scale(X, Y, bits(2), WIDTH),
                Steps.scale(Y, Z, THREE, WIDTH),
                Steps.add(Z, Y, S, WIDTH),
            ]
        )


def test_a_marked_input_not_complete_where_read_is_refused_naming_the_core():
    why = (
        r"call 2 \(Steps.add_next\): core 0 reads out \(0x20000\) tile 2, written by "
        r"call 0 \(Steps.scale\) \(block 1\), and nothing it waits for tells it"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [
                Steps.scale(X, Y, bits(2), WIDTH),
                Steps.scale(Y, Z, THREE, WIDTH),
                Steps.add_next(Z, km.past(Y), S, WIDTH),
            ]
        )


def test_a_mark_out_of_place_is_refused():
    why = (
        r"call 1 \(Steps.scale\): km.past marks out \(0x20000\), the call before's "
        r"output"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [Steps.scale(X, Y, bits(2), WIDTH), Steps.scale(km.past(Y), Z, 1, WIDTH)]
        )
    why = r"call 0 \(Steps.scale\): km.past marks a tensor its handler does not read"
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of([Steps.scale(X, km.past(Y), bits(2), WIDTH)])
