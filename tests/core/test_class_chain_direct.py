"""km.direct(t) at a chain's call: only an input the handler declares
(km.handler(direct=(...))), only the call before's output; the class's kernels do not
take a direct input in this version, so a call that may have one is refused saying so
(no card)."""

import pytest
from chain_fixture import CORES, WIDTH, Steps, bits, made, run_of, tensors

import komorebi as km
from komorebi import chain as kc

GIVEN = tensors()
X, Y, Z, S = (GIVEN[name] for name in "xyzs")


def chain_of(calls):
    return run_of(made(Steps, GIVEN)).chain(calls, cores=(0, CORES))


def test_a_handler_declares_the_inputs_that_may_arrive_direct():
    assert Steps.add.direct == ("b",) and Steps.scale.direct == ()
    assert km.handler(direct="g")(lambda self, g: None).direct == ("g",)


def test_an_input_the_handler_does_not_declare_is_refused():
    why = (
        r"Steps.add: km.direct\(a\) marks a, which the handler does not declare "
        r"\(km.handler\(direct=\(...\)\) names b\)"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [Steps.scale(X, Z, bits(2), WIDTH), Steps.add(km.direct(Z), X, S, WIDTH)]
        )


def test_a_direct_input_is_the_call_befores_output():
    why = (
        r"call 1 \(Steps.add\): km.direct\(a \(0x10000\)\): a direct input is the call"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [Steps.scale(X, Z, bits(2), WIDTH), Steps.add(Z, km.direct(X), S, WIDTH)]
        )


def test_the_classs_kernels_do_not_take_a_direct_input_yet():
    why = (
        r"Steps.add: a direct input is declared, and the class's kernels do not take "
        r"an input from the arrival area in this version"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [Steps.scale(X, Z, bits(2), WIDTH), Steps.add(X, km.direct(Z), S, WIDTH)]
        )
