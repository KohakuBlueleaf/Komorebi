"""km.direct(t) at a chain's call: only an input the handler declares
(km.handler(direct=(...))), only the call before's output read by a K share; each root
sends its block's part to every core whose share meets it, the tensor then refused to a
host read and to a later call of the chain (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, Steps, bits, made, run_of, tensors

import komorebi as km
from komorebi import chain as kc
from komorebi import tensor as tensor_module

GIVEN = tensors()
X, Y, Z, S, T = (GIVEN[name] for name in "xyzst")


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


def test_a_direct_input_read_by_each_rank_s_range_is_refused():
    why = r"call 1 \(Steps.add\): .*a direct input is the call before's output, read by"
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of(
            [Steps.scale(X, Z, bits(2), WIDTH), Steps.add(X, km.direct(Z), S, WIDTH)]
        )


class DirectSum(Steps):
    """Steps whose sum may take s direct: a K share of it a core."""

    sum = km.handler(direct=("s",))(Steps.sum.function)


def test_a_k_share_from_several_roots_is_sent_by_each_and_not_read_from_dram():
    program = made(DirectSum, GIVEN)
    assert program.direct_area == (TILES, 4096, [("add", "b"), ("sum", "s")])
    run = run_of(program)
    chain = run.chain(
        [DirectSum.add(X, Y, S, WIDTH), DirectSum.sum(km.direct(S), T, TILES)],
        cores=(0, CORES),
    )
    assert chain.report == {1: [(WIDTH * 4096, 1)] * CORES}  # a root to one core
    assert chain.direct == [(S, "call 0 (DirectSum.add) of a chain")]
    assert chain.written == [T]
    tensor_module.sent_direct(chain.written, chain.direct)
    with pytest.raises(RuntimeError, match=r"sent direct by call 0 \(DirectSum.add\)"):
        S.read_bytes()
    tensor_module.sent_direct([S], [])  # a later chain that writes it
    assert S.sent is None
    with pytest.raises(
        kc.ChainRefused,
        match=r"call 2 \(DirectSum.add\): reads out .*, which call 1 .* takes direct",
    ):
        run.chain(
            [
                DirectSum.add(X, Y, S, WIDTH),
                DirectSum.sum(km.direct(S), T, TILES),
                DirectSum.add(km.past(S), Y, Z, WIDTH),
            ],
            cores=(0, CORES),
        )
