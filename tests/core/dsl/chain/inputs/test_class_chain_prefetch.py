"""prefetch: a read of a tensor no call of the chain writes goes before the record's
input is marked (each record's mask of the chain-written tensors), off for a chain
(run.chain(..., prefetch=False)) or a stream (km.Stream(..., prefetch=False)), and
stated (prefetch=True) proven or refused at its line (no card)."""

import pytest
from chain_fixture import CORES, WIDTH, Steps, bits, host_list, made, run_of, tensors

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.dsl.compile.records import Refused

GIVEN = tensors()
X, Y, Z = (GIVEN[name] for name in "xyz")


def chain_of(calls, **options):
    program = made(Steps, GIVEN)
    return program, run_of(program).chain(calls, cores=(0, CORES), **options)


def test_each_record_carries_the_mask_of_its_chain_written_tensors():
    program, chain = chain_of(host_list(GIVEN))
    masks = [call.args[len(call.args) - 1] for call in chain.calls]
    assert masks == [0b10, 0b11, 0b111, 0b11]  # x is written by no call: bit 0 clear
    assert [len(call.args) for call in chain.calls] == [5, 5, 5, 4]
    program, chain = chain_of(host_list(GIVEN), prefetch=False)
    assert {call.args[len(call.args) - 1] for call in chain.calls} == {0xFFFFFFFF}


def test_prefetch_true_on_a_tensor_the_chain_writes_is_refused_at_its_line():
    why = (
        r"call 1 \(Steps.early\): line \d+ reads out \(0x20000\) with prefetch=True, "
        r"and a call of the chain writes it"
    )
    with pytest.raises(kc.ChainRefused, match=why):
        chain_of([Steps.scale(X, Y, bits(2), WIDTH), Steps.early(Y, Z, WIDTH)])
    chain_of([Steps.early(X, Y, WIDTH)])  # x: written by no call, proven


class Unstated(km.Program):
    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        results = km.Stream(out.pages(0, width), write=True)
        for tile in km.Stream(a.pages(0, width), prefetch=1):
            results.push(tile)


class Written(km.Program):
    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        results = km.Stream(out.pages(0, width), write=True, prefetch=False)
        for tile in km.Stream(a.pages(0, width)):
            results.push(tile)


def test_prefetch_is_true_or_false_on_a_read():
    with pytest.raises(Refused, match="prefetch= is True or False as written"):
        Unstated(l1=1 << 20, log=print, a=X, out=Y)
    with pytest.raises(Refused, match="a stream has no prefetch="):
        Written(l1=1 << 20, log=print, a=X, out=Y)
