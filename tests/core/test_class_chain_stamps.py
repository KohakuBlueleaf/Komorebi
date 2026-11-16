"""A class's chain stamps (chain_stamps=True): KM_CHAIN_STAMPS and the read-out in its
kernels, refused without the walk, each record's stamp row, and the terms read from
rows (no card)."""

import numpy as np
import pytest
from chain_fixture import CORES, WIDTH, Handlers, Steps, bits, quiet, run_of, tensors

from komorebi import chain as kc
from komorebi.dsl import chain_stamps
from komorebi.errors import ProgramError

GIVEN = tensors()
X, Y, Z, S, T = (GIVEN[name] for name in "xyzst")


def made(cls, **given):
    return cls(l1=1 << 20, log=quiet, a=X, b=X, out=Y, s=S, t=T, **given)


def test_the_stamps_are_built_into_a_chain_build_and_change_nothing_else():
    plain, stamped = made(Steps), made(Steps, chain_stamps=True)
    assert stamped.defines["KM_CHAIN_STAMPS"] == "1" and stamped.chain_stamps
    assert "KM_CHAIN_STAMPS" not in (plain.defines or {})
    text = (stamped.built.folder / "movement.cpp").read_text()
    assert "km::chain_stamps_out(movement, command.a0, 2048u)" in text
    assert "km::chain_stamp(3);  // the record's body done" in text
    compute = (stamped.built.folder / "compute.cpp").read_text()
    assert "if (command.a7 != 0)  // 0: the stamps' read-out" in compute
    assert "chain_stamp" not in (plain.built.folder / "movement.cpp").read_text()
    assert stamped.built.folder != plain.built.folder


def test_stamps_without_the_walk_are_refused():
    with pytest.raises(ProgramError, match=r"chain_stamps=True stamps the chain's"):
        made(Handlers, chain_stamps=True)


def test_each_record_takes_its_index_as_its_stamp_row():
    w = kc.words()
    run = run_of(made(Steps, chain_stamps=True))
    calls = [Steps.scale(X, Y, bits(2), WIDTH)] * 9
    chain = run.chain(calls, cores=(0, CORES))
    assert [call.row for call in chain.calls] == [*range(8), w["kChainRowMask"]]
    plain = run_of(made(Steps)).chain(calls[:2], cores=(0, CORES))
    assert [call.row for call in plain.calls] == [0, 0]


def test_the_terms_of_a_record_from_its_rows():
    w = kc.words()
    rows = np.zeros((2, 3, 8, 16), np.uint32)  # two cores, record 0, 2 runs each
    at = {"kChainStampTaken": 100, "kChainStampMarked": 300, "kChainStampLast": 1500}
    for core in range(2):
        writer = rows[core, chain_stamps.WRITER, 0]
        writer[0] = 2
        for name, cycles in at.items():
            writer[w[name]] = 2 * cycles * (core + 1)
        writer[3] = 2 * 1000 * (core + 1)  # the body's end
        writer[w["kChainStampMarks"]] = 2 * 1600 * (core + 1)
    record, empty = chain_stamps.terms(rows, 100.0, 2)  # 100 MHz: 100 cycles a us
    assert record["cores"] == 2 and empty == {}
    assert record["input wait"] == (3.0, 4.0)  # 2 us and 4 us
    assert record["body"] == (10.5, 14.0)
    assert record["hand-over"] == (7.5, 10.0)
    assert record["marks"] == (1.5, 2.0)
    assert record["overlay load"] == (0.0, 0.0)
