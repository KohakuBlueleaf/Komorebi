"""Cards 0 and 1 as one session's islands (KMR_CARD=1, this file alone): a small class's
command over every rank, across the cards' edge, exact, then a second session in the
same process after the first closed; example 29's chains across the cards, each run as
a command a step and as one chain, exact. See docs/reference/islands.md."""

import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import komorebi as km

pytestmark = [pytest.mark.card, pytest.mark.mesh(cards=(0, 1))]
ACROSS = Path(__file__).resolve().parents[3] / "examples" / "29_chain" / "across.py"


class Put(km.Program):
    """put: rank r of the group adds `value` to word 0 of page r."""

    @km.handler
    def put(self, out: km.Tensor, value: int):
        word = out.pages(self.rank, 1).read()
        word += value
        out.pages(self.rank, 1).write(word)


def one_session(value, kept):
    """A session on cards 0 and 1, kept alive in `kept`: PUT over every rank; its pages
    and each island's kernel error at the stop."""
    with km.Session(cards=[0, 1], log=print) as session:
        ranks, edge = session.ranks, session.graph.first(1)
        out = session.tensor(np.zeros((ranks, 16), np.uint32), "uint32")
        kept.append((session, out))
        run = session.load(Put(out=out, log=print), deadline=2.0)
        run.wait(run.submit(Put.put, (0, ranks), args=(out, value)), timeout=10.0)
        stopped = run.stop()
        got = out.read()
    assert 0 < edge < ranks
    return got, [block["error_name"] for block in stopped.islands]


def test_two_sessions_in_turn_each_exact_across_the_edge():
    kept = []
    for value in (1, 2):
        got, errors = one_session(value, kept)
        assert errors == ["none", "none"]
        assert (got[:, 0] == value).all() and not got[:, 1:].any()


def test_the_chain_across_two_cards_gives_what_its_commands_give():
    saved = sys.argv, list(sys.path)
    sys.argv = [str(ACROSS), "--cards", "0,1", "--runs", "1"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as stop:
            runpy.run_path(str(ACROSS), run_name="__main__")
    finally:
        sys.argv, sys.path[:] = saved
    assert stop.value.code in (None, 0), out.getvalue()[-2000:]
    lines = [line for line in out.getvalue().splitlines() if line.startswith("{")]
    result = json.loads(lines[-1])
    assert result["islands"] == 2 and result["kernel"] == ["none", "none"], result
    for form, calls in (("straddle", 4), ("hops", 3)):
        assert len(result[form]["groups"]) == calls
        for way, bad in result[form]["checks"].items():
            assert not any(bad.values()), (form, way, bad)
