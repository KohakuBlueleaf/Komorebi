"""The acceptance script's verdict: the words of a program that ran decide an ok, from
a saved one-island run or from the twin program of the same run; a refusal is a result
that is not judged, and a run with nothing to compare says no ok."""

import numpy as np

from experiments.islands.acceptance import accept


def words(values, kind="u2"):
    return np.array(values, dtype=kind).view(np.uint8)


def report(**programs):
    return {"programs": programs}


def test_the_islands_program_is_judged_against_its_twin_in_the_same_run():
    same = {
        "swiglu:hidden": words([1, 2, 3]),
        "swiglu_islands:hidden": words([1, 2, 3]),
    }
    done = report(swiglu={"status": "ran"}, swiglu_islands={"status": "ran"})
    accept.verdict(done, same)
    assert done["ok"] is True and done["compared"] == 1
    other = {
        "swiglu:hidden": words([1, 2, 3]),
        "swiglu_islands:hidden": words([1, 2, 4]),
    }
    done = report(swiglu={"status": "ran"}, swiglu_islands={"status": "ran"})
    accept.verdict(done, other)
    assert done["ok"] is False
    assert done["alike"]["swiglu_islands"]["hidden"] == {"differ": 1, "words": 3}


def test_a_saved_run_judges_the_words_compared_with_it():
    record = {"status": "ran", "words": {"o": {"differ": 0, "words": 8}}}
    done = report(attention=record)
    accept.verdict(done, {})
    assert done["ok"] is True and done["compared"] == 1
    record = {"status": "ran", "words": {"o": {"differ": 366, "words": 65536}}}
    done = report(attention=record)
    accept.verdict(done, {})
    assert done["ok"] is False


def test_a_refusal_and_a_run_with_nothing_to_compare_make_no_verdict():
    refused = {"status": "refused or failed", "error": "ValueError: group (0, 120)"}
    done = report(attention=refused, decode={"status": "ran", "words": {}})
    accept.verdict(done, {"decode:chain_o": words([1])})
    assert "ok" not in done and done["compared"] == 0
    missing = {"status": "ran", "words": {"o": "no one-island output"}}
    done = report(attention=missing)
    accept.verdict(done, {})
    assert "ok" not in done


def test_the_float32_output_is_compared_by_its_32_bit_words():
    first = {
        "swiglu:y": words([1.0, 2.0], "f4"),
        "swiglu_islands:y": words([1.0, 2.0], "f4"),
    }
    done = report(swiglu={"status": "ran"}, swiglu_islands={"status": "ran"})
    accept.verdict(done, first)
    assert done["alike"]["swiglu_islands"]["y"] == {"differ": 0, "words": 2}
