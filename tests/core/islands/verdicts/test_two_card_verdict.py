"""The paged two-card script's verdict: ok says the check rows are finite; it is not an
accuracy bound, and the rows that are not checks are not judged."""

from experiments.language.bounds.page_tiles import two_card

CHECKS = {"qkv": 8.4e-3, "o": 8.5e-3, "gate": 8.5e-3, "down": 8.3e-3}


def test_finite_check_rows_are_ok_whatever_their_size():
    report = {"words": dict(CHECKS), "whole_words": dict(CHECKS)}
    assert two_card.finite(report)
    report["whole_words"]["gate"] = 3.0  # a bad number, but a finite one
    assert two_card.finite(report)


def test_a_nan_or_an_infinity_in_a_check_row_is_not_ok():
    for bad in (float("nan"), float("inf")):
        for key, name in (("words", "o"), ("whole_words", "gate"), ("words", "down")):
            report = {"words": dict(CHECKS), "whole_words": dict(CHECKS)}
            report[key][name] = bad
            assert not two_card.finite(report)


def test_the_rows_that_are_not_checks_are_not_judged():
    report = {"words": {**CHECKS, "gate": 0.977, "up": float("nan")}}
    report["whole_words"] = {**CHECKS, "up": float("nan")}
    assert two_card.finite(report)
