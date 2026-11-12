"""Timing and error statistics."""

import numpy as np
import pytest

from komorebi.stats import error_stats, timing_stats


def test_timing_stats_of_known_samples():
    stats = timing_stats([4, 1, 100, 2, 3])
    assert stats["n"] == 5 and stats["min"] == 1 and stats["max"] == 100
    assert stats["median"] == 3 and stats["mean"] == 22
    assert stats["p95"] == pytest.approx(80.8)


def test_timing_stats_of_one_sample():
    stats = timing_stats([2.5])
    assert {stats[k] for k in ("min", "median", "mean", "p95", "max")} == {2.5}


def test_timing_stats_needs_a_sample():
    with pytest.raises(ValueError):
        timing_stats([])


def test_exact_result():
    ref = np.random.default_rng(0).standard_normal((64, 64))
    stats = error_stats(ref.copy(), ref)
    assert stats["rel_rms"] == 0 and stats["max_abs"] == 0 and stats["pcc"] > 0.999999


def test_bias_and_scale():
    ref = np.random.default_rng(1).standard_normal((256, 256))
    stats = error_stats(ref + 0.01, ref)

    # Bias and rel-RMS are relative to the reference RMS, to 4 significant digits.
    assert abs(stats["mean_bias"] - 0.01 / stats["ref_rms"]) < 1e-4
    assert abs(stats["rel_rms"] - 0.01 / stats["ref_rms"]) < 1e-4
    assert stats["max_abs"] == pytest.approx(0.01, rel=1e-3)


def test_sign_of_the_bias_and_of_the_correlation():
    ref = np.random.default_rng(2).standard_normal(1000)
    assert error_stats(ref - 0.05, ref)["mean_bias"] < 0
    assert error_stats(-ref, ref)["pcc"] == pytest.approx(-1.0)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        error_stats([], [])


def test_exact_result_on_a_zero_reference(recwarn):
    stats = error_stats(np.zeros(8), np.zeros(8))
    assert stats == {
        "rel_rms": 0.0,
        "max_abs": 0.0,
        "mean_bias": 0.0,
        "pcc": None,
        "ref_rms": 0.0,
    }
    assert not recwarn.list


def test_nonzero_error_against_a_zero_reference_is_infinite(recwarn):
    stats = error_stats(np.array([0.0, 1.0, -1.0]), np.zeros(3))
    assert stats["rel_rms"] == stats["mean_bias"] == np.inf
    assert stats["max_abs"] == 1.0 and stats["ref_rms"] == 0.0
    assert not recwarn.list


@pytest.mark.parametrize("constant", ["result", "reference", "both"])
def test_pcc_is_none_when_a_side_is_constant(constant, recwarn):
    varying = np.array([1.0, 2.0, 4.0])
    flat = np.full(3, 0.1)
    result = flat if constant != "reference" else varying
    reference = flat if constant != "result" else varying
    stats = error_stats(result, reference)
    assert stats["pcc"] is None
    assert np.isfinite(stats["rel_rms"])
    assert not recwarn.list


def test_exact_result_has_no_warnings(recwarn):
    ref = np.random.default_rng(3).standard_normal(100)
    assert error_stats(ref, ref)["pcc"] == 1.0
    assert not recwarn.list
