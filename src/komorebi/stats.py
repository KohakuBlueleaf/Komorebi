"""Statistics helpers: timing samples and error against a reference.

Statistics are reported, not judged.
"""

import numpy as np


def timing_stats(samples):
    """n, min, median, mean, p95 and max of a non-empty sequence of timings."""
    values = np.asarray(samples, dtype=np.float64).ravel()
    if not values.size:
        raise ValueError("timing_stats needs at least one sample")
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def error_stats(result, reference):
    """rel_rms, max_abs, mean_bias, pcc and ref_rms of result against reference.

    rel_rms and mean_bias (mean signed error) are relative to the reference RMS, to 4
    significant digits; pcc, to 6 decimals, is None when either side is constant.
    """
    result = np.asarray(result, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if not result.size:
        raise ValueError("error_stats needs at least one value")
    difference = result - reference
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    error_rms = float(np.sqrt(np.mean(difference**2)))
    bias = float(difference.mean())

    # Against a zero reference the error is 0.0 when exact and inf otherwise.
    if ref_rms == 0:
        rel_rms, mean_bias = (0.0, 0.0) if error_rms == 0 else (np.inf, np.inf)
    else:
        rel_rms, mean_bias = error_rms / ref_rms, bias / ref_rms

    # Pearson correlation of the centred vectors; undefined for a constant vector.
    pcc = None
    if (result != result[0]).any() and (reference != reference[0]).any():
        centered_r = result - result.mean()
        centered_ref = reference - reference.mean()
        pcc = round(
            float(
                centered_r
                @ centered_ref
                / (np.linalg.norm(centered_r) * np.linalg.norm(centered_ref))
            ),
            6,
        )

    return {
        "rel_rms": float(f"{rel_rms:.3e}"),
        "max_abs": float(f"{np.abs(difference).max():.3e}"),
        "mean_bias": float(f"{mean_bias:.3e}"),
        "pcc": pcc,
        "ref_rms": float(f"{ref_rms:.3e}"),
    }
