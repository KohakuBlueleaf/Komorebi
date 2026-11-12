"""Per-index hash random numbers, identical in numpy, C++ and CUDA (rng.hpp)."""

import numpy as np

GOLDEN = np.uint32(0x9E3779B1)
SCALE = np.float32(1.0 / 16777216.0)


def hash32(index, stream=0, seed=0):
    """A 32-bit hash of (index, stream, seed); index may be an array of integers."""
    with np.errstate(over="ignore"):
        h = np.asarray(index).astype(np.uint32) * GOLDEN
        h = h + np.uint32(((stream + 1) * 0x85EBCA6B) & 0xFFFFFFFF)
        h = h + np.uint32(seed & 0xFFFFFFFF)
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(0x7FEB352D)
        h = h ^ (h >> np.uint32(15))
        h = h * np.uint32(0x846CA68B)
        return h ^ (h >> np.uint32(16))


def uniform(index, stream=0, seed=0):
    """Uniform float32 in [0, 1) with 24 random bits, from `hash32`."""
    h = hash32(index, stream, seed)
    return (h >> np.uint32(8)).astype(np.float32) * SCALE
