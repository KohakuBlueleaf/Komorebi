"""The session on one card: the session, a run, the state words and page bounds. Its
names come from komorebi.session.session on first use: importing a run or the state
words does not load the session."""

import importlib

__all__ = [
    "WORDS_SPIN_S",
    "Session",
    "compute_config",
    "kernel_sizes",
    "read_tensors",
]


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module("komorebi.session.session"), name)
    raise AttributeError(f"module 'komorebi.session' has no attribute {name!r}")
