"""Komorebi: a programming model and runtime for the Tensix grid of Blackhole cards.

Importing the package needs no tt-metal tree and opens no device. `Session` (and with
it ttnn) is imported on first use, after activating the runtime package if it is used.
"""

import importlib

from komorebi import _env
from komorebi.errors import (
    CardBusy,
    EngineError,
    EngineRefused,
    EngineTimeout,
    GridTimeout,
    KernelFailed,
    KernelInstructionError,
    KernelSizeError,
    KomorebiError,
    ProgramError,
    RecordRefused,
    RecordTimeout,
    StartError,
    ThreadFailed,
    WaitTimeout,
)
from komorebi.program import Buffer, KernelSizes, Program

__version__ = "0.1.0"

__all__ = [
    "Buffer",
    "CardBusy",
    "EngineError",
    "EngineRefused",
    "EngineTimeout",
    "GridTimeout",
    "KernelFailed",
    "KernelInstructionError",
    "KernelSizeError",
    "KernelSizes",
    "KomorebiError",
    "Program",
    "ProgramError",
    "RecordRefused",
    "RecordTimeout",
    "Session",
    "StartError",
    "ThreadFailed",
    "WaitTimeout",
]


def __getattr__(name):
    if name == "Session":
        _env.activate()
        return importlib.import_module("komorebi.session").Session
    raise AttributeError(f"module 'komorebi' has no attribute {name!r}")
