"""Komorebi: a programming model and runtime for the Tensix grid of Blackhole cards.

Importing the package needs no tt-metal tree and opens no device. `Session` is
imported on first use, after preparing the process to run on the tree in use.
"""

import importlib

from komorebi import _env
from komorebi.errors import (
    CardBusy,
    CommandRejected,
    CommandTimeout,
    EngineError,
    EngineRejected,
    EngineTimeout,
    GridTimeout,
    KernelFailed,
    KernelInstructionError,
    KernelSizeError,
    KomorebiError,
    L1BudgetError,
    ProgramError,
    StartError,
    ThreadFailed,
    WaitTimeout,
)
from komorebi.movement import Pattern
from komorebi.overlay import OverlayError
from komorebi.program import Buffer, BufferPair, KernelSizes, Program

__version__ = "0.1.0"

__all__ = [
    "Buffer",
    "BufferPair",
    "CardBusy",
    "EngineError",
    "EngineRejected",
    "EngineTimeout",
    "GridTimeout",
    "KernelFailed",
    "KernelInstructionError",
    "KernelSizeError",
    "KernelSizes",
    "KomorebiError",
    "L1BudgetError",
    "OverlayError",
    "Pattern",
    "Program",
    "ProgramError",
    "CommandRejected",
    "CommandTimeout",
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
