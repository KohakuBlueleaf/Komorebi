"""Komorebi: a programming model and runtime for the Tensix grid of Blackhole cards.

Importing the package needs no tt-metal tree and opens no device. `Session` is
imported on first use, after preparing the process to run on the tree in use.
"""

import importlib

from komorebi import _env
from komorebi.dsl import front as _front
from komorebi.dsl.classes import Program
from komorebi.errors import (
    CardBusy,
    ChannelError,
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
    L1LayoutError,
    ProgramError,
    SlowCodeError,
    StartError,
    ThreadFailed,
    WaitTimeout,
)
from komorebi.merge import merge
from komorebi.program.description import Buffer, BufferPair, Handler, KernelSizes
from komorebi.program.movement import Pattern
from komorebi.program.overlay import OverlayError

__version__ = "0.2.0"

__all__ = [
    "Buffer",
    "BufferPair",
    "CardBusy",
    "ChannelError",
    "EngineError",
    "EngineRejected",
    "EngineTimeout",
    "GridTimeout",
    "Handler",
    "KernelFailed",
    "KernelInstructionError",
    "KernelSizeError",
    "KernelSizes",
    "KomorebiError",
    "L1BudgetError",
    "L1LayoutError",
    "merge",
    "OverlayError",
    "Pattern",
    "Program",
    "ProgramError",
    "CommandRejected",
    "CommandTimeout",
    "Session",
    "SlowCodeError",
    "StartError",
    "ThreadFailed",
    "WaitTimeout",
]

# A program written as a class: km.handler, km.L1, km.Stream, ... (komorebi.dsl.names).
globals().update(_front.exports())
__all__ += sorted(_front.exports())


def __getattr__(name):
    if name == "chain":  # the framework's chain, and @km.chain of a class (chain/)
        return importlib.import_module("komorebi.chain")
    if name == "Session":
        _env.activate()
        return importlib.import_module("komorebi.sessions").Session
    raise AttributeError(f"module 'komorebi' has no attribute {name!r}")
