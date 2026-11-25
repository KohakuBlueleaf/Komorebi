"""Komorebi: a programming model and runtime for the Tensix grid of Blackhole cards.

Importing the package needs no tt-metal tree and opens no device. `Session` is
imported on first use, after preparing the process to run on the tree in use.
"""

import importlib

from komorebi import _env
from komorebi.dsl.classes import Program
from komorebi.dsl.front import front as _front
from komorebi.errors import (
    CardBusy,
    CardNotReady,
    ChannelError,
    ChipsHidden,
    CommandOrderWarning,
    CommandRejected,
    CommandTimeout,
    CrossingWarning,
    DispatcherStackError,
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
    MessagesGaveUp,
    NotServed,
    ProgramError,
    SlowCodeError,
    StackError,
    StartError,
    StopTimeout,
    ThreadFailed,
    WaitTimeout,
)
from komorebi.merge import merge
from komorebi.program.description import Buffer, BufferPair, Handler, KernelSizes
from komorebi.program.movement import Pattern
from komorebi.program.overlay import OverlayError

__version__ = "0.5.3"

__all__ = [
    "Buffer",
    "BufferPair",
    "CardBusy",
    "CardNotReady",
    "ChannelError",
    "ChipsHidden",
    "DispatcherStackError",
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
    "MessagesGaveUp",
    "merge",
    "NotServed",
    "OverlayError",
    "Pattern",
    "Program",
    "ProgramError",
    "CommandOrderWarning",
    "CrossingWarning",
    "CommandRejected",
    "CommandTimeout",
    "Session",
    "SlowCodeError",
    "StackError",
    "StartError",
    "StopTimeout",
    "ThreadFailed",
    "WaitTimeout",
]

# A program written as a class: km.handler, km.L1, km.Stream, ...
# (komorebi.dsl.front.names).
globals().update(_front.exports())
__all__ += sorted(_front.exports())


def __getattr__(name):
    if name == "chain":  # the framework's chain, and @km.chain of a class (chain/)
        return importlib.import_module("komorebi.chain")
    if name == "Session":
        _env.activate()
        return importlib.import_module("komorebi.sessions").Session
    raise AttributeError(f"module 'komorebi' has no attribute {name!r}")
