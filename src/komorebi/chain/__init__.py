"""The chain on the host: a call's description and the records it becomes."""

import sys

from komorebi.chain import calls
from komorebi.chain.calls import (
    Call,
    ChainRefused,
    Range,
    Read,
    blocks_met,
    build,
    tiles_of,
    words,
)

__all__ = [
    "Call",
    "ChainRefused",
    "Range",
    "Read",
    "blocks_met",
    "build",
    "tiles_of",
    "words",
]

sys.modules[__name__].__class__ = calls._Spelled  # `@km.chain` beside `km.chain.Call`
