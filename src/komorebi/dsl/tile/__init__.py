"""The tile layer (experimental): a tile expression traced into the tile IR
(`trace`, `ir`), its vector chain (`chain`, `sfpu`), the unit program a tile on one
core (`unit`, `lower`) and its C++ compute body (`emit`). Nothing imports it yet.
"""

from komorebi.dsl.tile.emit import emit
from komorebi.dsl.tile.lower import stream_chain, unit_program
from komorebi.dsl.tile.rows import row_operand
from komorebi.dsl.tile.unit import UnitError, UnitProgram

__all__ = [
    "UnitError",
    "UnitProgram",
    "emit",
    "row_operand",
    "stream_chain",
    "unit_program",
]
