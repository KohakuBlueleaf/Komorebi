"""A row broadcast of a 16-bit tile into an fp32 Dst on Wormhole (no card): through
km_worker.hpp the kernels reach core/km_bcast.hpp, whose unary_bcast<ROW> for an fp32
Dst sets SrcA to tf32 around the SrcB moves and back, on Wormhole alone (tile.md, On
Wormhole); the tile compiler's text is the same on both chips."""

import re
from pathlib import Path

from komorebi.dsl.compile.tiles import functions

ROOT = Path(__file__).resolve().parents[3]
DEVICE = ROOT / "src/komorebi_native/include/komorebi/device"
HEADER = DEVICE / "core/km_bcast.hpp"
SRCA = "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_"
COPY = "llk_math_eltwise_unary_datacopy<DataCopyType::"


def code(path):
    """The lines of `path` without comments, stripped."""
    lines = (line.split("//")[0].strip() for line in path.read_text().split("\n"))
    return [line for line in lines if line]


def joined(path):
    """The code of `path` on one line, no space inside its brackets."""
    text = " ".join(code(path))
    return re.sub(r"\s+([)>])", r"\1", re.sub(r"([<(])\s+", r"\1", text))


def test_the_compute_kernels_include_the_header():
    assert '#include "core/km_bcast.hpp"' in code(DEVICE / "km_worker.hpp")


def test_the_specialization_is_wormholes_alone_and_an_fp32_dsts():
    lines = code(HEADER)
    guard = "#if defined(COMPILE_FOR_TRISC) && defined(ARCH_WORMHOLE)"
    assert lines[:2] == ["#pragma once", guard] and lines[-1] == "#endif"
    assert "template <>" in lines
    assert "ALWI void unary_bcast<BroadcastType::ROW, true>(" in lines


def test_srca_is_tf32_around_the_srcb_moves_and_back_after():
    text = joined(HEADER)
    stall = "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);"
    order = [
        "EltwiseBinaryReuseDestType::NONE, false>(",  # the tile unpacked to SrcB
        stall,
        f"{SRCA}val_RMW>(static_cast<uint32_t>(DataFormat::Tf32));",
        f"{SRCA}override_RMW>(1);",
        f"{COPY}B2D, true, BroadcastType::ROW, false>(",
        stall,
        f"{SRCA}val_RMW>(static_cast<uint32_t>(DataFormat::Float32));",
        f"{SRCA}override_RMW>(0);",
    ]
    at = 0
    for words in order:
        at = text.index(words, at) + len(words)
    # the 32-bit tiles (unpacked to Dst) keep the library's own moves, no format change
    to_dest = text[: text.index("} else {")]
    assert f"{COPY}A2D, true, BroadcastType::ROW, true>(" in to_dest
    assert SRCA not in to_dest


def test_the_tile_compiler_calls_the_library_row_broadcast():
    lines = functions.row_broadcast(None, "scale", None, 1, [], "ROW")
    assert "unary_bcast<BroadcastType::ROW>(kFieldScale, 0, 1);" in lines
