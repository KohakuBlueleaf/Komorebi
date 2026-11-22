//! The matrix unit's words of a tile product (today's `komorebi.dsl.tile.matrix`): the
//! library's Blackhole no-MOP matmul (`matmul_block_no_mop`) expanded, one replay of 16
//! MVMUL a fidelity phase in the address-mode order of `mm_no_mop_init_short`, writing
//! the Dst tile's 8-row blocks 0 .. 7, then 0 .. 7 again (two words on a block 8
//! apart), and Wormhole's own image. In0 to SrcB, in1 to SrcA.
//!
//! Not ported: `block_cost` and `product_program`, which the unit program's path does
//! not call.

use super::unit::{Arg, Insn, UnitError};

/// MathFidelity: the number of phases (the library's enum value; LoFi replays once).
pub const PHASES: [(&str, i64); 4] = [("LoFi", 0), ("HiFi2", 2), ("HiFi3", 3), ("HiFi4", 4)];
/// ADDR_MOD 0: SrcB + 8, Dst + 8; 1: SrcA + 16, SrcB back, Dst + 8; 2: SrcA back, SrcB
/// + 32, Dst + 8; 4: SrcA + 32, SrcB + 48, Dst back; 5: all back, the phase + 1.
pub const REPLAY_MODES: [i64; 16] = [0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 5];
/// Wormhole: tt-llk's own image (matmul_emit_replay_program_no_mop), in the modes 0 to
/// 3 that its init sets there (matmul_configure_addrmod; 3 is mode 0's steps).
pub const WORMHOLE_MODES: [i64; 16] = [0, 1, 0, 2, 0, 1, 3, 0, 0, 1, 0, 2, 0, 1, 3, 1];
/// 8-row Dst blocks in a 32x32 tile.
pub const BLOCKS: i64 = 8;
/// p_setrwc: the Src banks a word releases (none, A, B).
pub const CLR_NONE: i64 = 0;
/// p_setrwc: SrcA released.
pub const CLR_A: i64 = 1;
/// p_setrwc: SrcB released.
pub const CLR_B: i64 = 2;
/// p_setrwc: SrcA, SrcB, Dst and fidelity counters back to 0.
pub const SET_ABD_F: i64 = 0xF;
/// p_setrwc: SrcA, SrcB and Dst counters back to 0.
pub const SET_ABD: i64 = 0x7;

/// A matrix word: (op, its Dst block; none for a counter's step) and its instruction.
pub type Word = ((String, Option<i64>), Insn);

fn mvmul(mode: i64, clear: i64, block: i64, note: &str) -> Result<Word, UnitError> {
    let args = [
        ("clear_dvalid", Arg::Num(clear)),
        ("instr_mod19", Arg::Num(0)),
        ("addr_mode", Arg::Num(mode)),
        ("dst", Arg::Num(0)),
    ];
    Ok((
        ("MVMUL".into(), Some(block)),
        Insn::new("MVMUL", &args, note)?,
    ))
}

fn setrwc(clear: i64, note: &str, mask: i64) -> Result<Word, UnitError> {
    let args = [
        ("clear_ab_vld", Arg::Num(clear)),
        ("rwc_cr", Arg::Num(0)),
        ("rwc_d", Arg::Num(0)),
        ("rwc_b", Arg::Num(0)),
        ("rwc_a", Arg::Num(0)),
        ("BitMask", Arg::Num(mask)),
    ];
    Ok((("SETRWC".into(), None), Insn::new("SETRWC", &args, note)?))
}

/// The phases of `fidelity`, or today's ValueError's text.
pub fn phases(fidelity: &str) -> Result<i64, UnitError> {
    PHASES
        .iter()
        .find(|(f, _)| *f == fidelity)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            let names: Vec<&str> = PHASES.iter().map(|(f, _)| *f).collect();
            UnitError(format!(
                "no fidelity {}: one of {}",
                super::py::str_repr(fidelity),
                names.join(", ")
            ))
        })
}

/// The words of one `matmul_block_no_mop(..., dst_index, 0, ct_dim, rt_dim, kt_dim)`
/// call (one K tile: kt_dim only steps the operands' indices), in issue order. The Dst
/// base of each tile is set by the RISC-V (`set_dst_write_addr`) before its words; a
/// word's block is 8 Dst tile + its block in the tile.
pub fn tile_products(
    fidelity: &str,
    ct_dim: i64,
    rt_dim: i64,
    dst_index: i64,
) -> Result<Vec<Word>, UnitError> {
    let phases = phases(fidelity)?;
    let reuse_a = ct_dim >= rt_dim;
    let (t_dim, rut_dim) = if reuse_a {
        (rt_dim, ct_dim)
    } else {
        (ct_dim, rt_dim)
    };
    let mut out = Vec::new();
    for t in 0..t_dim {
        for rut in 0..rut_dim {
            let tile = dst_index
                + if reuse_a {
                    ct_dim * t + rut
                } else {
                    t + rut * ct_dim
                };
            for phase in 0..phases.max(1) {
                for (i, &mode) in REPLAY_MODES.iter().enumerate() {
                    let mut clear = CLR_NONE;
                    if phases == 0 && i == REPLAY_MODES.len() - 1 {
                        clear = if reuse_a { CLR_A } else { CLR_B };
                    }
                    let i = i as i64;
                    let block = BLOCKS * tile + i % BLOCKS;
                    let note = format!("tile {tile}, phase {phase}, block {}", i % BLOCKS);
                    out.push(mvmul(mode, clear, block, &note)?);
                }
            }
            if phases != 0 {
                let clear = if reuse_a { CLR_A } else { CLR_B };
                out.push(setrwc(clear, "after the phases: releases", SET_ABD_F)?);
            }
            if rut == rut_dim - 1 {
                let clear = if reuse_a { CLR_B } else { CLR_A };
                out.push(setrwc(clear, "end of the reuse row", SET_ABD_F)?);
            }
        }
    }
    Ok(out)
}

/// The words of one Wormhole `matmul_block_no_mop(..., dst_index, 0, 1, 1, 1)`
/// (tt-llk's `_llk_math_matmul_no_mop_`, one tile, SrcA reused): its 16-word image a
/// fidelity phase, the replays expanded, then its counter resets.
pub fn wormhole_products(fidelity: &str, dst_index: i64) -> Result<Vec<Word>, UnitError> {
    let phases = phases(fidelity)?;
    let mut out = Vec::new();
    for phase in 0..phases.max(1) {
        for (i, &mode) in WORMHOLE_MODES.iter().enumerate() {
            let last = phases == 0 && i == WORMHOLE_MODES.len() - 1;
            let i = i as i64;
            let block = BLOCKS * dst_index + i % BLOCKS;
            let note = format!(
                "tile {dst_index}, phase {phase}, block {} (Wormhole)",
                i % BLOCKS
            );
            let clear = if last { CLR_A } else { CLR_NONE };
            out.push(mvmul(mode, clear, block, &note)?);
        }
    }
    if phases != 0 {
        out.push(setrwc(CLR_A, "after the phases: releases", SET_ABD_F)?);
    }
    out.push(setrwc(CLR_B, "end of the reuse row", SET_ABD)?);
    Ok(out)
}
