"""The matrix unit's words of a tile product: the library's Blackhole no-MOP matmul
(`matmul_block_no_mop`) expanded, one replay of 16 MVMUL a fidelity phase in the
address-mode order of `mm_no_mop_init_short`, writing the Dst tile's 8-row blocks
0 .. 7, then 0 .. 7 again (two words on a block 8 apart). In0 to SrcB, in1 to SrcA."""

from komorebi.dsl.tile.unit import Call, Insn, Product, UnitProgram, matrix_cost

# MathFidelity: the number of phases (the library's enum value; LoFi replays once)
PHASES = {"LoFi": 0, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
# ADDR_MOD 0: SrcB + 8, Dst + 8; 1: SrcA + 16, SrcB back, Dst + 8; 2: SrcA back, SrcB
# + 32, Dst + 8; 4: SrcA + 32, SrcB + 48, Dst back; 5: all back, the phase + 1
REPLAY_MODES = (0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 5)
# Wormhole: tt-llk's own image (matmul_emit_replay_program_no_mop), in the modes 0 to 3
# that its init sets there (matmul_configure_addrmod; 3 is mode 0's steps)
WORMHOLE_MODES = (0, 1, 0, 2, 0, 1, 3, 0, 0, 1, 0, 2, 0, 1, 3, 1)
BLOCKS = 8  # 8-row Dst blocks in a 32x32 tile
CLR_NONE, CLR_A, CLR_B = 0, 1, 2  # p_setrwc: the Src banks a word releases
SET_ABD_F = 0xF  # p_setrwc: SrcA, SrcB, Dst and fidelity counters back to 0
SET_ABD = 0x7  # p_setrwc: SrcA, SrcB and Dst counters back to 0


def _mvmul(mode, clear, block, note):
    args = {"clear_dvalid": clear, "instr_mod19": 0, "addr_mode": mode, "dst": 0}
    return ("MVMUL", block), Insn("MVMUL", args, note)


def _setrwc(clear, note, mask=SET_ABD_F):
    args = {
        "clear_ab_vld": clear,
        "rwc_cr": 0,
        "rwc_d": 0,
        "rwc_b": 0,
        "rwc_a": 0,
        "BitMask": mask,
    }
    return ("SETRWC", None), Insn("SETRWC", args, note)


def tile_products(fidelity, ct_dim, rt_dim, dst_index):
    """[((op, Dst block), Insn)] of one `matmul_block_no_mop(..., dst_index, 0, ct_dim,
    rt_dim, kt_dim)` call (one K tile: kt_dim only steps the operands' indices), in
    issue order. The Dst base of each tile is set by the RISC-V (`set_dst_write_addr`)
    before its words; a word's block is 8 Dst tile + its block in the tile."""
    if fidelity not in PHASES:
        raise ValueError(f"no fidelity {fidelity!r}: one of {', '.join(PHASES)}")
    phases = PHASES[fidelity]
    reuse_a = ct_dim >= rt_dim
    t_dim, rut_dim = (rt_dim, ct_dim) if reuse_a else (ct_dim, rt_dim)
    out = []
    for t in range(t_dim):
        for rut in range(rut_dim):
            tile = dst_index + (ct_dim * t + rut if reuse_a else t + rut * ct_dim)
            for phase in range(max(phases, 1)):
                for i, mode in enumerate(REPLAY_MODES):
                    clear = CLR_NONE
                    if phases == 0 and i == len(REPLAY_MODES) - 1:
                        clear = CLR_A if reuse_a else CLR_B
                    block = BLOCKS * tile + i % BLOCKS
                    note = f"tile {tile}, phase {phase}, block {i % BLOCKS}"
                    out.append(_mvmul(mode, clear, block, note))
            if phases:
                out.append(
                    _setrwc(CLR_A if reuse_a else CLR_B, "after the phases: releases")
                )
            if rut == rut_dim - 1:
                out.append(_setrwc(CLR_B if reuse_a else CLR_A, "end of the reuse row"))
    return out


def wormhole_products(fidelity, dst_index):
    """[((op, Dst block), Insn)] of one Wormhole `matmul_block_no_mop(..., dst_index,
    0, 1, 1, 1)` (tt-llk's `_llk_math_matmul_no_mop_`, one tile, SrcA reused): its
    16-word image a fidelity phase, the replays expanded, then its counter resets."""
    if fidelity not in PHASES:
        raise ValueError(f"no fidelity {fidelity!r}: one of {', '.join(PHASES)}")
    phases, out = PHASES[fidelity], []
    for phase in range(max(phases, 1)):
        for i, mode in enumerate(WORMHOLE_MODES):
            last = phases == 0 and i == len(WORMHOLE_MODES) - 1
            block = BLOCKS * dst_index + i % BLOCKS
            note = f"tile {dst_index}, phase {phase}, block {i % BLOCKS} (Wormhole)"
            out.append(_mvmul(mode, CLR_A if last else CLR_NONE, block, note))
    if phases:
        out.append(_setrwc(CLR_A, "after the phases: releases"))
    out.append(_setrwc(CLR_B, "end of the reuse row", SET_ABD))
    return out


def block_cost(fidelity, ct_dim, rt_dim, kt_dim=1):
    """(cycles, words of one call) of `kt_dim` K steps of the library's
    `matmul_block(.., ct_dim, rt_dim, kt_dim)`, a call a K tile adding into the same Dst
    tiles. The MOP form issues the same words as `matmul_block_no_mop`: it replays the
    same 16-word image a fidelity phase and ends with the same SETRWC."""
    words = [w for w, _ in tile_products(fidelity, ct_dim, rt_dim, 0)]
    return matrix_cost(words * kt_dim), words


def product_program(name, fidelity):
    """The unit program of one K step of `matmul_block_no_mop(in0, in1, i0, i1, dst,
    false, 1, 1, kt_dim)` with the math CPU's words emitted; the library's unpack
    (`llk_unpack_AB_matmul`) and init (`mm_no_mop_init_short`, the address modes) are
    the caller's."""
    pairs = tile_products(fidelity, 1, 1, 0)
    unit = UnitProgram(name, "fp32")
    text = f"the no-MOP product's words at {fidelity}"
    product = Product(
        text,
        [w for w, _ in pairs],
        "1 cycle a word; a block's two words 8 apart",
        insns=[insn for _, insn in pairs],
    )
    unit.tile["math"] = [
        Call(
            "ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, "
            "UnpackDestination::SrcRegs>(dst)",
            "Dst base of tile dst (RISC-V writes)",
        ),
        product,
    ]
    return unit
