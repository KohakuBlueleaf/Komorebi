// A row broadcast of a 16-bit tile into an fp32 Dst on Wormhole (compute): the
// library's unary_bcast<ROW> with SrcA's format set to tf32 around its moves.
#pragma once

#if defined(COMPILE_FOR_TRISC) && defined(ARCH_WORMHOLE)
#include <stdint.h>

#include "api/compute/bcast.h"

namespace ckernel {

    // Wormhole's MOVB2D writes whole 32-bit Dst words only under a 32-bit SrcA format;
    // the library sets tf32 for UInt16 alone (tile.md, On Wormhole).
    template <>
    ALWI void unary_bcast<BroadcastType::ROW, true>(
        uint32_t icb, uint32_t in_tile_index, uint32_t dst_tile_index
    ) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
        const uint32_t dst_format = get_operand_dst_format(icb);
        const bool enable_unpack_to_dest = (
            (dst_format == static_cast<uint32_t>(DataFormat::Float32)) ||
            (dst_format == static_cast<uint32_t>(DataFormat::UInt32)) ||
            (dst_format == static_cast<uint32_t>(DataFormat::Int32))
        );
        if (enable_unpack_to_dest) {
            UNPACK(
                (
                    llk_unpack_A<
                        BroadcastType::ROW,
                        false,
                        EltwiseBinaryReuseDestType::NONE,
                        true>(icb, in_tile_index)
                )
            );
            MATH(
                (
                    llk_math_eltwise_unary_datacopy<
                        DataCopyType::A2D,
                        true,
                        BroadcastType::ROW,
                        true>(dst_tile_index, icb)
                )
            );
        } else {
            UNPACK(
                (
                    llk_unpack_A<
                        BroadcastType::ROW,
                        false,
                        EltwiseBinaryReuseDestType::NONE,
                        false>(icb, in_tile_index)
                )
            );
#if defined(TRISC_MATH)
            // the earlier moves end before the format changes; it is set back, as the
            // library leaves it, after these end
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(
                static_cast<uint32_t>(DataFormat::Tf32)
            );
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(1);
            llk_math_eltwise_unary_datacopy<
                DataCopyType::B2D,
                true,
                BroadcastType::ROW,
                false>(dst_tile_index, icb);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(
                static_cast<uint32_t>(DataFormat::Float32)
            );
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(0);
#endif
        }
#endif
    }

}  // namespace ckernel
#endif
