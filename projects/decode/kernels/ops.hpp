// What the operations' parts share: the K split's hook (SplitBlock) and buffers, and
// on compute an element-wise product with DEST at HiFi4.
#pragma once

#include <stdint.h>

#include "../../dense/kernels/split/split_op.hpp"
#include "../decode.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"

namespace decode {

    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr auto kMul = EltwiseBinaryType::ELWMUL;

    // DEST tile `dst` times tile `tile` of `buffer`, at HiFi4 whatever the program's
    // fidelity (the dense project's SwiGLU does the same).
    inline void multiply_init(uint32_t buffer) {
        state_configure(buffer, __LINE__);
        UNPACK(
            (llk_unpack_A_init<BroadcastType::NONE, true, kReuse>(false, false, buffer))
        );
        MATH(
            (
                llk_math_eltwise_binary_init<
                    kMul,
                    BroadcastType::NONE,
                    MathFidelity::HiFi4,
                    kReuse>(buffer, buffer, false)
            )
        );
    }

    inline void multiply(uint32_t buffer, uint32_t tile, uint32_t dst) {
        UNPACK((llk_unpack_A<BroadcastType::NONE, true, kReuse>(buffer, tile)));
        MATH(
            (
                llk_math_eltwise_binary<
                    kMul,
                    BroadcastType::NONE,
                    DST_ACCUM_MODE,
                    MathFidelity::HiFi4,
                    kReuse>(buffer, buffer, dst, true)
            )
        );
    }

}  // namespace decode
#else
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"

namespace decode {

    // Reader: the group's block of `tensor` (bf16 tiles, as wide as Y) into DC_CB_BLOCK
    // as one item of oh x ow tiles. False: gave up.
    inline bool read_block(
        km::Reader& reader, const SplitBlock& block, uint32_t tensor
    ) {
        const km_command_t& command = *block.command;
        const km::Tensor<0> matrix(tensor);
        const uint32_t n_tiles = reader.arg(block.plan + kSpNt);
        const uint32_t block_width = reader.arg(block.plan + kSpOw);
        const uint32_t pages = reader.arg(block.plan + kSpOh) * block_width;
        if (!reader.reserve(DC_CB_BLOCK, pages)) {
            return false;
        }
        const uint32_t to = get_write_ptr(DC_CB_BLOCK);
        for (uint32_t row = 0; row < block.rows; row++) {
            const uint32_t first = (
                (block.first_row + row) * n_tiles + block.first_column
            );
            if (!km::read_pages(
                reader,
                command,
                matrix,
                first,
                block.columns,
                to + row * block_width * kTileBytes
            )) {
                return false;
            }
        }
        if (!flushed(reader, km::kStReader, km::kFlushReads)) {
            return false;
        }
        cb_push_back(DC_CB_BLOCK, pages);
        return true;
    }

}  // namespace decode
#endif
