// What the operations' parts share: the K split's hook (SplitBlock) and buffers; on
// compute an element-wise product with DEST at HiFi4 and the norm's 1 / rms.
#pragma once

#include <stdint.h>

#include "../../dense/kernels/split/split_op.hpp"
#include "../decode.h"

namespace decode {

    // The rows that hold a token: a3's DC_ROWS field, 32 when it is 0.
    inline uint32_t live_rows(const km_command_t& command) {
        const uint32_t rows = (command.a3 >> DC_ROWS_SHIFT) & DC_ROWS_MASK;
        return rows == 0u || rows > 32u ? 32u : rows;
    }

}  // namespace decode

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "rows.hpp"

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

    // DEST 0 into one page of `buffer` (reserved and pushed here).
    inline bool pack_one(const km::Compute& compute, uint32_t buffer) {
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(buffer, 1);
        pack_reconfig_data_format(buffer);
        if (room) {
            pack_tile(0, buffer);
            cb_push_back(buffer, 1);
        }
        tile_regs_release();
        return room;
    }

    // DC_CB_SCALE: each live row's 1 / sqrt(mean square + epsilon) in every column,
    // from the squares (DC_CB_PARTS); unpack A is in the fp32 sum's format around it.
    inline bool rms_scale(const km::Compute& compute, uint32_t epsilon, uint32_t live) {
        if (
            !compute.wait_front(DC_CB_PARTS, DC_PARTS) ||
            !compute.wait_front(DC_CB_CONSTANT, DC_CONSTANTS)
        ) {
            return false;
        }
        reconfig_data_format_srca(kCbSum, DC_CB_PARTS);
        tile_regs_acquire();
        copy_init(DC_CB_PARTS);
        copy_tile(DC_CB_PARTS, 0, 0);
        add_reuse_dest_init<kReuse>(DC_CB_PARTS);
        for (uint32_t part = 1; part < DC_PARTS; part++) {
            add_reuse_dest_tiles<kReuse>(DC_CB_PARTS, part, 0);
        }
        if (
            !pack_one(compute, DC_CB_SQUARES) || !compute.wait_front(DC_CB_SQUARES, 1)
        ) {
            return false;
        }
        tile_regs_acquire();
        matmul_init(DC_CB_SQUARES, DC_CB_CONSTANT, 0);
        matmul_tiles(DC_CB_SQUARES, DC_CB_CONSTANT, 0, DC_CONSTANT_MEAN, 0);
        binop_with_scalar_tile_init();
        DC_ROWS(live, rows::plus, 0, epsilon);
        rsqrt_tile_init();
        DC_ROWS(live, rows::rsqrt, 0);
        cb_pop_front(DC_CB_SQUARES, 1);
        const bool scaled = pack_one(compute, DC_CB_SCALE);
        cb_pop_front(DC_CB_PARTS, DC_PARTS);
        cb_pop_front(DC_CB_CONSTANT, DC_CONSTANTS);
        reconfig_data_format_srca(DC_CB_PARTS, kCbSum);
        return scaled && compute.wait_front(DC_CB_SCALE, 1);
    }

}  // namespace decode
#else
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"

namespace decode {

    // Reader: pages `first` to `first + pages - 1` of `tensor` into `buffer` as one
    // item.
    inline bool read_item(
        km::Reader& reader,
        const km_command_t& command,
        uint32_t buffer,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages
    ) {
        const km::Tensor<0> matrix(tensor);
        if (
            !reader.reserve(buffer, pages) ||
            !km::read_pages(
                reader, command, matrix, first, pages, get_write_ptr(buffer)
            ) ||
            !flushed(reader, km::kStReader, km::kFlushReads)
        ) {
            return false;
        }
        cb_push_back(buffer, pages);
        return true;
    }

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
