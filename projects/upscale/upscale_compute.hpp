// The upscaler's compute bodies: a convolution's tile as ten products, and the last
// layer's pixel shuffle as products by 0/1 tiles (README.md, "The handler").
#pragma once

#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "komorebi/device/core/km_core.hpp"

namespace upscale {

    constexpr uint32_t kRowTiles = 16;     // an image row: 512 pixels
    constexpr uint32_t kLayerTiles = 10;   // a layer's weights: nine taps and the bias
    constexpr uint32_t kOne = 0x3F800000;  // 1.0f: Hardtanh's top
    constexpr uint32_t kShuffle = 70;  // the shuffle's 0/1 tiles: 10 of P, then 4 of Q
    constexpr uint32_t kRows = 5;      // a core's image rows

    // The product set-up this CPU last made (0 none, 1 a read transposed, 2 as it is).
    inline uint32_t setup_made = 0;

    inline void setup(uint32_t weights, uint32_t vectors, uint32_t form) {
        if (setup_made != form) {
            reconfig_data_format(vectors, weights);
            matmul_block_init(weights, vectors, form == 1 ? 1 : 0, 1, 1, 1);
            setup_made = form;
        }
    }

    // DST 0 = Hardtanh of output row r, tile t of `layer` (channels by pixels): nine
    // weight tiles (SrcB) by version tiles read transposed, and the bias by ones.
    template <uint32_t taps, uint32_t weights, uint32_t ones>
    inline void conv_tile(uint32_t r, uint32_t t, uint32_t layer) {
        setup(weights, taps, 1);
        const uint32_t w = layer * kLayerTiles;
        for (uint32_t dy = 0; dy < 3; dy++) {
            for (uint32_t dx = 0; dx < 3; dx++) {
                const uint32_t x = ((r + dy) * 3 + dx) * kRowTiles + t;
                matmul_block(weights, taps, w + dy * 3 + dx, x, 0, 1, 1, 1, 1);
            }
        }
        matmul_block(weights, ones, w + 9, 0, 0, 1, 1, 1, 1);
        relu_max_tile_init();
        relu_max_tile(0, kOne);
        setup_made = 0;  // the vector unit's set-up changed the matrix unit's
    }

    // Output row r of a layer: each tile's convolution, turned back to pixels by
    // channels, packed in bfloat16 into the write stream's block (out_buffer).
    template <uint32_t taps, uint32_t weights, uint32_t ones>
    inline void conv_row(uint32_t out_buffer, uint32_t r, uint32_t layer) {
        pack_reconfig_data_format(out_buffer);
        for (uint32_t t = 0; t < kRowTiles; t++) {
            tile_regs_acquire();
            conv_tile<taps, weights, ones>(r, t, layer);
            transpose_dest_init<DST_ACCUM_MODE, true>();
            transpose_dest<DST_ACCUM_MODE, true>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_buffer, t);
            tile_regs_release();
        }
    }

    // Column t of the last layer shuffled into `rows` (row 6 r + 3 i + c: output row
    // 2 (first + r) + i, colour c): Z_r, Z'_j = P Z, O_h^T = Q Z'^T (README.md).
    template <
        uint32_t taps,
        uint32_t weights,
        uint32_t ones,
        uint32_t zs,
        uint32_t zp,
        uint32_t rows>
    inline void last_column(uint32_t t, uint32_t layer) {
        pack_reconfig_data_format(zs);
        cb_reserve_back(zs, kRows);
        for (uint32_t r = 0; r < kRows; r++) {
            tile_regs_acquire();
            conv_tile<taps, weights, ones>(r, t, layer);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, zs, r);
            tile_regs_release();
        }
        cb_push_back(zs, kRows);
        cb_wait_front(zs, kRows);
        setup(weights, zs, 2);
        tile_regs_acquire();
        for (uint32_t j = 0; j < 2; j++) {
            for (uint32_t r = 0; r < kRows; r++) {
                matmul_block(weights, zs, kShuffle + j * kRows + r, r, j, 0, 1, 1, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(zp, 2);
        pack_tile<true>(0, zp, 0);
        pack_tile<true>(1, zp, 1);
        tile_regs_release();
        cb_pop_front(zs, kRows);
        cb_push_back(zp, 2);
        cb_wait_front(zp, 2);
        setup(weights, zp, 1);
        tile_regs_acquire();
        for (uint32_t h = 0; h < 2; h++) {
            for (uint32_t j = 0; j < 2; j++) {
                matmul_block(
                    weights, zp, kShuffle + 2 * kRows + 2 * h + j, j, h, 1, 1, 1, 1
                );
            }
        }
        transpose_dest_init<DST_ACCUM_MODE, true>();
        transpose_dest<DST_ACCUM_MODE, true>(0);
        transpose_dest<DST_ACCUM_MODE, true>(1);
        setup_made = 0;
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest_init<2, 2 * kRowTiles>(rows);
        pack_untilize_dest<2, 2 * kRowTiles>(rows, 1, t);
        pack_untilize_uninit(rows);
        tile_regs_release();
        cb_pop_front(zp, 2);
    }

}  // namespace upscale
