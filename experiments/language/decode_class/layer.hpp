// The decode layer's operations for the class's bodies (decode_layer.py): the hand's
// (projects/decode/kernels), each buffer an argument in place of its DC_CB_ index.
#pragma once

#include <stdint.h>

#include "../../../projects/decode/decode.h"

constexpr uint32_t kLayerTileBytes = 2048;  // a bf16 tile

// attend: a core's part of its KV head's keys (attend.hpp's Attend): its full tiles,
// the root's token tiles from `last`, `keys` the tiles it holds, `rows` the live rows.
struct Attend {
    uint32_t rank;
    uint32_t first;
    uint32_t count;
    uint32_t last;
    uint32_t row;
    uint32_t rows;
    uint32_t span;
    uint32_t keys;
};

inline Attend attend_of(
    uint32_t rank, uint32_t size, uint32_t position, uint32_t live
) {
    const uint32_t last = position / 32u;
    const uint32_t span = (position % 32u + live - 1u) / 32u;
    const km::Partition share = km::partition(last, size - 1u - rank, size);
    return {
        rank,
        share.start,
        share.len,
        last,
        position % 32u,
        live,
        span,
        share.len + (rank == 0 ? 1u + span : 0u)
    };
}

// attend's constants (head_tiles' item): ones, the row-0 tile, the identity, then the
// root's two masks.
constexpr uint32_t kAttendOnes = 0;
constexpr uint32_t kAttendRow = 1;
constexpr uint32_t kAttendIdentity = 2;
constexpr uint32_t kAttendMask = 3;
constexpr uint32_t kAttendItem = 8;
// head_tiles' item (8 pages): qkv's head mean and norm weight in pages 0 to 4, the
// identity and the mean tile (decode.h's DC_CONSTANT_*) in pages 5 and 6.
constexpr uint32_t kItemPages = 8;
constexpr uint32_t kItemIdentity = 5;
constexpr uint32_t kItemMean = 6;
constexpr uint32_t kQueryTiles = DC_HEAD_GROUP * DC_HEAD_TILES;
// attend's scores and P take own buffers of qkv's (y's 8 pages, normed's 4): each item
// is the buffer whole, so every handler's items stay whole (an item never wraps).
constexpr uint32_t kScoresPages = 8;
constexpr uint32_t kProbabilityPages = 4;
static_assert(
    2 * LAYER_KEYS_MOST <= kScoresPages && LAYER_KEYS_MOST <= kProbabilityPages
);
constexpr uint32_t kStateSum = 1;  // a head's state: maximum, sum, output tiles
constexpr uint32_t kStateOutput = 2;

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reduce.h"
#include "../../../projects/decode/kernels/rows.hpp"

// Every step inlined, so each own buffer's index is a constant where the math library
// reads the buffer's tile shape (a table the persistent kernel holds), as attend's.
#define LAYER_STEP inline __attribute__((always_inline))

// A stream's half is a word, not a constant: math is given `kShape`, an own buffer of
// the same 32 x 32 bf16 tiles, and unpack the half itself.
#if defined(TRISC_MATH)
#define LAYER_HALF(cb) kShape
#else
#define LAYER_HALF(cb) (cb)
#endif

namespace layer {

    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr auto kMul = EltwiseBinaryType::ELWMUL;

    // DEST tile `dst` times tile `tile` of `buffer` at HiFi4 (ops.hpp's multiply).
    LAYER_STEP void multiply_init(uint32_t buffer) {
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

    LAYER_STEP void multiply(uint32_t buffer, uint32_t tile, uint32_t dst) {
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

    // matmul_init with its math part given the tiles' shape (32 x 32 here) rather than
    // read from the table of buffers: a stream's half is not a constant (attend's).
    LAYER_STEP void product_init(uint32_t in0, uint32_t in1, uint32_t transpose) {
        MATH((llk_math_matmul_init_impl<MATH_FIDELITY, MM_THROTTLE>(
            32, 32, 32, 32, false, transpose, 1, 1
        )));
        UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose)));
    }

    // DEST 0 into one page of own `buffer`, reserved and pushed. Pack's room guards
    // only its pack and push: returns are decided by waits, which all three CPUs make.
    LAYER_STEP void pack_one(const km::Compute& compute, uint32_t buffer) {
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(buffer, 1);
        pack_reconfig_data_format(buffer);
        if (room) {
            pack_tile(0, buffer);
            cb_push_back(buffer, 1);
        }
        tile_regs_release();
    }

    // The class's formats after a body: both sources and the packer in bf16, as its
    // start-up leaves them (`bf16_in` a bf16 input buffer, `bf16_out` an output).
    LAYER_STEP void restore(uint32_t bf16_in, uint32_t bf16_out) {
        reconfig_data_format(bf16_in, bf16_in);
        pack_reconfig_data_format(bf16_out);
    }

    // add: Y = the sum + the residual's block into `out` (the line's) and own `again`,
    // then Y's rows' sums of squares (y y^T summed, the diagonal kept) into `part`.
    template <uint32_t kShape>
    LAYER_STEP bool residual_add(
        const km::Compute& compute,
        uint32_t sum,
        uint32_t residual,
        uint32_t constants,
        uint32_t out,
        uint32_t part,
        uint32_t again,
        uint32_t tiles,
        bool room
    ) {
        const bool kept = compute.reserve_back(again, tiles);
        reconfig_data_format(sum, LAYER_HALF(residual));
        pack_reconfig_data_format(out);
        for (uint32_t tile = 0; tile < tiles; tile++) {
            tile_regs_acquire();
            copy_init(sum);
            copy_tile(sum, tile, 0);
            add_reuse_dest_init<kReuse>(LAYER_HALF(residual));
            add_reuse_dest_tiles<kReuse>(LAYER_HALF(residual), tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (room && kept) {
                pack_tile(0, out);
                pack_tile(0, again);
            }
            tile_regs_release();
        }
        if (kept) {
            cb_push_back(again, tiles);
        }
        if (!compute.wait_front(again, tiles)) {
            return false;
        }
        reconfig_data_format(again, again);
        tile_regs_acquire();
        product_init(again, again, 1);
        for (uint32_t tile = 0; tile < tiles; tile++) {
            matmul_tiles(again, again, tile, tile, 0);
        }
        multiply_init(constants);
        multiply(constants, kItemIdentity, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(part);
        if (room) {
            pack_tile(0, part);
        }
        tile_regs_release();
        cb_pop_front(again, tiles);
        restore(again, out);
        return true;
    }

    // Own `scale`: each live row's 1 / sqrt(mean square + epsilon) in every column,
    // from the row's squares (`parts`, DC_PARTS tiles) and the mean tile.
    template <uint32_t kShape>
    LAYER_STEP bool rms_scale(
        const km::Compute& compute,
        uint32_t parts,
        uint32_t constants,
        uint32_t squared,
        uint32_t scale,
        uint32_t epsilon,
        uint32_t live
    ) {
        reconfig_data_format(LAYER_HALF(parts), LAYER_HALF(parts));
        tile_regs_acquire();
        copy_init(LAYER_HALF(parts));
        copy_tile(LAYER_HALF(parts), 0, 0);
        add_reuse_dest_init<kReuse>(LAYER_HALF(parts));
        for (uint32_t part = 1; part < DC_PARTS; part++) {
            add_reuse_dest_tiles<kReuse>(LAYER_HALF(parts), part, 0);
        }
        pack_one(compute, squared);
        if (!compute.wait_front(squared, 1)) {
            return false;
        }
        tile_regs_acquire();
        product_init(squared, constants, 0);
        matmul_tiles(squared, constants, 0, kItemMean, 0);
        binop_with_scalar_tile_init();
        DC_ROWS(live, rows::plus, 0, epsilon);
        rsqrt_tile_init();
        DC_ROWS(live, rows::rsqrt, 0);
        cb_pop_front(squared, 1);
        pack_one(compute, scale);
        return compute.wait_front(scale, 1);
    }

    // The sum's tiles times `scale` (rows' 1 / rms, freed) into `out`: the line's write
    // block (`own` false) or an own buffer, reserved and pushed here.
    LAYER_STEP void scaled(
        const km::Compute& compute,
        uint32_t sum,
        uint32_t scale,
        uint32_t out,
        uint32_t tiles,
        bool room,
        bool own
    ) {
        if (own) {
            room = compute.reserve_back(out, tiles);
        }
        reconfig_data_format(sum, scale);
        pack_reconfig_data_format(out);
        for (uint32_t tile = 0; tile < tiles; tile++) {
            tile_regs_acquire();
            copy_init(sum);
            copy_tile(sum, tile, 0);
            multiply_init(scale);
            multiply(scale, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_tile(0, out);
            }
            tile_regs_release();
        }
        if (own && room) {
            cb_push_back(out, tiles);
        }
        cb_pop_front(scale, 1);
    }

    // gate's compute: the sum is the gate's `tiles / 2` tiles then up's, each times
    // `scale`; out = silu(gate) * up, up rounded to bf16 first (own `up`).
    LAYER_STEP bool gate(
        const km::Compute& compute,
        uint32_t sum,
        uint32_t scale,
        uint32_t up,
        uint32_t out,
        uint32_t tiles,
        uint32_t live,
        bool room
    ) {
        const uint32_t half = tiles / 2;
        const bool kept = compute.reserve_back(up, tiles);
        reconfig_data_format(sum, scale);
        pack_reconfig_data_format(up);
        for (uint32_t tile = 0; tile < half; tile++) {
            tile_regs_acquire();
            copy_init(sum);
            copy_tile(sum, half + tile, 0);
            multiply_init(scale);
            multiply(scale, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (kept) {
                pack_tile<true>(0, up, tile);
            }
            tile_regs_release();
        }
        if (kept) {
            cb_push_back(up, tiles);
        }
        if (!compute.wait_front(up, tiles)) {
            return false;
        }
        pack_reconfig_data_format(out);
        for (uint32_t tile = 0; tile < half; tile++) {
            tile_regs_acquire();
            copy_init(sum);
            copy_tile(sum, tile, 0);
            multiply_init(scale);
            multiply(scale, 0, 0);
            silu_tile_init();
            DC_ROWS(live, rows::silu, 0);
            multiply_init(up);
            multiply(up, tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_tile<true>(0, out, tile);
            }
            tile_regs_release();
        }
        cb_pop_front(up, tiles);
        cb_pop_front(scale, 1);
        restore(up, out);
        return true;
    }

    // Head `head` of `y`: its rows' 1 / rms over 128 columns into `scale`, each tile
    // times it and the weight (`head_tiles`: mean tile, the weight's 4) into `normed`.
    LAYER_STEP bool head_norm(
        const km::Compute& compute,
        uint32_t y,
        uint32_t head_tiles,
        uint32_t squared,
        uint32_t scale,
        uint32_t normed,
        uint32_t head,
        uint32_t epsilon,
        uint32_t live
    ) {
        const uint32_t base = head * DC_HEAD_TILES;
        tile_regs_acquire();
        copy_init(y);
        copy_tile(y, base, 0);
        mul_binary_tile_init();
        DC_ROWS(live, rows::mul, 0, 0, 0);
        for (uint32_t tile = 1; tile < DC_HEAD_TILES; tile++) {
            copy_init(y);
            copy_tile(y, base + tile, 1);
            mul_binary_tile_init();
            DC_ROWS(live, rows::mul, 1, 1, 1);
            add_binary_tile_init();
            DC_ROWS(live, rows::add, 0, 1, 0);
        }
        pack_one(compute, squared);
        if (!compute.wait_front(squared, 1)) {
            return false;
        }
        tile_regs_acquire();
        product_init(squared, head_tiles, 0);
        matmul_tiles(squared, head_tiles, 0, 0, 0);
        binop_with_scalar_tile_init();
        DC_ROWS(live, rows::plus, 0, epsilon);
        rsqrt_tile_init();
        DC_ROWS(live, rows::rsqrt, 0);
        cb_pop_front(squared, 1);
        pack_one(compute, scale);
        if (!compute.wait_front(scale, 1)) {
            return false;
        }
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
            copy_init(y);
            copy_tile(y, base + tile, tile);
            multiply_init(scale);
            multiply(scale, 0, tile);
            multiply_init(head_tiles);
            multiply(head_tiles, 1 + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(normed, DC_HEAD_TILES);
        pack_reconfig_data_format(normed);
        for (uint32_t tile = 0; room && tile < DC_HEAD_TILES; tile++) {
            pack_tile(tile, normed);
        }
        tile_regs_release();
        cb_pop_front(scale, 1);
        if (room) {
            cb_push_back(normed, DC_HEAD_TILES);
        }
        return compute.wait_front(normed, DC_HEAD_TILES);
    }

    // The normed head rotated: tile j times cos_j plus tile (j + 2) % 4 times the
    // signed sin_j (`rope`: cos's 4 tiles, then sin's), packed in turn into `out`.
    LAYER_STEP void rotated(
        uint32_t normed, uint32_t rope, uint32_t out, uint32_t live, bool room
    ) {
        pack_reconfig_data_format(out);
        for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
            tile_regs_acquire();
            copy_init(normed);
            copy_tile(normed, tile, 0);
            copy_tile(normed, (tile + 2) % DC_HEAD_TILES, 2);
            copy_init(rope);
            copy_tile(rope, tile, 1);
            copy_tile(rope, DC_HEAD_TILES + tile, 3);
            mul_binary_tile_init();
            DC_ROWS(live, rows::mul, 0, 1, 0);
            DC_ROWS(live, rows::mul, 2, 3, 2);
            add_binary_tile_init();
            DC_ROWS(live, rows::add, 0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_tile(0, out);
            }
            tile_regs_release();
        }
        cb_pop_front(normed, DC_HEAD_TILES);
    }

    // qkv's root on a q or k block: the scaled tiles into own `y`, then each head's
    // norm and rotation into `out` (the write stream's block).
    LAYER_STEP bool heads(
        const km::Compute& compute,
        uint32_t sum,
        uint32_t scale,
        uint32_t y,
        uint32_t head_tiles,
        uint32_t rope,
        uint32_t squared,
        uint32_t normed,
        uint32_t out,
        uint32_t tiles,
        uint32_t epsilon,
        uint32_t live,
        bool room
    ) {
        scaled(compute, sum, scale, y, tiles, true, true);
        if (
            !compute.wait_front(y, tiles) ||
            !compute.wait_front(rope, 2 * DC_HEAD_TILES)
        ) {
            return false;
        }
        bool done = true;
        reconfig_data_format(y, y);
        for (uint32_t head = 0; done && head < tiles / DC_HEAD_TILES; head++) {
            done = (
                head_norm(
                    compute, y, head_tiles, squared, scale, normed, head, epsilon, live
                )
            );
            if (done) {
                rotated(normed, rope, out, live, room);
            }
        }
        cb_pop_front(rope, 2 * DC_HEAD_TILES);
        cb_pop_front(y, tiles);
        restore(y, out);
        return done;
    }

    // ---- attend (attend/compute.cpp), the state written in place in own `state` ----

    LAYER_STEP void copy(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        copy_tile(cb, tile, dst);
    }

    // The merge's products: fp32 buffers read as tf32 into both sources, HiFi4.
    LAYER_STEP void rescale_init(uint32_t state, uint32_t rescale) {
        constexpr uint32_t f32 = static_cast<uint32_t>(DataFormat::Float32);
        constexpr uint32_t tf32 = static_cast<uint32_t>(DataFormat::Tf32);
        UNPACK(
            (
                _llk_unpack_reconfig_data_format_srca_impl_<
                    DST_ACCUM_MODE,
                    p_dim_stride_target::IGNORE>(
                        f32, tf32, get_local_cb_interface(state).fifo_page_size
                    )
            )
        );
        UNPACK(
            (
                _llk_unpack_reconfig_data_format_srcb_impl_<
                    DST_ACCUM_MODE,
                    p_dim_stride_target::IGNORE>(
                        f32, tf32, get_local_cb_interface(rescale).fifo_page_size
                    )
            )
        );
        MATH((_llk_math_reconfig_data_format_<DST_ACCUM_MODE>(tf32, tf32)));
        UNPACK((llk_unpack_AB_matmul_init(rescale, state, 0)));
        MATH((llk_math_matmul_init_impl<MathFidelity::HiFi4, MM_THROTTLE>(
            32, 32, 32, 32, false, 0, 1, 1
        )));
    }

    LAYER_STEP void rescale_tile(
        uint32_t rescale, uint32_t left, uint32_t cb, uint32_t right, uint32_t dst
    ) {
        UNPACK((llk_unpack_AB_matmul(rescale, cb, left, right)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(dst)));
    }

    // DEST `dst` = exp(DEST `dst` / sqrt(128)) on the live rows.
    LAYER_STEP void scaled_exp(uint32_t dst, uint32_t live) {
        binop_with_scalar_tile_init();
        DC_ROWS(live, rows::times, dst, DC_SCALE_BITS);
        exp_tile_init<false>();
        DC_ROWS(live, rows::exp, dst);
    }

    LAYER_STEP void zeros(uint32_t dst, uint32_t live) {
        if (live < 32u) {
            fill_tile_init();
            fill_tile(dst, 0.0f);
        }
    }

    LAYER_STEP void pack_at(uint32_t dst, uint32_t cb, uint32_t index) {
        pack_reconfig_data_format(cb);
        pack_tile<true>(dst, cb, index);
    }

    // Both heads' scores (bf16, own `scores`) against each key tile as it comes,
    // tile 2 j + head; the root's last key tiles start from their masks.
    template <uint32_t kShape>
    LAYER_STEP bool scores_of(
        const km::Compute& compute,
        const Attend& at,
        uint32_t queries,
        uint32_t keys,
        uint32_t constants,
        uint32_t scores
    ) {
        const bool room = compute.reserve_back(scores, kScoresPages);
        for (uint32_t key = 0; key < at.keys; key++) {
            if (!compute.wait_front(keys, DC_HEAD_TILES)) {
                return false;
            }
            for (uint32_t head = 0; head < DC_HEAD_GROUP; head++) {
                tile_regs_acquire();
                if (at.rank == 0 && key >= at.count) {
                    copy(constants, kAttendMask + key - at.count, 0);
                }
                reconfig_data_format(keys, LAYER_HALF(queries));
                product_init(queries, keys, 1);
                for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                    matmul_tiles(queries, keys, head * DC_HEAD_TILES + tile, tile, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                if (room) {
                    pack_at(0, scores, 2 * key + head);
                }
                tile_regs_release();
            }
            cb_pop_front(keys, DC_HEAD_TILES);
        }
        const uint32_t left = (LAYER_KEYS_MOST - at.keys) * DC_HEAD_TILES;
        if (left != 0) {
            if (!compute.wait_front(keys, left)) {
                return false;
            }
            cb_pop_front(keys, left);
        }
        if (room) {
            cb_push_back(scores, kScoresPages);
        }
        return compute.wait_front(scores, kScoresPages);
    }

    // Head `head`'s state into `state` (in place): its maximum in every column, the
    // sum of P in every column, P V.
    LAYER_STEP bool head_state(
        const km::Compute& compute,
        const Attend& at,
        uint32_t head,
        uint32_t constants,
        uint32_t scores,
        uint32_t row_max,
        uint32_t reference,
        uint32_t probabilities,
        uint32_t values,
        uint32_t state
    ) {
        const uint32_t base = head * DC_STATE_TILES;
        tile_regs_acquire();  // each row's largest score, in column 0
        reconfig_data_format(scores, constants);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(scores, constants, row_max);
        for (uint32_t key = 0; key < at.keys; key++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                scores, constants, 2 * key + head, kAttendOnes, 0
            );
        }
        reduce_uninit();
        pack_one(compute, row_max);
        if (!compute.wait_front(row_max, 1)) {
            return false;
        }
        tile_regs_acquire();  // the maximum in every column: times the row-0 tile
        reconfig_data_format(constants, row_max);
        product_init(row_max, constants, 0);
        matmul_tiles(row_max, constants, 0, kAttendRow, 0);
        tile_regs_commit();
        tile_regs_wait();
        // `reference` is the merge's two-page buffer: a page of two, so the merge's
        // pair stays whole (an item never wraps)
        const bool kept = compute.reserve_back(reference, 2);
        pack_at(0, state, base);
        if (kept) {
            pack_reconfig_data_format(reference);
            pack_tile(0, reference);
            cb_push_back(reference, 2);
        }
        tile_regs_release();
        cb_pop_front(row_max, 1);
        if (!compute.wait_front(reference, 2)) {
            return false;
        }
        const bool room = compute.reserve_back(probabilities, kProbabilityPages);
        for (uint32_t key = 0; key < at.keys; key++) {
            tile_regs_acquire();  // P's rows past the live ones are 0
            zeros(0, at.rows);
            copy(scores, 2 * key + head, 1);
            copy(reference, 0, 2);
            sub_binary_tile_init();
            DC_ROWS(at.rows, rows::sub, 1, 2, 0);
            scaled_exp(0, at.rows);
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_at(0, probabilities, key);
            }
            tile_regs_release();
        }
        cb_pop_front(reference, 2);
        if (room) {
            cb_push_back(probabilities, kProbabilityPages);
        }
        if (
            !compute.wait_front(probabilities, kProbabilityPages) ||
            !compute.wait_front(values, LAYER_KEYS_MOST * DC_HEAD_TILES)
        ) {
            return false;
        }
        tile_regs_acquire();  // the sum of P: P times ones
        reconfig_data_format(constants, probabilities);
        product_init(probabilities, constants, 0);
        for (uint32_t key = 0; key < at.keys; key++) {
            matmul_tiles(probabilities, constants, key, kAttendOnes, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_at(0, state, base + kStateSum);
        tile_regs_release();
        tile_regs_acquire();  // P V, the head's 4 output tiles
        reconfig_data_format(values, probabilities);
        product_init(probabilities, values, 0);
        for (uint32_t key = 0; key < at.keys; key++) {
            for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                matmul_tiles(
                    probabilities, values, key, key * DC_HEAD_TILES + tile, tile
                );
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
            pack_at(tile, state, base + kStateOutput + tile);
        }
        tile_regs_release();
        cb_pop_front(probabilities, kProbabilityPages);
        return true;
    }

    // The root: the token's key (`keys`' tiles 0-3) and value (`values`' from `at`),
    // held blocks of qkv, into own `token`'s 8 pages, pushed for the movers' append.
    template <uint32_t kShape>
    LAYER_STEP void token_copy(
        const km::Compute& compute,
        uint32_t keys,
        uint32_t values,
        uint32_t at,
        uint32_t token
    ) {
        const bool room = compute.reserve_back(token, 2 * DC_HEAD_TILES);
        reconfig_data_format_srca(LAYER_HALF(keys));
        pack_reconfig_data_format(token);
        for (uint32_t tile = 0; tile < 2 * DC_HEAD_TILES; tile++) {
            const uint32_t from = tile < DC_HEAD_TILES ? keys : values;
            tile_regs_acquire();
            copy_init(LAYER_HALF(from));
            const uint32_t index = (
                tile < DC_HEAD_TILES ? tile : at + tile - DC_HEAD_TILES
            );
            copy_tile(LAYER_HALF(from), index, 0);
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_tile(0, token);
            }
            tile_regs_release();
        }
        if (room) {
            cb_push_back(token, 2 * DC_HEAD_TILES);
        }
    }

    // A core without keys: a state of zeros.
    LAYER_STEP void zero_state(uint32_t state) {
        for (uint32_t first = 0; first < DC_STATE_ITEM; first += 4) {
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t tile = 0; tile < 4; tile++) {
                fill_tile(tile, 0.0f);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < 4; tile++) {
                pack_at(tile, state, first + tile);
            }
            tile_regs_release();
        }
    }

    // A written state's token: pack pushes a page of own `ready` after its packs of
    // the state; the next reader waits for it and frees it (the state is in place).
    LAYER_STEP void state_written(const km::Compute& compute, uint32_t ready) {
        if (compute.reserve_back(ready, 1)) {
            cb_push_back(ready, 1);
        }
    }

    LAYER_STEP bool state_taken(const km::Compute& compute, uint32_t ready) {
        if (!compute.wait_front(ready, 1)) {
            return false;
        }
        cb_pop_front(ready, 1);
        return true;
    }

    // This core's state: the scores, each head's state, the values freed.
    template <uint32_t kShape>
    LAYER_STEP bool attend_state(
        const km::Compute& compute,
        const Attend& at,
        uint32_t queries,
        uint32_t keys,
        uint32_t values,
        uint32_t constants,
        uint32_t scores,
        uint32_t row_max,
        uint32_t reference,
        uint32_t probabilities,
        uint32_t state,
        uint32_t ready
    ) {
        if (
            !compute.wait_front(constants, kAttendItem) ||
            !scores_of<kShape>(compute, at, queries, keys, constants, scores)
        ) {
            return false;
        }
        bool ok = true;
        for (uint32_t head = 0; ok && at.keys != 0 && head < DC_HEAD_GROUP; head++) {
            ok = (
                head_state(
                    compute,
                    at,
                    head,
                    constants,
                    scores,
                    row_max,
                    reference,
                    probabilities,
                    values,
                    state
                )
            );
        }
        if (ok && at.keys == 0) {
            zero_state(state);
        }
        const uint32_t pages = LAYER_KEYS_MOST * DC_HEAD_TILES;
        if (!ok || !compute.wait_front(values, pages)) {
            return false;
        }
        cb_pop_front(scores, kScoresPages);
        cb_pop_front(values, pages);
        cb_pop_front(constants, kAttendItem);
        state_written(compute, ready);
        return true;
    }

    // Arrival `base` merged into the state in place: m = max(m1, m2), sum and output
    // diag(a1) X1 + diag(a2) X2, a_i = exp((m_i - m) / sqrt(128)).
    LAYER_STEP bool merge(
        const km::Compute& compute,
        uint32_t arrival,
        uint32_t base,
        uint32_t state,
        uint32_t rescale,
        uint32_t identity,
        uint32_t ready,
        uint32_t live
    ) {
        if (!state_taken(compute, ready)) {
            return false;
        }
        for (uint32_t head = 0; head < DC_HEAD_GROUP; head++) {
            const uint32_t at = head * DC_STATE_TILES;
            tile_regs_acquire();  // m, then each side's factor on the diagonal
            copy(state, at, 0);
            copy(arrival, base + at, 1);
            copy(identity, kAttendIdentity, 3);
            binary_max_tile_init();
            DC_ROWS(live, rows::max, 0, 1, 2);
            sub_binary_tile_init();
            DC_ROWS(live, rows::sub, 0, 2, 0);
            DC_ROWS(live, rows::sub, 1, 2, 1);
            scaled_exp(0, live);
            scaled_exp(1, live);
            mul_binary_tile_init();
            DC_ROWS(live, rows::mul, 0, 3, 0);
            DC_ROWS(live, rows::mul, 1, 3, 1);
            tile_regs_commit();
            tile_regs_wait();
            const bool kept = compute.reserve_back(rescale, 2);
            pack_at(2, state, at);
            if (kept) {
                pack_reconfig_data_format(rescale);
                pack_tile(0, rescale);
                pack_tile(1, rescale);
                cb_push_back(rescale, 2);
            }
            tile_regs_release();
            if (!compute.wait_front(rescale, 2)) {
                return false;
            }
            rescale_init(state, rescale);
            for (uint32_t first = kStateSum; first < DC_STATE_TILES; first += 4) {
                const uint32_t last = (
                    first + 4 < DC_STATE_TILES ? first + 4 : DC_STATE_TILES
                );
                tile_regs_acquire();
                for (uint32_t tile = first; tile < last; tile++) {
                    rescale_tile(rescale, 0, state, at + tile, tile - first);
                    rescale_tile(rescale, 1, arrival, base + at + tile, tile - first);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t tile = first; tile < last; tile++) {
                    pack_at(tile - first, state, at + tile);
                }
                tile_regs_release();
            }
            cb_pop_front(rescale, 2);
        }
        state_written(compute, ready);
        return true;
    }

    // The root: each output tile over its head's sum on the live rows (the others 0),
    // bf16 into `out` (the write stream's block).
    LAYER_STEP bool finish(
        const km::Compute& compute,
        uint32_t state,
        uint32_t ready,
        uint32_t out,
        uint32_t live,
        bool room
    ) {
        if (!state_taken(compute, ready)) {
            return false;
        }
        for (uint32_t head = 0; head < DC_HEAD_GROUP; head++) {
            const uint32_t at = head * DC_STATE_TILES;
            for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                tile_regs_acquire();
                zeros(0, live);
                copy(state, at + kStateOutput + tile, 1);
                copy(state, at + kStateSum, 2);
                div_binary_tile_init();
                DC_ROWS(live, rows::div, 1, 2, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (room) {
                    pack_at(0, out, head * DC_HEAD_TILES + tile);
                }
                tile_regs_release();
            }
        }
        return true;
    }

}  // namespace layer
#else
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"

namespace layer {

    // A timed cb_reserve_back on either mover (mover B's runtime has no reserve).
    template <typename Kernel>
    inline bool reserve(const Kernel& core, uint32_t cb, uint32_t pages) {
        if (!core.wait([&] {
            return cb_pages_reservable_at_back(cb, pages);
        })) {
            return false;
        }
        cb_reserve_back(cb, pages);
        return true;
    }

    // A timed cb_wait_front on either mover.
    template <typename Kernel>
    inline bool wait(const Kernel& core, uint32_t cb, uint32_t pages) {
        return core.wait([&] {
            return cb_pages_available_at_front(cb, pages);
        });
    }

    // One of two movers reading halves of `cb`'s item: its `pages` freed (both frees,
    // in either order, free the item) and its read pointer moved on to the next item.
    inline void taken_half(uint32_t cb, uint32_t pages) {
        cb_pop_front(cb, pages);
        LocalCBInterface& local = get_local_cb_interface(cb);
        local.fifo_rd_ptr += pages * local.fifo_page_size;
        if (local.fifo_rd_ptr >= local.fifo_limit) {
            local.fifo_rd_ptr -= local.fifo_size;
        }
    }

    // Whether the mover's reads have landed, timed.
    template <typename Kernel>
    inline bool reads_flushed(const Kernel& core) {
        return core.wait([] {
            return ncrisc_noc_reads_flushed(noc_index);
        });
    }

    // head_tiles' item, pushed: identity and mean tile (`table`'s pages 0, 1) in pages
    // 5, 6; with `qk`, its head mean tile and the norm weight's 4 in pages 0 to 4.
    template <typename Kernel, typename Table>
    inline bool read_item(
        const Kernel& core,
        const km_command_t& command,
        const Table& table,
        bool qk,
        uint32_t layer_index,
        uint32_t key,
        uint32_t head_tiles
    ) {
        if (!reserve(core, head_tiles, kItemPages)) {
            return false;
        }
        const uint32_t to = get_write_ptr(head_tiles);
        const uint32_t weight = (
            DC_QK_WEIGHTS + (2u * layer_index + key) * DC_HEAD_TILES
        );
        if (!km::read_pages(
            core, command, table, 0, 2, to + kItemIdentity * kLayerTileBytes
        ) ||
            (
                qk &&
                (
                    !km::read_pages(core, command, table, DC_QK_HEAD_MEAN, 1, to) ||
                    !km::read_pages(
                        core,
                        command,
                        table,
                        weight,
                        DC_HEAD_TILES,
                        to + kLayerTileBytes
                    )
                )
            ) ||
            !reads_flushed(core)) {
            return false;
        }
        cb_push_back(head_tiles, kItemPages);
        return true;
    }

    // The rotation rows from `first` into own `scratch`, then row i into row i of own
    // `rope`'s 8 tiles (cos's 4, sin's 4), pushed (the hand's read_rotary).
    template <typename Kernel, typename Table>
    inline bool read_rotary(
        const Kernel& core,
        const km_command_t& command,
        const Table& table,
        uint32_t first,
        uint32_t live,
        uint32_t scratch,
        uint32_t rope
    ) {
        const uint32_t rows = (live + 3u) & ~3u;
        constexpr uint32_t kRowsAPage = kLayerTileBytes / DC_ROPE_ROW;
        constexpr uint32_t kTiles = 2 * DC_HEAD_TILES;
        if (!reserve(core, rope, kTiles)) {
            return false;
        }
        const uint32_t from = get_write_ptr(scratch);
        const uint32_t to = get_write_ptr(rope);
        if (KM_GUARD && rows > 32u) {  // the copy below writes rows of one tile
            return core.gave_up_at(km::kGaveBadItem, km::kWhereData, __LINE__);
        }
        uint32_t asked = 0;
        for (uint32_t row = first; row < first + rows;) {
            const uint32_t in_page = row % kRowsAPage;
            const uint32_t count = (
                first + rows - row < kRowsAPage - in_page
                    ? first + rows - row
                    : kRowsAPage - in_page
            );
            if (!km::ask_bytes(
                core,
                command,
                table,
                DC_QK_ROTARY + row / kRowsAPage,
                in_page * DC_ROPE_ROW,
                count * DC_ROPE_ROW,
                from + (row - first) * DC_ROPE_ROW,
                asked
            )) {
                return false;
            }
            row += count;
        }
        if (!km::bytes_wait(core, command, asked)) {
            return false;
        }
        invalidate_l1_cache();  // the core reads what the NoC wrote
        // row i's 16 values of a face half: face (i / 16) x 2 + half, its row i % 16
        for (uint32_t row = 0; row < rows; row++) {
            const uint32_t at = ((row >> 4) * 2) * 512 + (row & 15) * 32;
            for (uint32_t piece = 0; piece < 2 * kTiles; piece++) {
                if (!km::copy_l1(
                    core,
                    to + (piece >> 1) * kLayerTileBytes + (piece & 1) * 512 + at,
                    rope,
                    from + row * DC_ROPE_ROW + piece * 32,
                    scratch,
                    8
                )) {
                    return false;
                }
            }
        }
        cb_push_back(rope, kTiles);
        return true;
    }

    // ---- attend's movers (attend.hpp) ----

    // The first page of key tile `tile` of KV head `head` in a cache tensor.
    inline uint32_t cache_page(uint32_t head, uint32_t tile) {
        return (head * LAYER_CACHE_TILES + tile) * DC_HEAD_TILES;
    }

    // attend's constants (ones, the row-0 tile, the identity) and the root's two
    // masks of the tokens' tiles into own `constants` as one item, pushed.
    template <typename Kernel, typename Table>
    inline bool read_attend_constants(
        const Kernel& core,
        const km_command_t& command,
        const Table& table,
        const Attend& at,
        uint32_t constants
    ) {
        if (!reserve(core, constants, kAttendItem)) {
            return false;
        }
        const uint32_t to = get_write_ptr(constants);
        const uint32_t first = DC_MASK_ZERO + at.row;
        const uint32_t second = at.span != 0 ? first - 32u : first;
        if (
            !km::read_pages(core, command, table, DC_ATTEND_ONES, 3, to) ||
            (
                at.rank == 0 &&
                (!km::read_pages(
                    core, command, table, first, 1, to + kAttendMask * kLayerTileBytes
                ) ||
                 !km::read_pages(
                     core,
                     command,
                     table,
                     second,
                     1,
                     to + (kAttendMask + 1) * kLayerTileBytes
                 ))
            ) ||
            !reads_flushed(core)
        ) {
            return false;
        }
        cb_push_back(constants, kAttendItem);
        return true;
    }

    // This core's cached key (value) tiles into `cb` (reserved whole), DC_KEY_BATCH
    // at a time, each pushed when landed.
    template <typename Kernel, typename Cache>
    inline bool read_cache(
        const Kernel& core,
        const km_command_t& command,
        const Attend& at,
        uint32_t head,
        uint32_t cb,
        const Cache& cache
    ) {
        if (!reserve(core, cb, LAYER_KEYS_MOST * DC_HEAD_TILES)) {
            return false;
        }
        for (uint32_t done = 0; done < at.count;) {
            const uint32_t batch = (
                at.count - done < DC_KEY_BATCH ? at.count - done : DC_KEY_BATCH
            );
            const uint32_t pages = batch * DC_HEAD_TILES;
            if (!km::read_pages(
                core,
                command,
                cache,
                cache_page(head, at.first + done),
                pages,
                get_write_ptr(cb)
            ) ||
                !reads_flushed(core)) {
                return false;
            }
            cb_push_back(cb, pages);
            done += batch;
        }
        return true;
    }

    // Whether the mover's writes have gone, timed.
    template <typename Kernel>
    inline bool writes_flushed(const Kernel& core) {
        return core.wait([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
    }

    // The root: the token tiles' key (value) rows (`token`: L1 address of its 4 tiles
    // in buffer `tokens`) put into the cache's tiles, written back, pushed as its last.
    template <typename Kernel, typename Cache>
    inline bool append(
        const Kernel& core,
        const km_command_t& command,
        const Attend& at,
        uint32_t head,
        uint32_t cb,
        const Cache& cache,
        uint32_t token,
        uint32_t tokens
    ) {
        const uint32_t slot = get_write_ptr(cb);
        const uint32_t page = cache_page(head, at.last);
        const uint32_t pages = (1u + at.span) * DC_HEAD_TILES;
        // the copy below: rows of the token's tile into at most two cache tiles
        if (
            KM_GUARD &&
            (at.span > 1u || at.rows > 32u || at.row + at.rows > (1u + at.span) * 32u)
        ) {
            return core.gave_up_at(km::kGaveBadItem, km::kWhereData, __LINE__);
        }
        if (
            !km::read_pages(core, command, cache, page, pages, slot) ||
            !reads_flushed(core)
        ) {
            return false;
        }
        invalidate_l1_cache();  // the core reads what the NoC wrote
        constexpr uint32_t kFaceBytes = 512;
        for (uint32_t i = 0; i < at.rows; i++) {
            const uint32_t place = at.row + i;
            const uint32_t to_row = place % 32u;
            const uint32_t to_offset = (
                (place / 32u) * DC_HEAD_TILES * kLayerTileBytes +
                (to_row >> 4) * 2 * kFaceBytes + (to_row & 15) * 32
            );
            const uint32_t from_offset = (i >> 4) * 2 * kFaceBytes + (i & 15) * 32;
            for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                for (uint32_t face = 0; face < 2; face++) {
                    const uint32_t at_face = tile * kLayerTileBytes + face * kFaceBytes;
                    if (!km::copy_l1(
                        core,
                        slot + at_face + to_offset,
                        cb,
                        token + at_face + from_offset,
                        tokens,
                        8
                    )) {
                        return false;
                    }
                }
            }
        }
        if (
            !km::write_pages(core, command, cache, page, pages, slot) ||
            !writes_flushed(core)
        ) {
            return false;
        }
        cb_push_back(cb, pages);
        return true;
    }

    // The buffer's pages past this core's keys, pushed empty.
    inline void pad(const Attend& at, uint32_t cb) {
        const uint32_t left = (LAYER_KEYS_MOST - at.keys) * DC_HEAD_TILES;
        if (left != 0) {
            cb_push_back(cb, left);
        }
    }

}  // namespace layer
#endif
