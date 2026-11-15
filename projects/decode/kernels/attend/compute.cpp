// Attend, compute: per query head, scores, row maximum, P = exp((s - max) / sqrt(128)),
// its sums and P V; the states merge up the group's tree, the root divides by the sum.
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reduce.h"
#include "attend.hpp"

// Every step inlined, so each buffer's index is a constant where the math library
// reads the buffer's tile shape (a table the persistent kernel holds).
#define ATTEND_STEP inline __attribute__((always_inline))

namespace {

    constexpr uint32_t kOnes = 0;  // DC_CB_ATTEND_CONSTANT: ones, the row-0 tile, identity
    constexpr uint32_t kRow = 1;
    constexpr uint32_t kIdentity = 2;
    constexpr uint32_t kConstants = 3;
    constexpr uint32_t kQueryTiles = DC_HEAD_GROUP * DC_HEAD_TILES;
    constexpr uint32_t kSum = 1;  // a head's state: maximum, sum, output tiles
    constexpr uint32_t kOutput = 2;

    ATTEND_STEP void copy(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        copy_tile(cb, tile, dst);
    }

    // matmul_init, its math part given the tiles' shape (32 x 32, every buffer here)
    // rather than read from the table of buffers.
    ATTEND_STEP void product_init(uint32_t in0, uint32_t in1, uint32_t transpose) {
        reconfig_data_format(in1, in0);  // in0 to source B, in1 to source A
        UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose)));
        MATH((llk_math_matmul_init_impl<MATH_FIDELITY, MM_THROTTLE>(
            kTileRows, kTileRows, kTileRows, kTileRows, false, transpose, 1, 1
        )));
    }

    // The merge's products: fp32 buffers that unpack to DEST elsewhere, read here as
    // tf32 into both sources, every fidelity phase (the factors' low bits count).
    ATTEND_STEP void rescale_init() {
        constexpr uint32_t f32 = static_cast<uint32_t>(DataFormat::Float32);
        constexpr uint32_t tf32 = static_cast<uint32_t>(DataFormat::Tf32);
        UNPACK(
            (
                _llk_unpack_reconfig_data_format_srca_impl_<
                    DST_ACCUM_MODE,
                    p_dim_stride_target::IGNORE>(
                        f32, tf32, get_local_cb_interface(DC_CB_STATE).fifo_page_size
                    )
            )
        );
        UNPACK(
            (
                _llk_unpack_reconfig_data_format_srcb_impl_<
                    DST_ACCUM_MODE,
                    p_dim_stride_target::IGNORE>(
                        f32, tf32, get_local_cb_interface(DC_CB_RESCALE).fifo_page_size
                    )
            )
        );
        MATH((_llk_math_reconfig_data_format_<DST_ACCUM_MODE>(tf32, tf32)));
        UNPACK((llk_unpack_AB_matmul_init(DC_CB_RESCALE, DC_CB_STATE, 0)));
        MATH((llk_math_matmul_init_impl<MathFidelity::HiFi4, MM_THROTTLE>(
            kTileRows, kTileRows, kTileRows, kTileRows, false, 0, 1, 1
        )));
    }

    // DEST `dst` += tile `left` of DC_CB_RESCALE times tile `right` of `cb`.
    ATTEND_STEP void rescale_tile(
        uint32_t left, uint32_t cb, uint32_t right, uint32_t dst
    ) {
        UNPACK((llk_unpack_AB_matmul(DC_CB_RESCALE, cb, left, right)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(dst)));
    }

    // DEST `dst` = exp(DEST `dst` / sqrt(128)) on the live rows.
    ATTEND_STEP void scaled_exp(uint32_t dst, uint32_t live) {
        binop_with_scalar_tile_init();
        DC_ROWS(live, rows::times, dst, DC_SCALE_BITS);
        exp_tile_init<false>();
        DC_ROWS(live, rows::exp, dst);
    }

    // DEST `dst` all zeros, so its rows past the live ones stay 0.
    ATTEND_STEP void zeros(uint32_t dst, uint32_t live) {
        if (live < kTileRows) {
            fill_tile_init();
            fill_tile(dst, 0.0f);
        }
    }

    // The section's DEST tile `dst` into tile `index` of `cb` (reserved).
    ATTEND_STEP void pack_at(uint32_t dst, uint32_t cb, uint32_t index) {
        pack_reconfig_data_format(cb);
        pack_tile<true>(dst, cb, index);
    }

    // Both heads' scores (bf16) against each key tile as it comes, tile 2 j + head;
    // the root's last key tile starts from its mask.
    ATTEND_STEP bool scores(const km::Compute& compute, const Attend& at) {
        const bool room = compute.reserve_back(DC_CB_SCORES, 2 * DC_KEYS_MOST);
        for (uint32_t key = 0; key < at.keys; key++) {
            if (!compute.wait_front(DC_CB_KEYS, DC_HEAD_TILES)) {
                return false;
            }
            for (uint32_t head = 0; head < DC_HEAD_GROUP; head++) {
                tile_regs_acquire();
                if (at.rank == 0 && key >= at.count) {
                    copy(DC_CB_MASK, key - at.count, 0);
                }
                product_init(DC_CB_QUERY, DC_CB_KEYS, 1);
                for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                    matmul_tiles(
                        DC_CB_QUERY, DC_CB_KEYS, head * DC_HEAD_TILES + tile, tile, 0
                    );
                }
                tile_regs_commit();
                tile_regs_wait();
                if (room) {
                    pack_at(0, DC_CB_SCORES, 2 * key + head);
                }
                tile_regs_release();
            }
            cb_pop_front(DC_CB_KEYS, DC_HEAD_TILES);
        }
        const uint32_t left = (DC_KEYS_MOST - at.keys) * DC_HEAD_TILES;
        if (left != 0) {
            if (!compute.wait_front(DC_CB_KEYS, left)) {
                return false;
            }
            cb_pop_front(DC_CB_KEYS, left);
        }
        if (room) {
            cb_push_back(DC_CB_SCORES, 2 * DC_KEYS_MOST);
        }
        return room && compute.wait_front(DC_CB_SCORES, 2 * DC_KEYS_MOST);
    }

    // Head `head`'s state into the item reserved in `cb`: its maximum in every column,
    // the sum of P in every column, P V.
    ATTEND_STEP bool head_state(
        const km::Compute& compute, const Attend& at, uint32_t head, uint32_t cb
    ) {
        const uint32_t state = head * DC_STATE_TILES;
        tile_regs_acquire();  // each row's largest score, in column 0
        reconfig_data_format(DC_CB_SCORES, DC_CB_ATTEND_CONSTANT);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(
            DC_CB_SCORES, DC_CB_ATTEND_CONSTANT, DC_CB_ROW_MAX
        );
        for (uint32_t key = 0; key < at.keys; key++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                DC_CB_SCORES, DC_CB_ATTEND_CONSTANT, 2 * key + head, kOnes, 0
            );
        }
        reduce_uninit();
        if (!decode::pack_one(compute, DC_CB_ROW_MAX)) {
            return false;
        }
        if (!compute.wait_front(DC_CB_ROW_MAX, 1)) {
            return false;
        }
        tile_regs_acquire();  // the maximum in every column: times the row-0 tile
        product_init(DC_CB_ROW_MAX, DC_CB_ATTEND_CONSTANT, 0);
        matmul_tiles(DC_CB_ROW_MAX, DC_CB_ATTEND_CONSTANT, 0, kRow, 0);
        tile_regs_commit();
        tile_regs_wait();
        const bool reference = compute.reserve_back(DC_CB_REFERENCE, 1);
        pack_at(0, cb, state);
        if (reference) {
            pack_reconfig_data_format(DC_CB_REFERENCE);
            pack_tile(0, DC_CB_REFERENCE);
            cb_push_back(DC_CB_REFERENCE, 1);
        }
        tile_regs_release();
        cb_pop_front(DC_CB_ROW_MAX, 1);
        if (!reference || !compute.wait_front(DC_CB_REFERENCE, 1)) {
            return false;
        }
        const bool room = compute.reserve_back(DC_CB_PROBABILITIES, DC_KEYS_MOST);
        for (uint32_t key = 0; key < at.keys; key++) {
            tile_regs_acquire();  // P's rows past the live ones are 0
            zeros(0, at.rows);
            copy(DC_CB_SCORES, 2 * key + head, 1);
            copy(DC_CB_REFERENCE, 0, 2);
            sub_binary_tile_init();
            DC_ROWS(at.rows, rows::sub, 1, 2, 0);
            scaled_exp(0, at.rows);
            if ((at.flags & DC_ATTEND_EXP_TWICE) != 0) {
                scaled_exp(0, at.rows);
            }
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_at(0, DC_CB_PROBABILITIES, key);
            }
            tile_regs_release();
        }
        cb_pop_front(DC_CB_REFERENCE, 1);
        if (!room) {
            return false;
        }
        cb_push_back(DC_CB_PROBABILITIES, DC_KEYS_MOST);
        if (
            !compute.wait_front(DC_CB_PROBABILITIES, DC_KEYS_MOST) ||
            !compute.wait_front(DC_CB_VALUES, DC_KEYS_MOST * DC_HEAD_TILES)
        ) {
            return false;
        }
        tile_regs_acquire();  // the sum of P: P times ones
        product_init(DC_CB_PROBABILITIES, DC_CB_ATTEND_CONSTANT, 0);
        for (uint32_t key = 0; key < at.keys; key++) {
            matmul_tiles(DC_CB_PROBABILITIES, DC_CB_ATTEND_CONSTANT, key, kOnes, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_at(0, cb, state + kSum);
        tile_regs_release();
        tile_regs_acquire();  // P V, the head's 4 output tiles
        product_init(DC_CB_PROBABILITIES, DC_CB_VALUES, 0);
        for (uint32_t key = 0; key < at.keys; key++) {
            for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                matmul_tiles(
                    DC_CB_PROBABILITIES,
                    DC_CB_VALUES,
                    key,
                    key * DC_HEAD_TILES + tile,
                    tile
                );
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
            pack_at(tile, cb, state + kOutput + tile);
        }
        tile_regs_release();
        cb_pop_front(DC_CB_PROBABILITIES, DC_KEYS_MOST);
        return true;
    }

    // A core without keys: a state of zeros (a maximum of 0 and nothing summed).
    ATTEND_STEP void zero_state(uint32_t cb) {
        for (uint32_t first = 0; first < DC_STATE_ITEM; first += 4) {
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t tile = 0; tile < 4; tile++) {
                fill_tile(tile, 0.0f);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < 4; tile++) {
                pack_at(tile, cb, first + tile);
            }
            tile_regs_release();
        }
    }

    // The front state and slot `slot`'s merged into `cb`: m = max(m1, m2), sum and
    // output diag(a1) X1 + diag(a2) X2 (matrix unit), a_i = exp((m_i - m) / sqrt(128)).
    ATTEND_STEP bool merge(
        const km::Compute& compute,
        const km::Arrival& arrival,
        uint32_t slot,
        uint32_t cb,
        uint32_t live
    ) {
        const uint32_t base = km::slot_page(arrival, slot);
        if (
            !compute.wait_front(DC_CB_STATE, DC_STATE_ITEM) ||
            !compute.reserve_back(cb, DC_STATE_ITEM)
        ) {
            return false;
        }
        for (uint32_t head = 0; head < DC_HEAD_GROUP; head++) {
            const uint32_t state = head * DC_STATE_TILES;
            tile_regs_acquire();  // m, then each side's factor on the diagonal
            copy(DC_CB_STATE, state, 0);
            copy(kCbArrive, base + state, 1);
            copy(DC_CB_ATTEND_CONSTANT, kIdentity, 3);
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
            const bool rescale = compute.reserve_back(DC_CB_RESCALE, 2);
            pack_at(2, cb, state);
            if (rescale) {
                pack_reconfig_data_format(DC_CB_RESCALE);
                pack_tile(0, DC_CB_RESCALE);
                pack_tile(1, DC_CB_RESCALE);
                cb_push_back(DC_CB_RESCALE, 2);
            }
            tile_regs_release();
            if (!rescale || !compute.wait_front(DC_CB_RESCALE, 2)) {
                return false;
            }
            rescale_init();
            for (uint32_t first = kSum; first < DC_STATE_TILES; first += 4) {
                const uint32_t last = (
                    first + 4 < DC_STATE_TILES ? first + 4 : DC_STATE_TILES
                );
                tile_regs_acquire();
                for (uint32_t tile = first; tile < last; tile++) {
                    rescale_tile(0, DC_CB_STATE, state + tile, tile - first);
                    rescale_tile(1, kCbArrive, base + state + tile, tile - first);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t tile = first; tile < last; tile++) {
                    pack_at(tile - first, cb, state + tile);
                }
                tile_regs_release();
            }
            cb_pop_front(DC_CB_RESCALE, 2);
        }
        cb_push_back(cb, DC_STATE_ITEM);
        cb_pop_front(DC_CB_STATE, DC_STATE_ITEM);
        return true;
    }

    // The root: each output tile over its head's sum on the live rows (the others 0),
    // in bf16 for the writer.
    ATTEND_STEP bool finish(const km::Compute& compute, uint32_t live) {
        if (
            !compute.wait_front(DC_CB_STATE, DC_STATE_ITEM) ||
            !compute.reserve_back(kCbSplitOut, kQueryTiles)
        ) {
            return false;
        }
        for (uint32_t head = 0; head < DC_HEAD_GROUP; head++) {
            const uint32_t state = head * DC_STATE_TILES;
            for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                tile_regs_acquire();
                zeros(0, live);
                copy(DC_CB_STATE, state + kOutput + tile, 1);
                copy(DC_CB_STATE, state + kSum, 2);
                div_binary_tile_init();
                DC_ROWS(live, rows::div, 1, 2, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_at(0, kCbSplitOut, head * DC_HEAD_TILES + tile);
                tile_regs_release();
            }
        }
        cb_push_back(kCbSplitOut, kQueryTiles);
        cb_pop_front(DC_CB_STATE, DC_STATE_ITEM);
        return true;
    }

    // Read only: every key and value page taken and dropped.
    ATTEND_STEP bool drop(const km::Compute& compute) {
        const uint32_t pages = DC_KEYS_MOST * DC_HEAD_TILES;
        if (
            !compute.wait_front(DC_CB_KEYS, pages) ||
            !compute.wait_front(DC_CB_VALUES, pages)
        ) {
            return false;
        }
        cb_pop_front(DC_CB_KEYS, pages);
        cb_pop_front(DC_CB_VALUES, pages);
        return true;
    }

    // The rounds in which this core receives a state (none without the merge).
    ATTEND_STEP uint32_t receives(const Attend& at, uint32_t form) {
        uint32_t rounds = 0;
        for (uint32_t port = 0; merges(at) && port < KM_PORTS / 2; port++) {
            rounds += km::reduce_wants(at.group, form, port) != 0 ? 1u : 0u;
        }
        return rounds;
    }

    // The state, then each received one merged in; a core other than the root leaves
    // its last in DC_CB_OUTBOX for its writer, the root its output in Y's buffer.
    ATTEND_STEP bool attend(
        const km::Compute& compute, const SplitBlock& block, const Attend& at
    ) {
        const bool root = at.rank == 0;
        const uint32_t form = split_form(*block.command);
        const uint32_t rounds = receives(at, form);
        if (
            !compute.wait_front(DC_CB_QUERY, kQueryTiles) ||
            !compute.wait_front(DC_CB_ATTEND_CONSTANT, kConstants) ||
            (root && !compute.wait_front(DC_CB_MASK, 2))
        ) {
            return false;
        }
        bool ok = true;
        if ((at.flags & DC_ATTEND_READ_ONLY) != 0) {
            ok = drop(compute);
        } else {
            const uint32_t own = rounds == 0 && !root ? DC_CB_OUTBOX : DC_CB_STATE;
            ok = scores(compute, at) && compute.reserve_back(own, DC_STATE_ITEM);
            for (
                uint32_t head = 0; ok && at.keys != 0 && head < DC_HEAD_GROUP; head++
            ) {
                ok = head_state(compute, at, head, own);
            }
            if (ok && at.keys == 0) {
                zero_state(own);
            }
            const uint32_t pages = DC_KEYS_MOST * DC_HEAD_TILES;
            ok = ok && compute.wait_front(DC_CB_VALUES, pages);
            if (ok) {
                cb_pop_front(DC_CB_SCORES, 2 * DC_KEYS_MOST);
                cb_pop_front(DC_CB_VALUES, pages);
                cb_push_back(own, DC_STATE_ITEM);
            }
        }
        const km::Arrival arrival = attend_arrival(compute.arg(block.plan + kSpArrive));
        for (uint32_t port = 0, round = 0; ok && round < rounds; port++) {
            if (km::reduce_wants(at.group, form, port) == 0) {
                continue;
            }
            round++;
            const uint32_t into = round == rounds && !root ? DC_CB_OUTBOX : DC_CB_STATE;
            if (!km::take(compute, arrival)) {
                ok = false;
                break;
            }
            ok = merge(compute, arrival, port, into, at.rows);
            km::untake(arrival);
        }
        if (ok && root && (at.flags & DC_ATTEND_READ_ONLY) == 0) {
            ok = finish(compute, at.rows);
        }
        cb_pop_front(DC_CB_QUERY, kQueryTiles);
        cb_pop_front(DC_CB_ATTEND_CONSTANT, kConstants);
        if (root) {
            cb_pop_front(DC_CB_MASK, 2);
        }
        return ok;
    }

}  // namespace

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    const Attend at = attend_of(compute, block);
    const bool ok = attend(compute, block, at);
    reconfig_data_format(kCbSw, kCbSa);  // as the K split's product leaves them
    return ok ? 1 : 0;
}
