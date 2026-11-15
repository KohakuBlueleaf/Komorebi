// Attend, compute: each query head's scores against this core's keys, their largest
// in each row, P = exp((s - max) / sqrt(128)), its row sums and P V: the head's state.
// The states meet up the group's tree; the root divides the output by the sum.
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

    constexpr uint32_t kOnes = 0;  // DC_CB_ATTEND_CONSTANT: the ones, the row-0 tile
    constexpr uint32_t kRow = 1;
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

    // DEST `dst` = exp(DEST `dst` / sqrt(128)).
    ATTEND_STEP void scaled_exp(uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, DC_SCALE_BITS);
        exp_tile_init<false>();
        exp_tile<false>(dst);
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
                if (at.rank == 0 && key + 1 == at.keys) {
                    copy(DC_CB_MASK, 0, 0);
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
            tile_regs_acquire();
            copy(DC_CB_SCORES, 2 * key + head, 0);
            copy(DC_CB_REFERENCE, 0, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            scaled_exp(0);
            if ((at.flags & DC_ATTEND_EXP_TWICE) != 0) {
                scaled_exp(0);
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

    // The state at the front and the one in arrival slot `slot`, merged into an item
    // of `cb`: m = max(m1, m2), each sum and output tile times exp((m_i - m) /
    // sqrt(128)).
    ATTEND_STEP bool merge(
        const km::Compute& compute,
        const km::Arrival& arrival,
        uint32_t slot,
        uint32_t cb
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
            tile_regs_acquire();
            copy(DC_CB_STATE, state, 0);
            copy(kCbArrive, base + state, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 2);
            sub_binary_tile_init();
            sub_binary_tile(0, 2, 0);
            sub_binary_tile(1, 2, 1);
            scaled_exp(0);
            scaled_exp(1);
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
            for (uint32_t tile = kSum; tile < DC_STATE_TILES; tile++) {
                tile_regs_acquire();
                copy(DC_CB_RESCALE, 0, 0);
                copy(DC_CB_RESCALE, 1, 1);
                copy(DC_CB_STATE, state + tile, 2);
                copy(kCbArrive, base + state + tile, 3);
                mul_binary_tile_init();
                mul_binary_tile(2, 0, 2);
                mul_binary_tile(3, 1, 3);
                add_binary_tile_init();
                add_binary_tile(2, 3, 2);
                tile_regs_commit();
                tile_regs_wait();
                pack_at(2, cb, state + tile);
                tile_regs_release();
            }
            cb_pop_front(DC_CB_RESCALE, 2);
        }
        cb_push_back(cb, DC_STATE_ITEM);
        cb_pop_front(DC_CB_STATE, DC_STATE_ITEM);
        return true;
    }

    // The root: each output tile over its head's sum, bf16, for the writer.
    ATTEND_STEP bool finish(const km::Compute& compute) {
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
                copy(DC_CB_STATE, state + kOutput + tile, 0);
                copy(DC_CB_STATE, state + kSum, 1);
                div_binary_tile_init();
                div_binary_tile(0, 1, 0);
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

    // The state, then each state received merged into it; a core other than the root
    // leaves its last in DC_CB_OUTBOX for its writer, the root its output in Y's
    // buffer.
    ATTEND_STEP bool attend(
        const km::Compute& compute, const SplitBlock& block, const Attend& at
    ) {
        const bool root = at.rank == 0;
        const uint32_t form = split_form(*block.command);
        const uint32_t rounds = receives(at, form);
        if (
            !compute.wait_front(DC_CB_QUERY, kQueryTiles) ||
            !compute.wait_front(DC_CB_ATTEND_CONSTANT, 2) ||
            (root && !compute.wait_front(DC_CB_MASK, 1))
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
            ok = merge(compute, arrival, port, into);
            km::untake(arrival);
        }
        if (ok && root && (at.flags & DC_ATTEND_READ_ONLY) == 0) {
            ok = finish(compute);
        }
        cb_pop_front(DC_CB_QUERY, kQueryTiles);
        cb_pop_front(DC_CB_ATTEND_CONSTANT, 2);
        if (root) {
            cb_pop_front(DC_CB_MASK, 1);
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
