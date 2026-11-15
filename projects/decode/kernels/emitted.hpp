// The operations' compute on emitted vector bodies (bodies.py, kernels/generated):
// tiles copied into DEST with their own buffer's format, a body, a pack.
#pragma once

#include <stdint.h>

#include "generated/add.hpp"
#include "generated/gate.hpp"
#include "generated/mad.hpp"
#include "generated/mul.hpp"
#include "generated/mul3.hpp"
#include "generated/rms_head.hpp"
#include "generated/rms_row.hpp"
#include "generated/square.hpp"
#include "generated/square_add.hpp"
#include "generated/squares3.hpp"
#include "generated/sum3.hpp"
#include "ops.hpp"

// The matrix unit waits for the vector unit: a copy after a body may overwrite a tile
// the body still reads (an unpack to DEST waits already).
#define DC_VECTOR_DONE() MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)))

// Emitted body `name` on DEST tiles dst, dst + 1, ... and the first `rows` rows (the
// live ones, in bands of 4): its init, the body, the wait.
#define DC_BODY(name, dst, rows)          \
    do {                                  \
        MATH((km_tile::name##_init()));   \
        MATH((km_tile::name(dst, rows))); \
        DC_VECTOR_DONE();                 \
    } while (0)

namespace decode::emitted {

    // Unpack A's format follows the buffer last copied from; it starts and ends as
    // kCbSum's (the K split leaves it so around the hook).
    struct Copies {
        uint32_t buffer = kCbSum;
        bool init = false;

        void copy(uint32_t from, uint32_t tile, uint32_t dst) {
            if (from != buffer || !init) {
                reconfig_data_format_srca(buffer, from);
                copy_init(from);
                buffer = from;
                init = true;
            }
            copy_tile(from, tile, dst);
        }

        void restore() {
            reconfig_data_format_srca(buffer, kCbSum);
            buffer = kCbSum;
            init = false;
        }
    };

    // DEST 0 (and 1 when `two`) into tile `index` (and `second`) of `buffer`, which
    // the caller reserved; ends the section.
    inline void pack_at(uint32_t buffer, uint32_t index, bool room) {
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(buffer);
        if (room) {
            pack_tile<true>(0, buffer, index);
        }
        tile_regs_release();
    }

    inline void pack_two(uint32_t buffer, uint32_t index, uint32_t second, bool room) {
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(buffer);
        if (room) {
            pack_tile<true>(0, buffer, index);
            pack_tile<true>(1, buffer, second);
        }
        tile_regs_release();
    }

    // DC_CB_SCALE (fp32): each live row's 1 / sqrt(mean square + epsilon) in every
    // column, from DC_CB_PARTS (their row sums add up to the row's sum of squares).
    inline bool rms_scale(const km::Compute& compute, uint32_t live) {
        if (!compute.wait_front(DC_CB_PARTS, DC_PARTS)) {
            return false;
        }
        Copies copies;
        tile_regs_acquire();
        copies.copy(DC_CB_PARTS, 0, 0);
        uint32_t part = 1;
        for (; part + 1 < DC_PARTS; part += 2) {
            copies.copy(DC_CB_PARTS, part, 1);
            copies.copy(DC_CB_PARTS, part + 1, 2);
            DC_BODY(sum3, 0, live);
        }
        if (part < DC_PARTS) {
            copies.copy(DC_CB_PARTS, part, 1);
            DC_BODY(add, 0, live);
        }
        DC_BODY(rms_row, 0, live);
        const bool room = compute.reserve_back(DC_CB_SCALE, 1);
        pack_at(DC_CB_SCALE, 0, room);
        if (room) {
            cb_push_back(DC_CB_SCALE, 1);
        }
        copies.restore();
        cb_pop_front(DC_CB_PARTS, DC_PARTS);
        return room && compute.wait_front(DC_CB_SCALE, 1);
    }

    // Tiles `first` .. `first + tiles - 1` of the sum times DC_CB_SCALE into tiles 0 ..
    // of `out` (reserved by the caller), on the live rows.
    inline void scale_tiles(
        uint32_t first, uint32_t tiles, uint32_t out, bool room, uint32_t live
    ) {
        Copies copies;
        for (uint32_t tile = 0; tile < tiles; tile++) {
            tile_regs_acquire();
            copies.copy(kCbSum, first + tile, 0);
            copies.copy(DC_CB_SCALE, 0, 1);
            DC_BODY(mul, 0, live);
            pack_at(out, tile, room);
        }
        copies.restore();
    }

    // Y = the sum + DC_CB_BLOCK into kCbSplitOut and DC_CB_Y (bf16); then the block's
    // squares into DC_CB_PART, a tile whose row sums are the rows' sums of squares.
    inline bool residual_add(
        const km::Compute& compute, uint32_t tiles, uint32_t live
    ) {
        if (
            !compute.wait_front(kCbSum, tiles) ||
            !compute.wait_front(DC_CB_BLOCK, tiles)
        ) {
            return false;
        }
        const bool out = compute.reserve_back(kCbSplitOut, tiles);
        const bool again = compute.reserve_back(DC_CB_Y, tiles);
        Copies copies;
        for (uint32_t tile = 0; tile < tiles; tile++) {
            tile_regs_acquire();
            copies.copy(kCbSum, tile, 0);
            copies.copy(DC_CB_BLOCK, tile, 1);
            DC_BODY(add, 0, live);
            tile_regs_commit();
            tile_regs_wait();
            if (out && again) {
                pack_reconfig_data_format(kCbSplitOut);
                pack_tile<true>(0, kCbSplitOut, tile);
                pack_tile<true>(0, DC_CB_Y, tile);
            }
            tile_regs_release();
        }
        if (!out || !again) {
            copies.restore();
            return false;
        }
        cb_push_back(kCbSplitOut, tiles);
        cb_push_back(DC_CB_Y, tiles);
        cb_pop_front(DC_CB_BLOCK, tiles);
        cb_pop_front(kCbSum, tiles);
        if (!compute.wait_front(DC_CB_Y, tiles)) {
            copies.restore();
            return false;
        }
        tile_regs_acquire();
        copies.copy(DC_CB_Y, 0, 0);
        DC_BODY(square, 0, live);
        if (tiles > 1) {
            MATH((km_tile::square_add_init()));
        }
        for (uint32_t tile = 1; tile < tiles; tile++) {
            copies.copy(DC_CB_Y, tile, 1);
            MATH((km_tile::square_add(0, live)));
            DC_VECTOR_DONE();
        }
        const bool part = compute.reserve_back(DC_CB_PART, 1);
        pack_at(DC_CB_PART, 0, part);
        if (part) {
            cb_push_back(DC_CB_PART, 1);
        }
        copies.restore();
        cb_pop_front(DC_CB_Y, tiles);
        return part;
    }

    // A gate_up block: up's half times DC_CB_SCALE into DC_CB_UP (bf16, as the
    // reference rounds u), then silu(gate's half times it) times u into kCbSplitOut.
    inline bool gate(const km::Compute& compute, uint32_t tiles, uint32_t live) {
        const uint32_t half = tiles / 2;
        const bool up = compute.reserve_back(DC_CB_UP, tiles);  // whole items
        scale_tiles(half, half, DC_CB_UP, up, live);
        if (up) {
            cb_push_back(DC_CB_UP, tiles);
        }
        if (!compute.wait_front(DC_CB_UP, tiles)) {
            return false;
        }
        const bool out = compute.reserve_back(kCbSplitOut, tiles);
        Copies copies;
        for (uint32_t tile = 0; tile < half; tile++) {
            tile_regs_acquire();
            copies.copy(kCbSum, tile, 0);
            copies.copy(DC_CB_SCALE, 0, 1);
            copies.copy(DC_CB_UP, tile, 2);
            DC_BODY(gate, 0, live);
            pack_at(kCbSplitOut, tile, out);
        }
        copies.restore();
        if (out) {
            cb_push_back(kCbSplitOut, tiles);
        }
        cb_pop_front(DC_CB_UP, tiles);
        return out && up;
    }

    // Head `head` of DC_CB_Y: its rows' 1 / rms into DC_CB_SCALE, each tile times it
    // and the weight into DC_CB_NORMED, then n_j c_j + n_(j+2)%4 s_j into kCbSplitOut.
    inline bool head(
        const km::Compute& compute, uint32_t head, bool room, uint32_t live
    ) {
        constexpr uint32_t kHalf = DC_HEAD_TILES / 2;
        const uint32_t base = head * DC_HEAD_TILES;
        Copies copies;
        tile_regs_acquire();
        copies.copy(DC_CB_Y, base, 0);
        copies.copy(DC_CB_Y, base + 1, 1);
        copies.copy(DC_CB_Y, base + 2, 2);
        DC_BODY(squares3, 0, live);
        copies.copy(DC_CB_Y, base + 3, 1);
        DC_BODY(square_add, 0, live);
        DC_BODY(rms_head, 0, live);
        const bool scaled = compute.reserve_back(DC_CB_SCALE, 1);
        pack_at(DC_CB_SCALE, 0, scaled);
        if (!scaled) {
            copies.restore();
            return false;
        }
        cb_push_back(DC_CB_SCALE, 1);
        const bool normed = compute.reserve_back(DC_CB_NORMED, DC_HEAD_TILES);
        if (!compute.wait_front(DC_CB_SCALE, 1)) {
            copies.restore();
            return false;
        }
        for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile += 2) {
            // n_j = y_j w_j r in DEST 0 from tiles (0, 1, 2), n_j+1 in 1 from (1, 2, 3)
            tile_regs_acquire();
            copies.copy(DC_CB_Y, base + tile, 0);
            copies.copy(DC_CB_HEAD, 1 + tile, 1);
            copies.copy(DC_CB_SCALE, 0, 2);
            DC_BODY(mul3, 0, live);
            copies.copy(DC_CB_Y, base + tile + 1, 1);
            copies.copy(DC_CB_HEAD, 2 + tile, 3);
            DC_BODY(mul3, 1, live);
            pack_two(DC_CB_NORMED, tile, tile + 1, normed);
        }
        cb_pop_front(DC_CB_SCALE, 1);
        if (!normed) {
            copies.restore();
            return false;
        }
        cb_push_back(DC_CB_NORMED, DC_HEAD_TILES);
        if (!compute.wait_front(DC_CB_NORMED, DC_HEAD_TILES)) {
            copies.restore();
            return false;
        }
        for (uint32_t tile = 0; tile < kHalf; tile++) {
            // tiles j and k = j + 2: n_k s_j + n_j c_j in DEST 0, n_j s_k + n_k c_k in
            // 1
            const uint32_t other = tile + kHalf;
            tile_regs_acquire();
            copies.copy(DC_CB_NORMED, other, 0);
            copies.copy(DC_CB_ROPE, DC_HEAD_TILES + tile, 1);
            DC_BODY(mul, 0, live);
            copies.copy(DC_CB_NORMED, tile, 1);
            copies.copy(DC_CB_ROPE, tile, 2);
            DC_BODY(mad, 0, live);
            copies.copy(DC_CB_ROPE, DC_HEAD_TILES + other, 2);
            DC_BODY(mul, 1, live);
            copies.copy(DC_CB_NORMED, other, 2);
            copies.copy(DC_CB_ROPE, other, 3);
            DC_BODY(mad, 1, live);
            pack_two(kCbSplitOut, base + tile, base + other, room);
        }
        cb_pop_front(DC_CB_NORMED, DC_HEAD_TILES);
        copies.restore();
        return true;
    }

}  // namespace decode::emitted
