// Scale, compute: Y = the sum times each row's 1 / rms (the norm's weight is in W's
// rows), bf16; on q's and k's blocks then each head's norm and rotation (DC_QK).
#include "../ops.hpp"
#if DC_EMIT_NORM || DC_EMIT_QK
#include "../emitted.hpp"
#endif

#if !defined(DC_EPSILON)
#error "DC_EPSILON: the norms' epsilon (float bits), the overlay's define"
#endif

// The sum's tiles times DC_CB_SCALE into `out` (reserved and pushed here).
static bool scaled(
    km::Compute& compute, uint32_t tiles, uint32_t out, [[maybe_unused]] uint32_t live
) {
#if DC_EMIT_NORM
    const bool room = compute.reserve_back(out, tiles);
    decode::emitted::scale_tiles(0, tiles, out, room, live);
#else
    const bool room = compute.reserve_back(out, tiles);
    pack_reconfig_data_format(out);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_init(kCbSum);
        copy_tile(kCbSum, tile, 0);
        decode::multiply_init(DC_CB_SCALE);
        decode::multiply(DC_CB_SCALE, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (room) {
            pack_tile(0, out);
        }
        tile_regs_release();
    }
#endif
    if (room) {
        cb_push_back(out, tiles);
    }
    cb_pop_front(DC_CB_SCALE, 1);
    cb_pop_front(kCbSum, tiles);
    return room;
}

#if !DC_EMIT_QK
// Head `head` of the scaled tiles (DC_CB_Y): its rows' 1 / rms over its 128 columns
// into DC_CB_SCALE, then each tile times it and the norm's weight into DC_CB_NORMED.
static bool head_norm(km::Compute& compute, uint32_t head, uint32_t live) {
    const uint32_t base = head * DC_HEAD_TILES;
    tile_regs_acquire();
    copy_init(DC_CB_Y);
    copy_tile(DC_CB_Y, base, 0);
    mul_binary_tile_init();
    DC_ROWS(live, rows::mul, 0, 0, 0);
    for (uint32_t tile = 1; tile < DC_HEAD_TILES; tile++) {
        copy_init(DC_CB_Y);
        copy_tile(DC_CB_Y, base + tile, 1);
        mul_binary_tile_init();
        DC_ROWS(live, rows::mul, 1, 1, 1);
        add_binary_tile_init();
        DC_ROWS(live, rows::add, 0, 1, 0);
    }
    if (
        !decode::pack_one(compute, DC_CB_SQUARES) ||
        !compute.wait_front(DC_CB_SQUARES, 1)
    ) {
        return false;
    }
    tile_regs_acquire();
    matmul_init(DC_CB_SQUARES, DC_CB_HEAD, 0);
    matmul_tiles(DC_CB_SQUARES, DC_CB_HEAD, 0, 0, 0);
    binop_with_scalar_tile_init();
    DC_ROWS(live, rows::plus, 0, DC_EPSILON);
    rsqrt_tile_init();
    DC_ROWS(live, rows::rsqrt, 0);
    cb_pop_front(DC_CB_SQUARES, 1);
    if (
        !decode::pack_one(compute, DC_CB_SCALE) || !compute.wait_front(DC_CB_SCALE, 1)
    ) {
        return false;
    }
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
        copy_init(DC_CB_Y);
        copy_tile(DC_CB_Y, base + tile, tile);
        decode::multiply_init(DC_CB_SCALE);
        decode::multiply(DC_CB_SCALE, 0, tile);
        decode::multiply_init(DC_CB_HEAD);
        decode::multiply(DC_CB_HEAD, 1 + tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    const bool room = compute.reserve_back(DC_CB_NORMED, DC_HEAD_TILES);
    pack_reconfig_data_format(DC_CB_NORMED);
    for (uint32_t tile = 0; room && tile < DC_HEAD_TILES; tile++) {
        pack_tile(tile, DC_CB_NORMED);
    }
    tile_regs_release();
    cb_pop_front(DC_CB_SCALE, 1);
    if (room) {
        cb_push_back(DC_CB_NORMED, DC_HEAD_TILES);
    }
    return room && compute.wait_front(DC_CB_NORMED, DC_HEAD_TILES);
}

// The normed head rotated: tile j times cos_j, plus tile (j + 2) % 4 times the signed
// sin_j (rotate_half's halves), on the live rows; packed in turn into kCbSplitOut.
static void rotated(uint32_t live, bool room) {
    for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
        tile_regs_acquire();
        copy_init(DC_CB_NORMED);
        copy_tile(DC_CB_NORMED, tile, 0);
        copy_tile(DC_CB_NORMED, (tile + 2) % DC_HEAD_TILES, 2);
        copy_init(DC_CB_ROPE);
        copy_tile(DC_CB_ROPE, tile, 1);
        copy_tile(DC_CB_ROPE, DC_HEAD_TILES + tile, 3);
        mul_binary_tile_init();
        DC_ROWS(live, rows::mul, 0, 1, 0);
        DC_ROWS(live, rows::mul, 2, 3, 2);
        add_binary_tile_init();
        DC_ROWS(live, rows::add, 0, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (room) {
            pack_tile(0, kCbSplitOut);
        }
        tile_regs_release();
    }
    cb_pop_front(DC_CB_NORMED, DC_HEAD_TILES);
}
#endif

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    [[maybe_unused]] const uint32_t live = decode::live_rows(*block.command);
    const uint32_t tiles = (
        compute.arg(block.plan + kSpOh) * compute.arg(block.plan + kSpOw)
    );
#if DC_EMIT_NORM
    if (
        !compute.wait_front(kCbSum, tiles) || !decode::emitted::rms_scale(compute, live)
    ) {
        return 0;
    }
#else
    if (
        !compute.wait_front(kCbSum, tiles) ||
        !decode::rms_scale(compute, DC_EPSILON, live)
    ) {
        return 0;
    }
#endif
    if (!decode::qk_block(compute, block)) {
        return scaled(compute, tiles, kCbSplitOut, live) ? 1 : 0;
    }
    if (
        !scaled(compute, tiles, DC_CB_Y, live) || !compute.wait_front(DC_CB_Y, tiles) ||
        !compute.wait_front(DC_CB_HEAD, DC_HEAD_ITEM) ||
        !compute.wait_front(DC_CB_ROPE, 2 * DC_HEAD_TILES)
    ) {
        return 0;
    }
    const bool out = compute.reserve_back(kCbSplitOut, tiles);
    bool done = true;
#if DC_EMIT_QK
    for (uint32_t head = 0; done && head < tiles / DC_HEAD_TILES; head++) {
        done = decode::emitted::head(compute, head, out, live);
    }
#else
    reconfig_data_format_srca(kCbSum, DC_CB_Y);
    for (uint32_t head = 0; done && head < tiles / DC_HEAD_TILES; head++) {
        done = head_norm(compute, head, live);
        if (done) {
            pack_reconfig_data_format(kCbSplitOut);
            rotated(live, out);
        }
    }
    reconfig_data_format_srca(DC_CB_Y, kCbSum);
#endif
    cb_pop_front(DC_CB_ROPE, 2 * DC_HEAD_TILES);
    cb_pop_front(DC_CB_HEAD, DC_HEAD_ITEM);
    cb_pop_front(DC_CB_Y, tiles);
    if (out && done) {
        cb_push_back(kCbSplitOut, tiles);
    }
    return out && done ? 1 : 0;
}
