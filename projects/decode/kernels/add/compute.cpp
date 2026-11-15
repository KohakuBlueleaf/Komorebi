// Add, compute: Y = the sum + the block (the residual), each tile in DEST in fp32,
// packed in bf16 for the root's writer; then the rows' squares of Y for the next norm.
#include "../ops.hpp"

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    const uint32_t tiles = (
        compute.arg(block.plan + kSpOh) * compute.arg(block.plan + kSpOw)
    );
    if (!compute.wait_front(kCbSum, tiles) || !compute.wait_front(DC_CB_BLOCK, tiles)) {
        return 0;
    }
    const bool out = compute.reserve_back(kCbSplitOut, tiles);
    const bool again = compute.reserve_back(DC_CB_Y, tiles);
    pack_reconfig_data_format(kCbSplitOut);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_init(kCbSum);
        copy_tile(kCbSum, tile, 0);
        add_reuse_dest_init<decode::kReuse>(DC_CB_BLOCK);
        add_reuse_dest_tiles<decode::kReuse>(DC_CB_BLOCK, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (out && again) {
            pack_tile(0, kCbSplitOut);
            pack_tile(0, DC_CB_Y);
        }
        tile_regs_release();
    }
    if (!out || !again) {
        return 0;
    }
    cb_push_back(kCbSplitOut, tiles);
    cb_push_back(DC_CB_Y, tiles);
    cb_pop_front(DC_CB_BLOCK, tiles);
    cb_pop_front(kCbSum, tiles);
    // the block's sums of squares: sum of y y^T over its tiles, the diagonal kept
    if (
        !compute.wait_front(DC_CB_Y, tiles) ||
        !compute.wait_front(DC_CB_CONSTANT, DC_CONSTANTS)
    ) {
        return 0;
    }
    reconfig_data_format_srca(kCbSum, DC_CB_Y);
    tile_regs_acquire();
    matmul_init(DC_CB_Y, DC_CB_Y, 1);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        matmul_tiles(DC_CB_Y, DC_CB_Y, tile, tile, 0);
    }
    decode::multiply_init(DC_CB_CONSTANT);
    decode::multiply(DC_CB_CONSTANT, DC_CONSTANT_IDENTITY, 0);
    const bool part = decode::pack_one(compute, DC_CB_PART);
    cb_pop_front(DC_CB_Y, tiles);
    cb_pop_front(DC_CB_CONSTANT, DC_CONSTANTS);
    reconfig_data_format_srca(DC_CB_Y, kCbSum);
    return part ? 1 : 0;
}
