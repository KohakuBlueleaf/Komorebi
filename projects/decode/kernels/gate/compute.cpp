// Gate, compute: Y = silu(the sum) * the block (the up product), each tile in DEST in
// fp32, packed in bf16 for the root's writer.
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "../ops.hpp"

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    const uint32_t tiles = (
        compute.arg(block.plan + kSpOh) * compute.arg(block.plan + kSpOw)
    );
    if (!compute.wait_front(kCbSum, tiles) || !compute.wait_front(DC_CB_BLOCK, tiles)) {
        return 0;
    }
    const bool out = compute.reserve_back(kCbSplitOut, tiles);
    pack_reconfig_data_format(kCbSplitOut);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_init(kCbSum);
        copy_tile(kCbSum, tile, 0);
        silu_tile_init();
        silu_tile(0);
        decode::multiply_init(DC_CB_BLOCK);
        decode::multiply(DC_CB_BLOCK, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (out) {
            pack_tile(0, kCbSplitOut);
        }
        tile_regs_release();
    }
    if (out) {
        cb_push_back(kCbSplitOut, tiles);
    }
    cb_pop_front(DC_CB_BLOCK, tiles);
    cb_pop_front(kCbSum, tiles);
    return out ? 1 : 0;
}
