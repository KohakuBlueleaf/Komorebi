// Scale, compute: the norm after the product (its weight is in W's rows): Y = the sum
// times each row's 1 / rms, packed in bf16 for the root's writer.
#include "../ops.hpp"

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const uint32_t tiles = (
        compute.arg(block.plan + kSpOh) * compute.arg(block.plan + kSpOw)
    );
    if (
        !compute.wait_front(kCbSum, tiles) ||
        !decode::rms_scale(compute, command.a9, decode::live_rows(command))
    ) {
        return 0;
    }
    const bool out = compute.reserve_back(kCbSplitOut, tiles);
    pack_reconfig_data_format(kCbSplitOut);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_init(kCbSum);
        copy_tile(kCbSum, tile, 0);
        decode::multiply_init(DC_CB_SCALE);
        decode::multiply(DC_CB_SCALE, 0, 0);
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
    cb_pop_front(DC_CB_SCALE, 1);
    cb_pop_front(kCbSum, tiles);
    return out ? 1 : 0;
}
