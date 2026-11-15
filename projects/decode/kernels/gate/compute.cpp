// Gate, compute: a gate_up block's sum is GATE_TILES tiles of the gate, then as many of
// up, each times 1 / rms (the norm); out = silu(gate) * up, up rounded to bf16 first.
#include "api/compute/compute_kernel_api.h"
#include "../ops.hpp"
#if DC_EMIT_NORM || DC_EMIT_GATE
#include "../emitted.hpp"
#endif

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    const uint32_t tiles = (
        compute.arg(block.plan + kSpOh) * compute.arg(block.plan + kSpOw)
    );
    [[maybe_unused]] const uint32_t half = tiles / 2;
    [[maybe_unused]] const uint32_t live = decode::live_rows(*block.command);
#if DC_EMIT_NORM
    if (!compute.wait_front(kCbSum, tiles) || !decode::emitted::rms_scale(compute)) {
        return 0;
    }
#else
    if (
        !compute.wait_front(kCbSum, tiles) ||
        !decode::rms_scale(compute, block.command->a9, live)
    ) {
        return 0;
    }
#endif
#if DC_EMIT_GATE
    const bool gated = decode::emitted::gate(compute, tiles);
    cb_pop_front(DC_CB_SCALE, 1);
    cb_pop_front(kCbSum, tiles);
    return gated ? 1 : 0;
#else
    const bool up = compute.reserve_back(DC_CB_UP, tiles);  // whole items: tiles pages
    pack_reconfig_data_format(DC_CB_UP);
    for (uint32_t tile = 0; tile < half; tile++) {
        tile_regs_acquire();
        copy_init(kCbSum);
        copy_tile(kCbSum, half + tile, 0);
        decode::multiply_init(DC_CB_SCALE);
        decode::multiply(DC_CB_SCALE, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (up) {
            pack_tile<true>(0, DC_CB_UP, tile);
        }
        tile_regs_release();
    }
    if (up) {
        cb_push_back(DC_CB_UP, tiles);
    }
    if (!compute.wait_front(DC_CB_UP, tiles)) {
        return 0;
    }
    const bool out = compute.reserve_back(kCbSplitOut, tiles);
    pack_reconfig_data_format(kCbSplitOut);
    for (uint32_t tile = 0; tile < half; tile++) {
        tile_regs_acquire();
        copy_init(kCbSum);
        copy_tile(kCbSum, tile, 0);
        decode::multiply_init(DC_CB_SCALE);
        decode::multiply(DC_CB_SCALE, 0, 0);
        silu_tile_init();
        DC_ROWS(live, rows::silu, 0);
        decode::multiply_init(DC_CB_UP);
        decode::multiply(DC_CB_UP, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (out) {
            pack_tile<true>(0, kCbSplitOut, tile);
        }
        tile_regs_release();
    }
    if (out) {
        cb_push_back(kCbSplitOut, tiles);
    }
    cb_pop_front(DC_CB_UP, tiles);
    cb_pop_front(DC_CB_SCALE, 1);
    cb_pop_front(kCbSum, tiles);
    return out && up ? 1 : 0;
#endif
}
