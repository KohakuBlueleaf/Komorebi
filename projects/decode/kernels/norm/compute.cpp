// Norm, compute: S = sum of x x^T over the row (squares on its diagonal), diag(S) J
// (each row's mean square), 1 / sqrt(that + eps), then this core's tiles x it x weight.
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "../ops.hpp"

namespace {

    // DEST 0 into one page of `buffer` (reserved and pushed here).
    bool pack_one(const km::Compute& compute, uint32_t buffer) {
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(buffer, 1);
        pack_reconfig_data_format(buffer);
        if (room) {
            pack_tile(0, buffer);
            cb_push_back(buffer, 1);
        }
        tile_regs_release();
        return room;
    }

}  // namespace

uint32_t km_overlay(km::Compute& compute, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const uint32_t first = (compute.me() - km::first(command)) * DC_NORM_SHARE;
    if (
        !compute.wait_front(DC_CB_CONSTANT, DC_CONSTANTS) ||
        !compute.wait_front(DC_CB_ROW, DC_ROW_TILES) ||
        !compute.wait_front(DC_CB_WEIGHT, DC_NORM_SHARE)
    ) {
        return 0;
    }
    // S = sum of x_t x_t^T, its diagonal kept (times the identity)
    tile_regs_acquire();
    matmul_init(DC_CB_ROW, DC_CB_ROW, 1);
    for (uint32_t tile = 0; tile < DC_ROW_TILES; tile++) {
        matmul_tiles(DC_CB_ROW, DC_CB_ROW, tile, tile, 0);
    }
    decode::multiply_init(DC_CB_CONSTANT);
    decode::multiply(DC_CB_CONSTANT, DC_CONSTANT_IDENTITY, 0);
    if (!pack_one(compute, DC_CB_SQUARES) || !compute.wait_front(DC_CB_SQUARES, 1)) {
        return 0;
    }
    // each row's mean square in every column, then 1 / sqrt(it + epsilon)
    tile_regs_acquire();
    matmul_init(DC_CB_SQUARES, DC_CB_CONSTANT, 0);
    matmul_tiles(DC_CB_SQUARES, DC_CB_CONSTANT, 0, DC_CONSTANT_MEAN, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, command.a1);
    rsqrt_tile_init();
    rsqrt_tile(0);
    cb_pop_front(DC_CB_SQUARES, 1);
    if (!pack_one(compute, DC_CB_SCALE) || !compute.wait_front(DC_CB_SCALE, 1)) {
        return 0;
    }
    // this core's tiles: x times the scale, times the weight
    const bool out = compute.reserve_back(DC_CB_NORMED, DC_NORM_SHARE);
    pack_reconfig_data_format(DC_CB_NORMED);
    for (uint32_t tile = 0; tile < DC_NORM_SHARE; tile++) {
        tile_regs_acquire();
        copy_init(DC_CB_ROW);
        copy_tile(DC_CB_ROW, first + tile, 0);
        decode::multiply_init(DC_CB_SCALE);
        decode::multiply(DC_CB_SCALE, 0, 0);
        decode::multiply_init(DC_CB_WEIGHT);
        decode::multiply(DC_CB_WEIGHT, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (out) {
            pack_tile(0, DC_CB_NORMED, tile);
        }
        tile_regs_release();
    }
    if (out) {
        cb_push_back(DC_CB_NORMED, DC_NORM_SHARE);
    }
    cb_pop_front(DC_CB_SCALE, 1);
    cb_pop_front(DC_CB_WEIGHT, DC_NORM_SHARE);
    cb_pop_front(DC_CB_ROW, DC_ROW_TILES);
    cb_pop_front(DC_CB_CONSTANT, DC_CONSTANTS);
    return out ? 1 : 0;
}
