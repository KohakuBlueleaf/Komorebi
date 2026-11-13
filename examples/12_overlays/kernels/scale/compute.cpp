// Overlay scale, its compute part (one build per compute core): each input tile into
// the vector unit's registers, times the scalar, packed to the output buffer.
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "../overlays.h"

// --8<-- [start:entry]
uint32_t km_overlay(km::Compute& compute, const ScaleArgument& argument) {
    for (uint32_t tile = 0; tile < argument.tiles; tile++) {
        if (!compute.wait_front(kCbIn, 1)) {
            return 0;
        }
        copy_init(kCbIn);
        tile_regs_acquire();
        copy_tile(kCbIn, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, argument.scalar);
        tile_regs_commit();
        tile_regs_wait();
        if (compute.reserve_back(kCbOut, 1)) {
            pack_tile(0, kCbOut);
            cb_push_back(kCbOut, 1);
        }
        tile_regs_release();
        cb_pop_front(kCbIn, 1);
    }
    return argument.tiles;
}

// --8<-- [end:entry]
