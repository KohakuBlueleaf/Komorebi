// scale and shift on compute: the reader's tile into the vector unit's registers, times
// or plus the scalar, packed to the output buffer. In a part, or compiled in.
#pragma once

#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "leader_overlay.h"

// One tile through the vector unit: times the scalar (`times`) or plus it. 0: the tile
// did not come (recorded).
inline uint32_t compute_tile(km::Compute& compute, uint32_t scalar, bool times) {
    if (!compute.wait_front(kCbIn, 1)) {
        return 0;
    }
    copy_init(kCbIn);
    tile_regs_acquire();
    copy_tile(kCbIn, 0, 0);
    binop_with_scalar_tile_init();
    if (times) {
        mul_unary_tile(0, scalar);
    } else {
        add_unary_tile(0, scalar);
    }
    tile_regs_commit();
    tile_regs_wait();
    const bool room = compute.reserve_back(kCbOut, 1);
    if (room) {
        pack_tile(0, kCbOut);
        cb_push_back(kCbOut, 1);
    }
    tile_regs_release();
    cb_pop_front(kCbIn, 1);
    return room ? 1 : 0;
}
