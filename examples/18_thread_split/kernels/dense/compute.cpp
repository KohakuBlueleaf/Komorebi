// Overlay dense, its compute part (one build per compute core): each input tile into
// the vector unit's registers, times 2, packed to the output buffer.
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "../../split.h"

uint32_t km_overlay(km::Compute& compute, const ts_scale& scale) {
    for (uint32_t tile = 0; tile < scale.count; tile++) {
        if (!compute.wait_front(TS_CB_IN, 1)) {
            return 0;
        }
        copy_init(TS_CB_IN);
        tile_regs_acquire();
        copy_tile(TS_CB_IN, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x40000000u);  // 2.0f
        tile_regs_commit();
        tile_regs_wait();
        if (compute.reserve_back(TS_CB_OUT, 1)) {
            pack_tile(0, TS_CB_OUT);
            cb_push_back(TS_CB_OUT, 1);
        }
        tile_regs_release();
        cb_pop_front(TS_CB_IN, 1);
    }
    return scale.count;
}
