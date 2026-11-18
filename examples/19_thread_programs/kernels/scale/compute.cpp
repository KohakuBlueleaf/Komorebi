// The scale program's compute: SCALE multiplies each of the core's tiles by a4 on the
// vector unit.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/core/km_program.hpp"
#include "../../programs.h"

void km_program(km::Compute& compute, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SCALE) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, compute.me() - km::first(command), km::count(command))
    );
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!compute.wait_front(TP_CB_IN, 1)) {
            return;
        }
        copy_init(TP_CB_IN);
        tile_regs_acquire();
        copy_tile(TP_CB_IN, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, command.a4);
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(TP_CB_OUT, 1);
        if (room) {
            pack_tile(0, TP_CB_OUT);
            cb_push_back(TP_CB_OUT, 1);
        }
        tile_regs_release();
        cb_pop_front(TP_CB_IN, 1);
        if (!room) {
            return;
        }
    }
}
