// The sum program's compute, as in example 16: SUM adds the core's input tiles into one
// tile (a zero tile when its partition is empty).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_program.hpp"
#include "../two_programs.h"

void km_program(km::Compute& compute, const km_command_t& command) {
    if (km::handler(command) != kHandlerSum) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, compute.me() - km::first(command), km::count(command))
    );
    tile_regs_acquire();
    bool ok = true;  // the same on all three threads (wait_front tells them)
    if (partition.len == 0) {
        fill_tile_init();
        fill_tile(0, 0.0f);
    } else {
        // the sum stays in DST (fp32): each tile is copied to DST 1, the SFPU adds it
        copy_init(kCbIn);
        ok = compute.wait_front(kCbIn, 1);
        if (ok) {
            copy_tile(kCbIn, 0, 0);
            cb_pop_front(kCbIn, 1);
        }
        for (uint32_t tile = 1; ok && tile < partition.len; tile++) {
            ok = compute.wait_front(kCbIn, 1);
            if (ok) {
                copy_init(kCbIn);
                copy_tile(kCbIn, 0, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                cb_pop_front(kCbIn, 1);
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    if (ok && compute.reserve_back(kCbOut, 1)) {  // a give-up: no output tile
        pack_tile(0, kCbOut);
        cb_push_back(kCbOut, 1);
    }
    tile_regs_release();
}
