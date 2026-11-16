// The scale program's compute: SCALE doubles each of the core's input tiles (the tile
// added to itself on the SFPU, exact in bfloat16).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/core/km_program.hpp"
#include "../two_programs.h"

void km_program(km::Compute& compute, const km_command_t& command) {
    if (km::handler(command) != kHandlerScale) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, compute.me() - km::first(command), km::count(command))
    );
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!compute.wait_front(kCbIn, 1)) {
            return;
        }
        tile_regs_acquire();
        copy_init(kCbIn);
        copy_tile(kCbIn, 0, 0);
        copy_tile(kCbIn, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        cb_pop_front(kCbIn, 1);
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(kCbOut, 1);
        if (room) {
            pack_tile(0, kCbOut);
            cb_push_back(kCbOut, 1);
        }
        tile_regs_release();
        if (!room) {
            return;
        }
    }
}
