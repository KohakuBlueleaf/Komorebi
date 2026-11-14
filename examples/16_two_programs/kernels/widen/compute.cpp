// The widen program's compute: WIDEN copies each bfloat16 input tile and packs it as
// float32 (the unpacker and packer set up for widen's formats at its load).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_program.hpp"
#include "../two_programs.h"

void km_program(km::Compute& compute, const km_command_t& command) {
    if (km::handler(command) != kHandlerWiden) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, compute.me() - km::first(command), km::count(command))
    );
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!compute.wait_front(kCbWideIn, 1)) {
            return;
        }
        tile_regs_acquire();
        copy_init(kCbWideIn);
        copy_tile(kCbWideIn, 0, 0);
        cb_pop_front(kCbWideIn, 1);
        tile_regs_commit();
        tile_regs_wait();
        const bool room = compute.reserve_back(kCbWideOut, 1);
        if (room) {
            pack_tile(0, kCbWideOut);
            cb_push_back(kCbWideOut, 1);
        }
        tile_regs_release();
        if (!room) {
            return;
        }
    }
}
