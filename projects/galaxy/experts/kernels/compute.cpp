// The tree's compute: an island's sum of the block up the tree, tile by tile: its own
// slice's tile and each child's, added in DST (fp32) by the SFPU, packed to kCbSum.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/core/km_program.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "common.h"

constexpr uint32_t kKids = TREE_KIDS;

void km_program(km::Compute& compute, const km_command_t& command) {
    if (km::handler(command) != kHandlerTree) {
        return;
    }
    const km::Islands view = km::islands_at(compute.control());
    const uint32_t here = view.here();
    const tree::Word w = tree::word_of(command.a0);
    if (
        (w.phase & kUp) == 0 || compute.me() - view.first() >= w.senders ||
        tree::parent(command, here) == kNone
    ) {
        return;
    }
    const uint32_t n = TREE_PAGES;
    uint32_t kids[kKids];
    const uint32_t count = tree::children(command, here, kids, kKids);
    if (!compute.wait_front(kCbOwn, n)) {
        return;
    }
    if (count != 0 && !compute.wait_front(kCbUp, kKids * n)) {
        return;
    }
    for (uint32_t t = 0; t < n; t++) {
        tile_regs_acquire();
        copy_init(kCbOwn);
        copy_tile(kCbOwn, t, 0);
        for (uint32_t j = 0; j < count; j++) {
            copy_init(kCbUp);
            copy_tile(kCbUp, j * n + t, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (!compute.reserve_back(kCbSum, 1)) {
            tile_regs_release();
            return;
        }
        pack_tile(0, kCbSum);
        cb_push_back(kCbSum, 1);
        tile_regs_release();
    }
    cb_pop_front(kCbOwn, n);
    if (count != 0) {
        cb_pop_front(kCbUp, kKids * n);
    }
}
