// Receiver compute: waits for all S pages, copies page k and then page j by index
// (no pop before), packs each into the output buffer, pops S.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "probe.h"

void kernel_main() {
    const uint32_t k = get_arg_val<uint32_t>(0), j = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(kCbArrive, kCbOut);
    cb_wait_front(kCbArrive, kPages);  // the reader always pushes S pages
    cb_reserve_back(kCbOut, 2);        // two free pages from the start
    const uint32_t pages[2] = {k & 3u, j & 3u};
    for (uint32_t i = 0; i < 2; i++) {
        tile_regs_acquire();
        copy_init(kCbArrive);
        copy_tile(kCbArrive, pages[i], 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kCbOut);  // the next page of the reservation
        tile_regs_release();
    }
    cb_push_back(kCbOut, 2);
    cb_pop_front(kCbArrive, kPages);
}
