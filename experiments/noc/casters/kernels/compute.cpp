// Casters compute: takes each block and gives it back (the broadcast's DONE-BY).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "casters.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    const bool included = compute.arg(0) != 0;
    const uint32_t window = compute.arg(1), pages = compute.arg(2);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCast || !included) {
            return;
        }
        const uint32_t blocks = command.a1;
        bool ok = true;
        for (uint32_t block = 0; ok && block < blocks; block++) {
            ok = km::broadcast_receive(compute, kCbBlocks, pages);
            if (ok) {
                km::broadcast_done(kCbBlocks, pages);
            }
        }
        if (ok) {
            km::broadcast_end(compute, kCbBlocks, pages, window, blocks);
        }
    });
}
