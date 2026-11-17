// Compute takes each block mover A pushes and frees it: the read dropped.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRead) {
            return;
        }
        const uint32_t pages = command.a2;
        const uint32_t block_pages = command.a3;
        const uint32_t buffer = command.a4;
        const uint32_t repetitions = command.a5;
        const uint32_t block_tiles = command.a7;
        const uint32_t blocks = (pages + block_pages - 1) / block_pages * repetitions;
        for (uint32_t block = 0; block < blocks; block++) {
            if (!compute.wait_front(buffer, block_tiles)) {
                return;
            }
            cb_pop_front(buffer, block_tiles);
        }
    });
}
