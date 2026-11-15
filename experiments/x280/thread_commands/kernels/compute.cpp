// Commands to an X280 thread, grid side: the compute kernel has nothing to do.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t&) {});
}
