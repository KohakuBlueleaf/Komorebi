// An L1 tensor moved across islands, compute: no handler uses the compute units.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t&) {});
}
