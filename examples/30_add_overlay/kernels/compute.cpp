// The compute kernel of the load: RUN has no compute work; the start-up takes the
// formats of the buffers an added handler's compute part uses.
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_worker.hpp"
#include "add.h"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    compute.run([&](const km_command_t&) {});
}
