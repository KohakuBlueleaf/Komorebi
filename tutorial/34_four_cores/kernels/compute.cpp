// The compute kernel. It has nothing to do.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t& command) {});
}
