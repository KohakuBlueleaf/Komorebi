// The compute kernel: no handler of this program computes on the vector unit.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute cp;
    cp.run([&](const km_command_t& r) {});
}
