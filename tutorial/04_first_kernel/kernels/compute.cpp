// The compute kernel: it has nothing to do for this program.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute cp;
    cp.run([&](const km_command_t& r) {});
}
