// A link core's compute: the runtime's loop alone (a link core runs no program).
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    compute.run([](const km_command_t&) {});
}
