// Bucket sort, Tensix side, the COMPUTE kernel (the three TRISC cores). It has nothing
// to do: a bucket number is one shift, which the writer does as it reads (README.md).
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute cp;
    // It still takes every command and returns at once, so the writer may report it.
    cp.run([&](const km_command_t&) {});
}
