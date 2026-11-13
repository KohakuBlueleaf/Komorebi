// X280 ports experiment, grid side: no segment uses the matrix or vector units.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute cp;
    cp.run([&](const km_record_t&) {});
}
