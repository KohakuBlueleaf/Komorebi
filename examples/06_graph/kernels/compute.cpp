// Graph compute: no segment uses the matrix or vector units.
#include <stdint.h>

#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute cp;
    cp.run([&](const km_record_t&) {});
}
