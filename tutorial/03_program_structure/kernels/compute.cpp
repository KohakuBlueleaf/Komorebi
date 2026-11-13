// The compute kernel: runs on the three compute cores of every Tensix core.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute cp;
    cp.run([&](const km_record_t& r) {});
}
