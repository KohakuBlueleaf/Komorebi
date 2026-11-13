// The reader kernel: runs on RISC-V core 0 of every Tensix core. It has nothing to do.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {});
}
