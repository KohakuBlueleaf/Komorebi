// The reader kernel: it has nothing to do for this program.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {});
}
