// The reader kernel: no handler of this program reads anything.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {});
}
