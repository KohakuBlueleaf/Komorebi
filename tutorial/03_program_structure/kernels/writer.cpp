// The writer kernel: it reports every command as done.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        // There is nothing to write for any command. Return true to report it done.
        return true;
    });
}
