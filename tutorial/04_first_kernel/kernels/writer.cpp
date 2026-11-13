// The writer kernel: ADD adds two words of the command and returns the sum.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer wr;
    // --8<-- [start:body]
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) == kSegAdd) {
            wr.result(r.a0 + r.a1);
        }
        return true;
    });
    // --8<-- [end:body]
}
