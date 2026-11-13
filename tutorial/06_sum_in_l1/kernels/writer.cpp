// The writer kernel: SUM adds up the words of each row where the reader put it in L1.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer wr;
    // --8<-- [start:body]
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegSum) {
            return true;
        }
        uint32_t sum = 0;
        for (uint32_t p = 0; p < r.a1; p++) {
            if (!wr.wait_front(kCb, 1)) {
                return false;
            }
            invalidate_l1_cache();  // the NoC wrote this page behind the core's cache
            const volatile uint32_t* row = (
                reinterpret_cast<const volatile uint32_t*>(get_read_ptr(kCb))
            );
            for (uint32_t i = 0; i < kPageWords; i++) {
                sum += row[i];  // L1 is plain memory: a pointer reads it
            }
            cb_pop_front(kCb, 1);
        }
        wr.result(sum);
        return true;
    });
    // --8<-- [end:body]
}
