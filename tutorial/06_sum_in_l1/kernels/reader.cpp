// The reader kernel: SUM reads the rows of a DRAM tensor into the circular buffer.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegSum) {
            return;
        }
        const auto in = TensorAccessor(in_args, r.a0);  // row p of the input tensor
        for (uint32_t p = 0; p < r.a1; p++) {
            if (!rd.reserve(kCb, 1)) {
                return;
            }
            noc_async_read_page(p, in, get_write_ptr(kCb));
            noc_async_read_barrier();
            cb_push_back(kCb, 1);
        }
    });
}
