// The reader kernel: COPY reads tiles from DRAM into the circular buffer.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    km::Reader rd;
    // --8<-- [start:body]
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegCopy) {
            return;
        }
        const auto in = TensorAccessor(in_args, r.a0);  // tile t of the input tensor
        for (uint32_t t = 0; t < r.a2; t++) {
            if (!rd.reserve(kCb, 1)) {  // wait for room in the buffer
                return;
            }
            noc_async_read_page(t, in, get_write_ptr(kCb));  // DRAM -> L1 over the NoC
            noc_async_read_barrier();                        // wait until it landed
            cb_push_back(kCb, 1);                            // tell the writer
        }
    });
    // --8<-- [end:body]
}
