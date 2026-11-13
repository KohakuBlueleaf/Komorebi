// The writer kernel: COPY writes the tiles the reader brought to DRAM.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer wr;
    // --8<-- [start:body]
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerCopy) {
            return true;
        }
        const auto out = TensorAccessor(out_args, r.a1);
        for (uint32_t t = 0; t < r.a2; t++) {
            if (!wr.wait_front(kCb, 1)) {  // wait until the reader pushed a tile
                return false;
            }
            noc_async_write_page(t, out, get_read_ptr(kCb));  // L1 -> DRAM
            noc_async_writes_flushed();  // the tile has left the buffer
            cb_pop_front(kCb, 1);        // the reader may reuse its page
        }
        return true;
    });
    // --8<-- [end:body]
}
