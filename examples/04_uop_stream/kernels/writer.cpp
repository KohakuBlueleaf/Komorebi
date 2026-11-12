// Writer: MAP stores the member's tile at output page a3 + member.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "uop_stream.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegMap) {
            return true;
        }
        if (!wr.wait_front(kCbOut, 1)) {
            return false;  // nothing to report: the tile is stopping
        }
        const auto out = TensorAccessor(out_args, r.a2);
        noc_async_write_page(r.a3 + wr.me() - km::first(r), out, get_read_ptr(kCbOut));
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
