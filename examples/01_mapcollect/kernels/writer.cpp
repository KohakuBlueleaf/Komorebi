// Map/collect writer: MAP stores the member's tile at output page a3 + member; STALL
// reports for every member but a0.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "mapcollect.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        const uint32_t index = wr.me() - km::first(r);
        switch (km::seg(r)) {
            case kSegMap: {
                if (!wr.wait_front(kCbOut, 1)) {
                    return true;
                }
                const auto out = TensorAccessor(out_args, r.a2);
                noc_async_write_page(r.a3 + index, out, get_read_ptr(kCbOut));
                noc_async_writes_flushed();
                cb_pop_front(kCbOut, 1);
                return true;
            }
            case kSegStall:
                return index != r.a0;
            case kSegAlign:
                wr.poll(
                    [&] {
                        return static_cast<int32_t>(km::wall() - r.a9) >= 0;
                    },
                    wr.args().wait_limit
                );
                return true;
            default:
                return true;
        }
    });
}
