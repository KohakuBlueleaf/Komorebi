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
                    return false;  // nothing to report: the tile is stopping
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
                return wr.poll(
                           [&] {
                               return static_cast<int32_t>(km::wall() - r.a9) >= 0;
                           },
                           wr.args().wait_limit
                       ) == km::kGaveNone;
            case kSegLate: {
                if (index != r.a0) {
                    return true;
                }
                const uint32_t t0 = km::wall();
                return wr.poll(
                           [&] {
                               return km::wall() - t0 >= r.a1;
                           },
                           wr.args().idle_limit
                       ) == km::kGaveNone;
            }
            case kSegResult:
                wr.result(r.a0 + index);
                return true;
            case kSegBadItem:
                return index != r.a0 || r.a1 != 0 || wr.wait_front(kCbOut, 2);
            default:
                return true;
        }
    });
}
