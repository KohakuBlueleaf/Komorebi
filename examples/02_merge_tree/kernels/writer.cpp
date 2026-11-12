// Merge-tree writer: member 0 writes the total to output page a3; every other member
// sends its part to its receiver (by owner or by tree). Member a5 stalls instead.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "merge.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr km::Arrival arrival = {kCbArrive, kArrivePages};
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegMerge) {
            return true;
        }
        const km::Group g = km::team_group(r, wr.me());
        if (g.rank == r.a5) {  // a test: waits for the stop word, sends nothing
            const uint32_t code = wr.poll(
                [] {
                    return false;
                },
                wr.args().idle_limit
            );
            wr.gave_up(km::kStWriter, code, km::kWhereStall, km::slot(r), r.seq);
            return false;
        }
        if (g.rank != 0) {
            return km::reduce_emit(wr, r, arrival, g, r.a4, kCbOut, 1);
        }
        if (!km::reduce_result(wr, kCbOut, 1)) {
            return false;
        }
        const auto out = TensorAccessor(out_args, r.a2);
        noc_async_write_page(r.a3, out, get_read_ptr(kCbOut));
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
