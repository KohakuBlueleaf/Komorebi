// Merge-tree writer: member 0 writes the total to output pages a3 on, every other
// member sends its part to its receiver (owner or tree); member a5 stalls instead.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "merge.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegMerge) {
            return true;
        }
        const km::Group g = km::team_group(r, wr.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, r.a7};
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
            return km::reduce_emit(wr, r, arrival, g, r.a4, kCbOut, r.a7);
        }
        if (!km::reduce_result(wr, kCbOut, r.a7)) {
            return false;
        }
        const auto out = TensorAccessor(out_args, r.a2);
        for (uint32_t d = 0; d < r.a7; d++) {
            noc_async_write_page(r.a3 + d, out, get_read_ptr(kCbOut) + d * kTileBytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, r.a7);
        return true;
    });
}
