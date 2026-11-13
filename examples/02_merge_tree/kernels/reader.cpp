// Merge-tree reader: reads the member's share of the T input tiles in blocks, then
// waits for the parts this member combines (by owner or by tree).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "merge.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    km::Reader rd;
    uint32_t at = 0;  // write position in kCbIn, in pages: a block never wraps
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerMerge) {
            return;
        }
        const km::Group g = km::group_of(r, rd.me());
        const km::Partition s = km::partition(r.a1, g.rank, g.size);
        const auto in = TensorAccessor(in_args, r.a0);
        for (uint32_t done = 0; done < s.len;) {
            uint32_t n = s.len - done < kBlock ? s.len - done : kBlock;
            n = n < kInPages - at ? n : kInPages - at;
            if (!rd.reserve(kCbIn, n)) {
                return;
            }
            const uint32_t dst = get_write_ptr(kCbIn);
            for (uint32_t k = 0; k < n; k++) {
                noc_async_read_page(
                    r.a6 + s.start + done + k, in, dst + k * kTileBytes
                );
            }
            noc_async_read_barrier();
            cb_push_back(kCbIn, n);
            done += n;
            at = (at + n) % kInPages;
        }
        km::reduce_arrive(rd, r, {kCbArrive, kArrivePages, r.a7}, g, r.a4);
    });
}
