// Broadcast reader: the row lane, on NoC0. Each row's sender reads the row's A blocks
// from DRAM and casts them along the row (READ: every member reads them itself).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"
#include "komorebi/device/km_lane.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if (seg != kSegCast && seg != kSegRead) {
            return;
        }
        km::Lane ln(rd, r, kArgLane);
        if (!ln.ok() || !ln.member()) {
            return;
        }
        const auto a = TensorAccessor(a_args, r.a0);
        const uint32_t first = rd.arg(kArgIndex) * r.a7 * ln.pages();
        if (seg == kSegRead) {
            read_blocks(rd, km::kStReader, r, ln, kCbRow, a, first, r.a3, r.a7);
            return;
        }
        const bool stalls = rd.me() - km::first(r) == r.a4;
        cast_blocks(ln, a, first, r.a3, r.a7, [&](uint32_t k) {
            if (!stalls || k != r.a5) {
                return false;
            }
            rd.wait_stop(r);  // a test: this receiver stops taking blocks
            return true;
        });
    });
}
