// Attention writer: reads its own copy of the member's job table, then stores each
// finished output row (DV tiles) of every job at O's pages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "attention.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr uint32_t DV = get_compile_time_arg_val(0);
    constexpr auto tiles = TensorAccessorArgs<1>();  // O: bf16 tiles in DRAM
    km::Writer wr;
    // The table copy's page: reserved once, used in place, never pushed.
    cb_reserve_back(kCbWriterJobs, 1);
    const uint32_t table = get_write_ptr(kCbWriterJobs);
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegAttend) {
            return true;
        }
        if (!read_table(wr, km::kStWriter, r.a4, wr.me() - km::first(r), table)) {
            return false;
        }
        const uint32_t jobs = km::l1(table)[kTabJobs];
        if (jobs > kJobsMax) {
            return false;
        }
        const auto o = TensorAccessor(tiles, r.a3);
        for (uint32_t j = 0; j < jobs; ++j) {
            const volatile tt_l1_ptr uint32_t* w = (
                km::l1(table + 4 * (kTabFirst + kJobWords * j))
            );
            const uint32_t o0 = w[kJobO], rows = w[kJobRows], stride = w[kJobStride];
            for (uint32_t i = 0; i < rows; ++i) {
                if (!wr.wait_front(kCbOut, DV)) {
                    return false;  // nothing to report: the tile is stopping
                }
                const uint32_t src = get_read_ptr(kCbOut);
                for (uint32_t c = 0; c < DV; ++c) {
                    noc_async_write_page(
                        o0 + i * stride * DV + c, o, src + c * kTileBytes
                    );
                }
                if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                    return false;
                }
                cb_pop_front(kCbOut, DV);
            }
        }
        return true;
    });
}
