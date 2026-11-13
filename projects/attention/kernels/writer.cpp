// Attention writer: reads its own copy of the member's job table, then stores each
// finished output row (DV tiles) of every job at O's pages. PROF: the profile page out.
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
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbProf));
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) == kSegProf) {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = r.a0, .page_size = kProfBytes
            };
            const uint32_t member = wr.me() - km::first(r);
            noc_async_write(
                get_write_ptr(kCbProf), get_noc_addr(member, pages), kProfBytes
            );
            if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                return false;
            }
            for (uint32_t w = 0; r.a1 != 0 && w < kProfWords; ++w) {
                prof[w] = 0;
            }
            return true;
        }
        if (km::seg(r) != kSegAttend) {
            return true;
        }
        const uint32_t t0 = km::wall();
        if (!read_table(wr, km::kStWriter, r.a4, wr.me() - km::first(r), table)) {
            return false;
        }
        const uint32_t jobs = km::l1(table)[kTabJobs];
        const uint32_t passes = r.a6 == 0 ? 1 : r.a6;
        if (jobs > kJobsMax) {
            return false;
        }
        // Cast only: compute finishes no row. Compute only: rows popped, not written.
        const uint32_t each = r.a5 == kModeCastOnly ? 0 : 1;
        const bool writes = r.a5 == kModeFull;
        const auto o = TensorAccessor(tiles, r.a3);
        uint32_t waited = 0, wrote = 0;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (uint32_t j = 0; j < jobs; ++j) {
                const volatile tt_l1_ptr uint32_t* w = (
                    km::l1(table + 4 * (kTabFirst + kJobWords * j))
                );
                const uint32_t o0 = w[kJobO], rows = w[kJobRows] * each;
                const uint32_t stride = w[kJobStride];
                for (uint32_t i = 0; i < rows; ++i) {
                    const uint32_t t1 = km::wall();
                    if (!wr.wait_front(kCbOut, DV)) {
                        return false;  // nothing to report: the tile is stopping
                    }
                    const uint32_t t2 = km::wall();
                    const uint32_t src = get_read_ptr(kCbOut);
                    for (uint32_t c = 0; writes && c < DV; ++c) {
                        noc_async_write_page(
                            o0 + i * stride * DV + c, o, src + c * kTileBytes
                        );
                    }
                    if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    cb_pop_front(kCbOut, DV);
                    waited += t2 - t1;
                    wrote += km::wall() - t2;
                }
            }
        }
        km::add_sat(prof + kPrWrWait, waited);
        km::add_sat(prof + kPrWrWrite, wrote);
        km::add_sat(prof + kPrWrBody, km::wall() - t0);
        return true;
    });
}
