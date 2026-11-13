// Matrix product writer: PRODUCT stores each packed column block at Y[m, j0 ..);
// PROF stores this core's profile page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto tiles = TensorAccessorArgs<0>();
    km::Writer wr;
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbProf));
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) == kSegProf) {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = r.a0, .page_size = kProfBytes
            };
            noc_async_write(
                get_write_ptr(kCbProf), get_noc_addr(wr.me(), pages), kProfBytes
            );
            return flushed(wr, km::kStWriter, km::kFlushSent);
        }
        if (km::seg(r) == kSegLoad) {  // completes once W is in L1
            if (!wr.wait_front(kCbLoaded, 1)) {
                return false;
            }
            cb_pop_front(kCbLoaded, 1);
            return true;
        }
        if (km::seg(r) != kSegProduct || r.a7 == kModeReadOnly) {
            return true;
        }
        const uint32_t t0 = km::wall();
        const km::Share s = km::share(r.a2, wr.me() - km::first(r), km::count(r));
        const uint32_t nt = r.a4, ct = r.a6;
        const bool store = r.a7 == kModeFull;
        const auto y = TensorAccessor(tiles, r.a1);
        uint32_t wait = 0, write = 0, n = 0;
        for (uint32_t m = s.start; m < s.start + s.len; ++m) {
            for (uint32_t j0 = 0; j0 < nt; j0 += ct) {
                const uint32_t t1 = km::wall();
                if (!wr.wait_front(kCbOut, ct)) {
                    return false;  // nothing to report: the tile is stopping
                }
                const uint32_t t2 = km::wall();
                if (store) {
                    uint32_t src = get_read_ptr(kCbOut);
                    for (uint32_t j = 0; j < ct; ++j) {
                        noc_async_write_page(m * nt + j0 + j, y, src);
                        src += kTileBytes;
                    }
                    if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    n += ct;
                }
                cb_pop_front(kCbOut, ct);
                wait += t2 - t1;
                write += km::wall() - t2;
            }
        }
        km::add_sat(prof + kPrWrWait, wait);
        km::add_sat(prof + kPrWrWrite, write);
        km::add_sat(prof + kPrWrTiles, n);
        km::add_sat(prof + kPrWrBody, km::wall() - t0);
        return true;
    });
}
