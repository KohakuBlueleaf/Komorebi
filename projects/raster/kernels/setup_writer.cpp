// COUNT and SETUP writer: LANES writes each block's 32 records in the window to its
// page of the output; COUNT the core's counters; SETUP each record to its list slot.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"

void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if ((seg != kSegLanes && seg != kSegCount && seg != kSegSetup) || r.a5 == 0 ||
            wr.me() >= r.a5) {
            return true;
        }
        auto flushed = [&] {
            return wr.poll(
                       [] {
                           return ncrisc_noc_nonposted_writes_flushed(noc_index);
                       },
                       wr.args().wait_limit
                   ) == km::kGaveNone;
        };
        if (seg == kSegCount) {
            if (!wr.wait_front(kCbCount, 1)) {
                return true;
            }
            const InterleavedAddrGen<true> out = {
                .bank_base_address = r.a1, .page_size = kCountBytes
            };
            noc_async_write(
                get_read_ptr(kCbCount), get_noc_addr(wr.me(), out), kCountBytes
            );
            const bool ok = flushed();
            cb_pop_front(kCbCount, 1);
            return ok;
        }
        if (seg == kSegSetup) {
            const InterleavedAddrGen<true> lists = {
                .bank_base_address = r.a1, .page_size = kListPageBytes
            };
            for (;;) {
                if (!wr.wait_front(kCbOut, 1, 4)) {
                    return true;
                }
                const uint32_t page = get_read_ptr(kCbOut);
                volatile tt_l1_ptr uint32_t* p = km::l1(page);
                const uint32_t head = p[0], n = head & ~kOutLast;
                for (uint32_t i = 0; i < n && i < kOutEntries; ++i) {
                    const uint32_t s = p[1 + i];
                    noc_async_write(
                        page + 128 * (1 + i),
                        get_noc_addr(s / 16, lists, (s % 16) * 128),
                        128
                    );
                }
                const bool ok = flushed();
                cb_pop_front(kCbOut, 1);
                if (!ok) {
                    return false;
                }
                if (head & kOutLast) {
                    return true;
                }
            }
        }
        const uint32_t b0 = wr.me() * r.a2 / r.a5, b1 = (wr.me() + 1) * r.a2 / r.a5;
        const InterleavedAddrGen<true> out = {
            .bank_base_address = r.a1, .page_size = kRecBytes
        };
        for (uint32_t b = b0; b < b1; ++b) {
            if (!wr.wait_front(kCbRec, 1)) {
                return true;
            }
            if (b - r.a4 >= r.a6) {  // outside the window (b < a4 wraps)
                cb_pop_front(kCbRec, 1);
                continue;
            }
            noc_async_write(
                get_read_ptr(kCbRec), get_noc_addr(b - r.a4, out), kRecBytes
            );
            if (!flushed()) {
                return false;
            }
            cb_pop_front(kCbRec, 1);
        }
        return true;
    });
}
