// Ray tracing writer: each output page to its place in DRAM, then the profile page
// (compute's counters, then the reader's and its own) to a4.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "bounded.hpp"
#include "komorebi/device/km_worker.hpp"
#include "raytrace.h"

namespace {

    uint32_t out_page_bytes(uint32_t seg) {
        if (seg == kSegTrace || seg == kSegTreelets || seg == kSegProbe) {
            return kHitBytes;
        }
        return seg == kSegArith ? kArOutVectors * kVec * 4 : kRecipPerPage * 4;
    }

    uint32_t out_per_page(uint32_t seg) {
        const bool hits = seg == kSegTrace || seg == kSegTreelets || seg == kSegProbe;
        return hits ? kHitPagePackets : 1;
    }

}  // namespace

void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if (seg != kSegTrace && seg != kSegArith && seg != kSegRecip &&
            seg != kSegTreelets && seg != kSegProbe) {
            return true;
        }
        const uint32_t t0 = km::wall();
        const uint32_t dir = get_write_ptr(kCbDirWriter);
        volatile tt_l1_ptr uint32_t* d = km::l1(dir);
        const InterleavedAddrGen<true> dirs = {
            .bank_base_address = r.a0, .page_size = kDirBytes
        };
        noc_async_read(get_noc_addr(wr.me(), dirs), dir, kDirBytes);
        if (!bounded::reads_done(wr, km::kStWriter)) {
            return true;
        }
        const uint32_t first = d[kDirFirst], count = d[kDirCount];
        const uint32_t bytes = out_page_bytes(seg), per = out_per_page(seg);
        const InterleavedAddrGen<true> out = {
            .bank_base_address = r.a2, .page_size = bytes * per
        };
        uint32_t wait = 0, write = 0;
        for (uint32_t p = 0; p < count; ++p) {
            const uint32_t a = km::wall();
            if (!wr.wait_front(kCbOut, 1)) {
                return true;
            }
            const uint32_t b = km::wall();
            const uint32_t g = first + p;
            const uint64_t at = get_noc_addr(g / per, out, (g % per) * bytes);
            noc_async_write(get_read_ptr(kCbOut), at, bytes);
            if (!bounded::writes_sent(wr, km::kStWriter)) {
                return true;
            }
            cb_pop_front(kCbOut, 1);
            wait += b - a;
            write += km::wall() - b;
        }
        // The profile page: compute filled words [0, 32); reader and writer follow.
        if (!wr.wait_front(kCbOut, 1)) {
            return true;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* prof = km::l1(get_read_ptr(kCbOut));
        volatile tt_l1_ptr uint32_t* rd = km::l1(get_write_ptr(kCbDirReader));
        for (uint32_t k = 0; k < kRdCounters; ++k) {
            prof[kProfReader + k] = rd[kRdWords + k];
        }
        prof[kProfWriter + 0] = wait;
        prof[kProfWriter + 1] = write;
        prof[kProfWriter + 2] = km::wall() - t0;
        prof[kProfWriter + 3] = count;
        const InterleavedAddrGen<true> profs = {
            .bank_base_address = r.a4, .page_size = kOutPageBytes
        };
        noc_async_write(
            get_read_ptr(kCbOut), get_noc_addr(wr.me(), profs), kOutPageBytes
        );
        if (!bounded::writes_sent(wr, km::kStWriter)) {
            return true;
        }
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
