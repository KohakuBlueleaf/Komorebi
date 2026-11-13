// The passes' writer: LANES the window's records, COUNT the counters, SETUP each record
// to its list slot, HIST the heavy tiles' histograms, PLACE the directory and marks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#include "setup_noc.hpp"

namespace {

    // Records the first give-up of this core's writer and asks the run to stop.
    void give_up(uint32_t code, uint32_t where) {
        const uint32_t ctl = get_arg_val<uint32_t>(0);
        volatile uint32_t* st = (
            reinterpret_cast<volatile uint32_t*>(ctl + KM_CTL_STATE)
        );
        if (st[km::kStWriter] == 0) {
            st[km::kStWriter] = code | where << 8;
        }
        *reinterpret_cast<volatile uint32_t*>(ctl + KM_CTL_STOP) = 1;
    }

}  // namespace

// The passes' writing for one record; the frame program's writer calls it as well.
bool setup_writer_record(const km::Writer& wr, const km_record_t& r) {
    const uint32_t seg = km::seg(r);
    if ((seg != kSegLanes && seg != kSegCount && seg != kSegSetup && seg != kSegHist &&
         seg != kSegPlace) ||
        r.a5 == 0 || wr.me() >= r.a5) {
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
    auto arrived = [&] {
        return wr.poll(
                   [] {
                       return ncrisc_noc_reads_flushed(noc_index);
                   },
                   wr.args().wait_limit
               ) == km::kGaveNone;
    };
    if (seg == kSegCount) {
        if (!wr.wait_front(kCbCount, 1)) {
            return true;
        }
        noc_async_write(
            get_read_ptr(kCbCount),
            su::addr(r.a8, r.a1, wr.me(), kCountBytes),
            kCountBytes
        );
        const bool ok = flushed();
        cb_pop_front(kCbCount, 1);
        return ok;
    }
    if (seg == kSegHist) {
        // the heavy tiles into the out buffer's space (unused by HIST)
        const uint32_t heavy = get_write_ptr(kCbOut);
        noc_async_read(su::addr(r.a8, r.a0, 0, kHeavyBytes), heavy, kHeavyBytes);
        if (!arrived()) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* p = km::l1(heavy);
        const uint32_t n = p[0];
        if (n >= kHeavyBytes / 4 || 64 * n > r.a4) {
            give_up(kGaveBadHeavy, n);
            return true;
        }
        const uint32_t hist = get_read_ptr(kCbHist);
        for (uint32_t h = 0; h < n; ++h) {
            const uint32_t t = p[1 + h] % kTiles;
            noc_async_write(
                hist + 64 * t, su::addr(r.a8, r.a1, wr.me(), r.a4, 64 * h), 64
            );
        }
        return flushed();
    }
    if (seg == kSegPlace) {
        if (r.a8 == 0) {
            give_up(kGaveBadPlace, 0);
            return true;
        }
        const uint32_t buf = get_write_ptr(kCbOut);
        if (wr.me() < r.a6) {
            noc_async_read(
                su::addr(r.a8, r.a0, wr.me(), kDirPageBytes), buf, kDirPageBytes
            );
            if (!arrived()) {
                return false;
            }
            noc_async_write(
                buf, su::addr(0, r.a1, wr.me(), kDirPageBytes), kDirPageBytes
            );
            if (!flushed()) {
                return false;
            }
        }
        const InterleavedAddrGen<true> lists = {
            .bank_base_address = r.a7, .page_size = kListPageBytes
        };
        const uint32_t m1 = (wr.me() + 1) * r.a4 / r.a5;
        for (uint32_t m = wr.me() * r.a4 / r.a5; m < m1; m += kMarkBatch) {
            const uint32_t n = m1 - m < kMarkBatch ? m1 - m : kMarkBatch;
            noc_async_read(su::addr(r.a8, r.a2, m, kMarkBytes), buf, n * kMarkBytes);
            if (!arrived()) {
                return false;
            }
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t at = buf + i * kMarkBytes;
                const uint32_t s = km::l1(at)[32];
                noc_async_write(at, get_noc_addr(s / 16, lists, (s % 16) * 128), 128);
            }
            if (!flushed()) {
                return false;
            }
        }
        return true;
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
        noc_async_write(get_read_ptr(kCbRec), get_noc_addr(b - r.a4, out), kRecBytes);
        if (!flushed()) {
            return false;
        }
        cb_pop_front(kCbRec, 1);
    }
    return true;
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        return setup_writer_record(wr, r);
    });
}
#endif
