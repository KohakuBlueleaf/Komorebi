// Ray tracing writer: each output page to its place in DRAM, then the profile page
// (compute's counters, then the reader's and its own) to a4.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "timed.hpp"
#include "dir_noc.hpp"
#include "komorebi/device/km_worker.hpp"
#include "raytrace.h"

namespace {

    // Hit pages: a nearest hit's with the attributes; the probe's as any hit's.
    uint32_t out_page_bytes(const km_command_t& r, uint32_t handler) {
        if (handler == kHandlerTrace || handler == kHandlerTreelets) {
            return r.a3 == kModeAny ? kHitAnyBytes : kHitBytes;
        }
        if (handler == kHandlerProbe) {
            return kHitAnyBytes;
        }
        if (handler == kHandlerMath) {
            return kMaOutVectors * kVec * 4;
        }
        return handler == kHandlerArith ? kArOutVectors * kVec * 4 : kRecipPerPage * 4;
    }

    uint32_t out_per_page(uint32_t handler) {
        const bool hits =
            (handler == kHandlerTrace || handler == kHandlerTreelets ||
             handler == kHandlerProbe);
        return hits ? kPagePackets : 1;
    }

    // The front output page to page g of a stream, popped; the wait for it is added to
    // `wait`.
    bool write_page(
        km::Writer& wr,
        uint32_t base,
        uint32_t bytes,
        uint32_t per,
        uint32_t g,
        uint32_t& wait
    ) {
        const uint32_t a = km::wall();
        if (!wr.wait_front(kCbOut, 1)) {
            return false;
        }
        wait += km::wall() - a;
        const InterleavedAddrGen<true> out = {
            .bank_base_address = base, .page_size = bytes * per
        };
        noc_async_write(
            get_read_ptr(kCbOut), get_noc_addr(g / per, out, (g % per) * bytes), bytes
        );
        if (!timed::writes_sent(wr, km::kStWriter)) {
            return false;
        }
        cb_pop_front(kCbOut, 1);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        if (handler != kHandlerTrace && handler != kHandlerArith &&
            handler != kHandlerRecip && handler != kHandlerTreelets &&
            handler != kHandlerProbe && handler != kHandlerMath && !is_stage(handler)) {
            return true;
        }
        const uint32_t t0 = km::wall();
        const uint32_t dir = get_write_ptr(kCbDirWriter);
        volatile tt_l1_ptr uint32_t* d = km::l1(dir);
        noc_async_read(dir_addr(r, handler, wr.me()), dir, kDirBytes);
        if (!timed::reads_done(wr, km::kStWriter)) {
            return true;
        }
        const uint32_t first = d[kDirFirst], count = d[kDirCount];
        uint32_t wait = 0, write = 0;
        if (is_stage(handler)) {
            // Each packet's page of every output stream, in stream order.
            const Stage st = stage_of(handler);
            for (uint32_t p = 0; p < count; ++p) {
                for (uint32_t s = st.inputs; s < st.inputs + st.outputs; ++s) {
                    const uint32_t a = km::wall();
                    const uint32_t bytes = st.vectors[s] * kVec * 4;
                    if (!write_page(
                            wr, stream_arg(r, s), bytes, kPagePackets, first + p, wait
                        )) {
                        return true;
                    }
                    write += km::wall() - a;
                }
            }
        } else {
            const uint32_t bytes = out_page_bytes(r, handler),
                           per = out_per_page(handler);
            for (uint32_t p = 0; p < count; ++p) {
                const uint32_t a = km::wall();
                if (!write_page(wr, r.a2, bytes, per, first + p, wait)) {
                    return true;
                }
                write += km::wall() - a;
            }
        }
        write -= wait;
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
        const uint32_t bank = handler == kHandlerTreelets ? r.a9 >> kBankShift : 0;
        if (bank != 0 && d[kDirCost] != 0) {
            noc_async_write(
                get_read_ptr(kCbOut) + 4 * kProfCost,
                get_noc_addr_from_bank_id<true>(bank - 1, d[kDirCost]),
                kCostBytes
            );
        }
        if (!timed::writes_sent(wr, km::kStWriter)) {
            return true;
        }
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
