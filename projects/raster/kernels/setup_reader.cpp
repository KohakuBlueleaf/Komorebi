// COUNT and SETUP reader: SETUP's slots and items pages, then the camera's page and
// this core's share of the mesh blocks into the mesh buffer, one page each.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#include "setup_noc.hpp"

namespace {

    // kSetupPrefix: base + SUM's entries before (a whole tile) or HSUM's starts before
    // row1 less ends before row0 (README.md); in place, tiles from the last.
    template <typename Arrived>
    bool slots_from_prefixes(
        const km::Reader& rd, const km_record_t& r, Arrived arrived
    ) {
        const uint32_t space = get_write_ptr(kCbHist);
        constexpr uint32_t kRead = kHprefixAddrAt - kBaseAt +
            64;  // base words, address
        noc_async_read(su::addr(r.a8, r.a7, 0, 0, kBaseAt), space, kRead);
        if (!arrived()) {
            return false;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* items = km::l1(get_write_ptr(kCbItems));
        volatile tt_l1_ptr uint32_t* base = km::l1(space);
        const uint32_t last = items[kTiles - 1];
        const uint32_t n_items = (last & 0xFFFF) + (last >> 16);
        if (n_items > kSlotsBytes / 4) {
            return false;
        }
        uint32_t nh = 0;
        for (uint32_t i = 0; i < n_items; ++i) {
            const uint32_t h = base[i] >> kBaseSlotBits;
            nh = h > nh ? h : nh;
        }
        const uint32_t counts = space + kRead;
        const uint32_t pages = base[(kHprefixAddrAt - kBaseAt) / 4];
        if (nh > kHprefixPageBytes / 128) {
            return false;
        }
        if (nh != 0) {
            noc_async_read(
                su::addr(0, pages, rd.me(), kHprefixPageBytes), counts, 128 * nh
            );
            if (!arrived()) {
                return false;
            }
            invalidate_l1_cache();
        }
        volatile tt_l1_ptr uint32_t* slot = km::l1(get_write_ptr(kCbSlots));
        const volatile tt_l1_ptr uint16_t*
            rows = (
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    get_write_ptr(kCbItems) + kItemRows
                )
            );
        constexpr uint32_t kSlotMask = (1u << kBaseSlotBits) - 1;
        for (uint32_t t = kTiles; t-- > 0;) {
            const uint32_t first = items[t] & 0xFFFF, n = items[t] >> 16;
            const uint32_t before = slot[t];
            for (uint32_t i = first; i < first + n && i < n_items; ++i) {
                const uint32_t b = base[i], h = b >> kBaseSlotBits;
                uint32_t s = b & kSlotMask;
                if (h == 0) {
                    s += before;
                } else {
                    volatile tt_l1_ptr uint32_t* c = km::l1(counts + 128 * (h - 1));
                    const uint32_t g1 = (rows[i] >> 8) >> 1, g0 = (rows[i] & 0xFF) >> 1;
                    s += (g1 ? c[g1 - 1] : 0) - (g0 ? c[16 + g0 - 1] : 0);
                }
                slot[i] = s;
            }
        }
        asm volatile("fence" ::: "memory");  // the stores land before the push
        return true;
    }

}  // namespace

// The passes' reading for one record; the frame program's reader calls it as well.
void setup_reader_record(const km::Reader& rd, const km_record_t& r) {
    const uint32_t seg = km::seg(r);
    if ((seg != kSegLanes && seg != kSegCount && seg != kSegSetup) || r.a5 == 0 ||
        rd.me() >= r.a5) {
        return;
    }
    const uint32_t t0 = km::wall();
    uint32_t wait = 0, read = 0, bytes = 0;
    auto reserve = [&](uint32_t cb) {
        const uint32_t a = km::wall();
        const bool ok = rd.reserve(cb, 1);
        wait += km::wall() - a;
        return ok;
    };
    auto arrived = [&] {
        return rd.poll(
                   [] {
                       return ncrisc_noc_reads_flushed(noc_index);
                   },
                   rd.args().wait_limit
               ) == km::kGaveNone;
    };
    if (seg == kSegSetup) {
        if (!reserve(kCbSlots) || !reserve(kCbItems)) {
            return;
        }
        noc_async_read(
            su::addr(0, r.a4, rd.me(), kSlotsBytes),
            get_write_ptr(kCbSlots),
            kSlotsBytes
        );
        noc_async_read(
            su::addr(r.a8, r.a7, 0, kItemsBytes), get_write_ptr(kCbItems), kItemsBytes
        );
        if (!arrived()) {
            return;
        }
        if ((r.a6 & kSetupPrefix) != 0 && !slots_from_prefixes(rd, r, arrived)) {
            return;
        }
        cb_push_back(kCbSlots, 1);
        cb_push_back(kCbItems, 1);
    }
    const uint32_t b0 = rd.me() * r.a2 / r.a5, b1 = (rd.me() + 1) * r.a2 / r.a5;
    const InterleavedAddrGen<true> mesh = {
        .bank_base_address = r.a0, .page_size = kMeshBytes
    };
    for (uint32_t b = b0; b <= b1; ++b) {
        if (!reserve(kCbMesh)) {
            return;
        }
        const uint32_t a = km::wall();
        const uint64_t src =
            (b == b0 ? su::addr(r.a8, r.a3, 0, 4 * kCamWords)
                     : get_noc_addr(b - 1, mesh));
        const uint32_t n = b == b0 ? 4 * kCamWords : kMeshBytes;
        noc_async_read(src, get_write_ptr(kCbMesh), n);
        if (!arrived()) {
            return;
        }
        read += km::wall() - a;
        bytes += n;
        cb_push_back(kCbMesh, 1);
    }
    if (r.a9 != 0) {
        volatile tt_l1_ptr uint32_t* pf = km::l1(get_write_ptr(kCbSProf));
        pf[kSProfReader + 0] = wait;
        pf[kSProfReader + 1] = read;
        pf[kSProfReader + 2] = km::wall() - t0;
        pf[kSProfReader + 3] = bytes;
        pf[kSProfReader + 4] = t0;
    }
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        setup_reader_record(rd, r);
    });
}
#endif
