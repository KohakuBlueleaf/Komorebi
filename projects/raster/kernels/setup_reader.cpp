// COUNT and SETUP reader: SETUP's slots and items pages, then the camera's page and
// this core's share of the mesh blocks into the mesh buffer, one page each.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#include "setup_noc.hpp"

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
        if (!reserve(kCbSlots)) {
            return;
        }
        noc_async_read(
            su::addr(r.a8, r.a4, rd.me(), kSlotsBytes),
            get_write_ptr(kCbSlots),
            kSlotsBytes
        );
        if (!arrived()) {
            return;
        }
        cb_push_back(kCbSlots, 1);
        if (!reserve(kCbItems)) {
            return;
        }
        noc_async_read(
            su::addr(r.a8, r.a7, 0, kItemsBytes), get_write_ptr(kCbItems), kItemsBytes
        );
        if (!arrived()) {
            return;
        }
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
