// Rasterizer reader: streams this core's list pages from DRAM into the input buffer,
// as its directory page names them. Its cycle counts go to the end of that page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "raster.h"

// Bounded noc_async_read_barrier. False: gave up (recorded in the reader's word).
bool read_done(const km::Reader& rd, uint32_t seq) {
    const uint32_t code = rd.poll(
        [] {
            return ncrisc_noc_reads_flushed(noc_index);
        },
        rd.args().wait_limit
    );
    if (code == km::kGaveNone) {
        return true;
    }
    rd.gave_up(km::kStReader, code, kWhereReadDone, 0, seq);
    return false;
}

// FILL and ARITH of one record; the frame program's reader calls it as well.
void fill_reader_command(const km::Reader& rd, const km_command_t& r) {
    if (km::handler(r) != kHandlerFill && km::handler(r) != kHandlerArith) {
        return;
    }
    const uint32_t t0 = km::wall();
    const uint32_t dir = get_write_ptr(kCbDirReader);
    volatile tt_l1_ptr uint32_t* d = km::l1(dir);
    const InterleavedAddrGen<true> dirs = {
        .bank_base_address = r.a0, .page_size = kDirBytes
    };
    noc_async_read(get_noc_addr(rd.me(), dirs), dir, kDirBytes);
    if (!read_done(rd, r.seq)) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t first = d[kDirFirst], pages = d[kDirPages];
    const InterleavedAddrGen<true> lists = {
        .bank_base_address = r.a1, .page_size = kInPageBytes
    };
    uint32_t wait = 0, read = 0;
    for (uint32_t p = 0; p < pages; ++p) {
        const uint32_t a = km::wall();
        if (!rd.reserve(kCbIn, 1)) {
            return;
        }
        const uint32_t b = km::wall();
        noc_async_read(
            get_noc_addr(first + p, lists), get_write_ptr(kCbIn), kInPageBytes
        );
        if (!read_done(rd, r.seq)) {
            return;
        }
        cb_push_back(kCbIn, 1);
        wait += b - a;
        read += km::wall() - b;
    }
    d[kRdWords + 0] = wait;
    d[kRdWords + 1] = read;
    d[kRdWords + 2] = km::wall() - t0;
    d[kRdWords + 3] = pages;
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_command_t& r) {
        fill_reader_command(rd, r);
    });
}
#endif
