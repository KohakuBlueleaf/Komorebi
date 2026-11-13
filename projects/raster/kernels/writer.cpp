// Rasterizer writer: each item's output pages (a3 per item) to the frame, then the
// profile page (compute's counters, then the reader's and its own) to a4.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "raster.h"

// An item of `per` pages at the read pointer must end inside the output buffer;
// otherwise the give-up is recorded and the stop flag set.
bool item_fits(const km::Writer& wr, uint32_t per, uint32_t seq) {
    if (get_read_ptr(kCbOut) + per * kOutPageBytes <=
        get_local_cb_interface(kCbOut).fifo_limit) {
        return true;
    }
    wr.gave_up(km::kStWriter, kGaveBadItem, kWhereBadSend, 0, seq);
    *wr.ctl(KM_CTL_STOP) = 1;
    return false;
}

// Bounded NoC waits: reads arrived, or writes sent. False: gave up (recorded).
template <typename Ready>
bool noc_done(const km::Writer& wr, Ready ready, uint32_t where, uint32_t seq) {
    const uint32_t code = wr.poll(ready, wr.args().wait_limit);
    if (code == km::kGaveNone) {
        return true;
    }
    wr.gave_up(km::kStWriter, code, where, 0, seq);
    return false;
}

bool read_done(const km::Writer& wr, uint32_t seq) {
    return noc_done(
        wr,
        [] {
            return ncrisc_noc_reads_flushed(noc_index);
        },
        kWhereReadDone,
        seq
    );
}

bool writes_sent(const km::Writer& wr, uint32_t seq) {
    return noc_done(
        wr,
        [] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        },
        kWhereFlush,
        seq
    );
}

// FILL and ARITH of one record; the frame program's writer calls it as well.
bool fill_writer_command(const km::Writer& wr, const km_command_t& r) {
    if (km::handler(r) != kHandlerFill && km::handler(r) != kHandlerArith) {
        return true;
    }
    const uint32_t t0 = km::wall();
    const uint32_t dir = get_write_ptr(kCbDirWriter);
    volatile tt_l1_ptr uint32_t* d = km::l1(dir);
    const InterleavedAddrGen<true> dirs = {
        .bank_base_address = r.a0, .page_size = kDirBytes
    };
    noc_async_read(get_noc_addr(wr.me(), dirs), dir, kDirBytes);
    if (!read_done(wr, r.seq)) {
        return false;
    }
    invalidate_l1_cache();
    const uint32_t items = d[kDirItems], per = r.a3, chunk = d[kDirFrame];
    // An item's pages lie in one of the frame's tensors, each named in the table.
    if (chunk == 0 || chunk % per != 0 || items > kDirMaxItems) {
        wr.gave_up(km::kStWriter, kGaveBadItem, kWhereBadSend, 0, r.seq);
        *wr.ctl(KM_CTL_STOP) = 1;
        return false;
    }
    auto frame_noc = [&](uint32_t page, uint32_t at) {
        const InterleavedAddrGen<true> part = {
            .bank_base_address = d[kDirFrame + 1 + page / chunk],
            .page_size = kOutPageBytes
        };
        return get_noc_addr(page % chunk, part, at);
    };
    // The empty tile's planes: depth 0, the background colour, triangle -1.
    const uint32_t empty = get_write_ptr(kCbEmpty);
    if (km::handler(r) == kHandlerFill) {
        constexpr uint32_t value[kPlanes] = {
            0, 0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A, 0xBF800000
        };
        volatile tt_l1_ptr uint32_t* e = km::l1(empty);
        for (uint32_t w = 0; w < kPlanes * kPlaneWords; ++w) {
            e[w] = value[w / kPlaneWords];
        }
    }
    uint32_t wait = 0, write = 0;
    for (uint32_t i = 0; i < items; ++i) {
        const uint32_t id = d[kDirIds + i];
        const bool none = (id & kItemEmpty) != 0;
        const uint32_t a = km::wall();
        if (!none && !wr.wait_front(kCbOut, per)) {
            return true;
        }
        if (!none && !item_fits(wr, per, r.seq)) {
            return false;
        }
        const uint32_t b = km::wall();
        const uint32_t page = (id & kItemSlot) * per;
        if (page / chunk >= kDirFrameMax) {
            wr.gave_up(km::kStWriter, kGaveBadItem, kWhereBadSend, 0, r.seq);
            *wr.ctl(KM_CTL_STOP) = 1;
            return false;
        }
        const uint32_t row0 = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t row1 = id >> kItemRow1Shift & kItemRowMask;
        const uint32_t at = row0 * kRowBytes;
        const uint32_t bytes = row1 > 0 ? (row1 - row0) * kRowBytes : kOutPageBytes;
        uint32_t src = (none ? empty : get_read_ptr(kCbOut)) + at;
        for (uint32_t p = 0; p < per; ++p) {
            noc_async_write(src, frame_noc(page + p, at), bytes);
            src += kOutPageBytes;
        }
        if (!writes_sent(wr, r.seq)) {
            return false;
        }
        if (!none) {
            cb_pop_front(kCbOut, per);
        }
        wait += b - a;
        write += km::wall() - b;
    }
    // The profile item: compute filled words [0, 32); reader and writer follow.
    if (!wr.wait_front(kCbOut, per)) {
        return true;
    }
    if (!item_fits(wr, per, r.seq)) {
        return false;
    }
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_read_ptr(kCbOut));
    volatile tt_l1_ptr uint32_t* rd = km::l1(get_write_ptr(kCbDirReader));
    for (uint32_t k = 0; k < 4; ++k) {
        prof[kProfReader + k] = rd[kRdWords + k];
    }
    prof[kProfWriter + 0] = wait;
    prof[kProfWriter + 1] = write;
    prof[kProfWriter + 2] = km::wall() - t0;
    prof[kProfWriter + 3] = items;
    const InterleavedAddrGen<true> profs = {
        .bank_base_address = r.a4, .page_size = kOutPageBytes
    };
    noc_async_write(get_read_ptr(kCbOut), get_noc_addr(wr.me(), profs), kOutPageBytes);
    if (!writes_sent(wr, r.seq)) {
        return false;
    }
    cb_pop_front(kCbOut, per);
    return true;
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_command_t& r) {
        return fill_writer_command(wr, r);
    });
}
#endif
