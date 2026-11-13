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

    // The writer's profile counters for one record (cycles, bytes, pages).
    struct WProf {
        uint32_t wait = 0, write = 0, bytes = 0, pages = 0;
    };

    // PLACE through HIST's space (HIST is done): the region's words, then this core's
    // directory page, its end slot and 32 headers at a time.
    template <typename Arrived, typename Flushed>
    bool place(
        const km::Writer& wr, const km_command_t& r, Arrived arrived, Flushed flushed
    ) {
        const uint32_t space = get_write_ptr(kCbHist);
        noc_async_read(su::addr(r.a8, r.a0, 0, 0, 0), space, kItemsBytes);
        noc_async_read(
            su::addr(r.a8, r.a0, 0, 0, kFrameWordsAt),
            space + kFrameWordsAt,
            kItemsRegionBytes - kFrameWordsAt
        );
        if (!arrived()) {
            return false;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* items = km::l1(space);
        const volatile tt_l1_ptr uint16_t*
            rows = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(space + kItemRows));
        volatile tt_l1_ptr uint32_t* iw = km::l1(space + kItemWordsAt);
        volatile tt_l1_ptr uint32_t* cw = km::l1(space + kCoreWordsAt);
        volatile tt_l1_ptr uint32_t* fw = km::l1(space + kFrameWordsAt);
        // each item's tile, in the base words' place (unused by PLACE)
        volatile tt_l1_ptr uint16_t*
            tile_of = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(space + kBaseAt));
        uint32_t n_items = 0;
        for (uint32_t t = 0; t < kTiles; ++t) {
            const uint32_t first = items[t] & 0xFFFF, n = items[t] >> 16;
            for (uint32_t i = first; i < first + n && i < kSlotsBytes / 4; ++i) {
                tile_of[i] = t;
            }
            n_items = first + n;
        }
        if (n_items > kSlotsBytes / 4 || r.a6 > kMaxFillCores) {
            give_up(kGaveBadPlace, 1);
            return true;
        }
        const InterleavedAddrGen<true> lists = {
            .bank_base_address = r.a7, .page_size = kListPageBytes
        };
        const uint32_t me = wr.me(), out = space + kItemsRegionBytes;
        static_assert(
            kItemsRegionBytes + kDirPageBytes + 128 + kOutBytes <= kHistBytes
        );
        auto slot_of = [&](uint32_t at, uint32_t mark, uint32_t i) {
            volatile tt_l1_ptr uint32_t* w = km::l1(at);
            for (uint32_t k = 0; k < 32; ++k) {
                w[k] = 0;
            }
            w[0] = mark;
            if (mark == kLayMarkTile) {
                w[kLayHdrTile] = tile_of[i];
                w[kLayHdrCount] = iw[kItemWords * i] >> 8;
                w[kLayHdrRow0] = rows[i] & 0xFF;
                w[kLayHdrRow1] = rows[i] >> 8;
            }
        };
        if (me < r.a6) {
            volatile tt_l1_ptr uint32_t* d = km::l1(out);
            for (uint32_t k = 0; k < kDirPageBytes / 4; ++k) {
                d[k] = 0;
            }
            uint32_t n = 0;
            for (uint32_t i = 0; i < n_items && n < kLayDirFrame - kLayDirIds; ++i) {
                const uint32_t w = iw[kItemWords * i];
                if ((w & 0xFF) != me) {
                    continue;
                }
                uint32_t id = tile_of[i];
                if (rows[i] != (32u << 8)) {
                    id |=
                        ((rows[i] & 0xFF) << kLayRow0Shift |
                         (rows[i] >> 8) << kLayRow1Shift);
                }
                d[kLayDirIds + n++] = (w >> 8) == 0 ? id | kLayItemEmpty : id;
            }
            d[0] = cw[kCoreWords * me];
            d[1] = cw[kCoreWords * me + 1];
            d[2] = n;
            for (uint32_t k = 0; k < 16; ++k) {
                d[kLayDirFrame + k] = fw[k];
            }
            asm volatile("fence" ::
                             : "memory");  // the page's stores before the NoC reads it
            noc_async_write(out, su::addr(0, r.a1, me, kDirPageBytes), kDirPageBytes);
            const uint32_t end = cw[kCoreWords * me + 3];
            const uint32_t at = out + kDirPageBytes;
            slot_of(at, kLayMarkEnd, 0);
            asm volatile("fence" ::: "memory");
            noc_async_write(at, get_noc_addr(end / 16, lists, (end % 16) * 128), 128);
            if (!flushed()) {
                return false;
            }
        }
        // headers, 32 at a time
        uint32_t k = 0;
        const uint32_t hdr = out + kDirPageBytes + 128;
        for (uint32_t i = me; i < n_items; i += r.a5) {
            if ((iw[kItemWords * i] >> 8) == 0) {
                continue;
            }
            const uint32_t at = hdr + 128 * k, s = iw[kItemWords * i + 1];
            slot_of(at, kLayMarkTile, i);
            asm volatile("fence" ::: "memory");
            noc_async_write(at, get_noc_addr(s / 16, lists, (s % 16) * 128), 128);
            if (++k == kOutBytes / 128) {
                if (!flushed()) {
                    return false;
                }
                k = 0;
            }
        }
        return flushed();
    }

}  // namespace

bool setup_writer_pass(const km::Writer& wr, const km_command_t& r, WProf& pf);

// The passes' writing for one record; the frame program's writer calls it as well.
// With a9, the threads' counters go to the core's profile page.
bool setup_writer_command(const km::Writer& wr, const km_command_t& r) {
    const uint32_t handler = km::handler(r);
    if ((handler != kHandlerLanes && handler != kHandlerCount &&
         handler != kHandlerSetup && handler != kHandlerHist &&
         handler != kHandlerPlace && handler != kHandlerSum &&
         handler != kHandlerHsum) ||
        r.a5 == 0 || wr.me() >= r.a5) {
        return true;
    }
    const uint32_t t0 = km::wall();
    WProf pf;
    const bool ok = setup_writer_pass(wr, r, pf);
    if (ok && (handler == kHandlerCount || handler == kHandlerSetup) &&
        (r.a2 & kRanges) != 0) {
        // the pass's ticks on this core, for the next frame's ranges
        const uint32_t line = get_write_ptr(kCbSProf) + 128;
        km::l1(line)[0] = km::wall() - t0;
        invalidate_l1_cache();
        noc_async_write(
            line,
            su::addr(
                r.a8,
                r.a3,
                0,
                0,
                kCostsAt + 128 * wr.me() + 64 * (handler - kHandlerCount)
            ),
            64
        );
        if (wr.poll(
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                wr.args().wait_limit
            ) != km::kGaveNone) {
            return false;
        }
    }
    if (!ok || r.a9 == 0 || handler < kHandlerCount || handler > kHandlerPlace) {
        return ok;
    }
    invalidate_l1_cache();
    const uint32_t page = get_write_ptr(kCbSProf);
    volatile tt_l1_ptr uint32_t* p = km::l1(page);
    p[kSProfWriter + 0] = pf.wait;
    p[kSProfWriter + 1] = pf.write;
    p[kSProfWriter + 2] = km::wall() - t0;
    p[kSProfWriter + 3] = pf.bytes;
    p[kSProfWriter + 4] = t0;
    p[kSProfWriter + 5] = pf.pages;
    noc_async_write(
        page,
        su::addr(0, r.a9, wr.me(), kSProfPageBytes, 128 * (handler - kHandlerCount)),
        128
    );
    return wr.poll(
               [] {
                   return ncrisc_noc_nonposted_writes_flushed(noc_index);
               },
               wr.args().wait_limit
           ) == km::kGaveNone;
}

// One record's writing, its counters in pf.
bool setup_writer_pass(const km::Writer& wr, const km_command_t& r, WProf& pf) {
    const uint32_t handler = km::handler(r);
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
    if (handler == kHandlerCount) {
        const uint32_t a = km::wall();
        if (!wr.wait_front(kCbCount, 1)) {
            return true;
        }
        const uint32_t b = km::wall();
        pf.wait += b - a;
        pf.bytes += kCountBytes;
        noc_async_write(
            get_read_ptr(kCbCount), su::addr(0, r.a1, wr.me(), kCountBytes), kCountBytes
        );
        const bool ok = flushed();
        pf.write += km::wall() - b;
        cb_pop_front(kCbCount, 1);
        return ok;
    }
    if (handler == kHandlerHist) {
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
                hist + 64 * t, su::addr(0, r.a1, wr.me(), r.a4, 64 * h), 64
            );
        }
        return flushed();
    }
    if ((handler == kHandlerSum || handler == kHandlerHsum) && r.a8 == 0) {
        give_up(kGaveBadPlace, handler);
        return true;
    }
    if (handler == kHandlerHsum &&
        (r.a0 > kHprefixPageBytes / 128 || 64 * r.a0 > r.a4)) {
        give_up(kGaveBadHeavy, r.a0);
        return true;
    }
    if (handler == kHandlerSum) {
        // a1 counters, a2 slots pages, a3 totals, a6 counting cores; 32 cores at a time
        // through one out page: their entries, then their rows
        constexpr uint32_t kBatch = 32;
        const uint32_t buf = get_write_ptr(kCbOut);
        for (uint32_t k = wr.me(); k < kTiles / 16; k += r.a5) {
            uint32_t ent[16] = {}, rows[16] = {};
            for (uint32_t c0 = 0; c0 < r.a6; c0 += kBatch) {
                const uint32_t n = r.a6 - c0 < kBatch ? r.a6 - c0 : kBatch;
                for (uint32_t c = 0; c < n; ++c) {
                    const uint64_t at = su::addr(0, r.a1, c0 + c, kCountBytes, 64 * k);
                    noc_async_read(at, buf + 64 * c, 64);
                    noc_async_read(at + 4 * kTiles, buf + 2048 + 64 * c, 64);
                }
                if (!arrived()) {
                    return false;
                }
                invalidate_l1_cache();
                for (uint32_t c = 0; c < n; ++c) {
                    volatile tt_l1_ptr uint32_t* e = km::l1(buf + 64 * c);
                    volatile tt_l1_ptr uint32_t* w = km::l1(buf + 2048 + 64 * c);
                    for (uint32_t j = 0; j < 16; ++j) {
                        const uint32_t x = e[j];
                        e[j] = ent[j];
                        ent[j] += x;
                        rows[j] += w[j];
                    }
                    asm volatile("fence" ::: "memory");
                    noc_async_write(
                        buf + 64 * c, su::addr(0, r.a2, c0 + c, kSlotsBytes, 64 * k), 64
                    );
                }
                if (!flushed()) {
                    return false;
                }
            }
            volatile tt_l1_ptr uint32_t* t = km::l1(buf);
            for (uint32_t j = 0; j < 16; ++j) {
                t[j] = ent[j];
                t[16 + j] = rows[j];
            }
            asm volatile("fence" ::: "memory");
            noc_async_write(buf, su::addr(r.a8, r.a3, 0, 0, 64 * k), 64);
            noc_async_write(
                buf + 64, su::addr(r.a8, r.a3, 0, 0, 4 * kTiles + 64 * k), 64
            );
            if (!flushed()) {
                return false;
            }
        }
        return true;
    }
    if (handler == kHandlerHsum) {
        // 16 cores at a time: their counts, then the 32 words each one's page gets
        // (README.md)
        constexpr uint32_t kBatch = 16;
        const uint32_t buf = get_write_ptr(kCbOut), out = buf + 64 * kBatch;
        for (uint32_t h = wr.me(); h < r.a0; h += r.a5) {
            uint32_t sum[32] = {};
            for (uint32_t c0 = 0; c0 < r.a6; c0 += kBatch) {
                const uint32_t n = r.a6 - c0 < kBatch ? r.a6 - c0 : kBatch;
                for (uint32_t c = 0; c < n; ++c) {
                    noc_async_read(
                        su::addr(0, r.a1, c0 + c, r.a4, 64 * h), buf + 64 * c, 64
                    );
                }
                if (!arrived()) {
                    return false;
                }
                invalidate_l1_cache();
                for (uint32_t c = 0; c < n; ++c) {
                    volatile tt_l1_ptr uint16_t*
                        in = (
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + 64 * c)
                        );
                    volatile tt_l1_ptr uint32_t* o = km::l1(out + 128 * c);
                    uint32_t f = 0, l = 0;
                    for (uint32_t g = 0; g < 16; ++g) {
                        f += sum[g];
                        l += sum[16 + g];
                        o[g] = f;
                        o[16 + g] = l;
                    }
                    for (uint32_t g = 0; g < 32; ++g) {
                        sum[g] += in[g];
                    }
                    asm volatile("fence" ::: "memory");
                    noc_async_write(
                        out + 128 * c,
                        su::addr(0, r.a2, c0 + c, kHprefixPageBytes, 128 * h),
                        128
                    );
                }
                if (!flushed()) {
                    return false;
                }
            }
            volatile tt_l1_ptr uint32_t* t = km::l1(buf);
            for (uint32_t g = 0; g < 32; ++g) {
                t[g] = sum[g];
            }
            asm volatile("fence" ::: "memory");
            noc_async_write(
                buf, su::addr(r.a8, r.a3, 0, 0, kSumsHeavyAt + 128 * h), 128
            );
            if (!flushed()) {
                return false;
            }
        }
        return true;
    }
    if (handler == kHandlerPlace) {
        if (r.a8 == 0) {
            give_up(kGaveBadPlace, 0);
            return true;
        }
        return place(wr, r, arrived, flushed);
    }
    if (handler == kHandlerSetup) {
        const InterleavedAddrGen<true> lists = {
            .bank_base_address = r.a1, .page_size = kListPageBytes
        };
        for (;;) {
            const uint32_t a = km::wall();
            if (!wr.wait_front(kCbOut, 1, 4)) {
                return true;
            }
            const uint32_t b = km::wall();
            pf.wait += b - a;
            const uint32_t page = get_read_ptr(kCbOut);
            volatile tt_l1_ptr uint32_t* p = km::l1(page);
            const uint32_t head = p[0], n = head & ~kOutLast;
            pf.bytes += 128 * n;
            ++pf.pages;
            for (uint32_t i = 0; i < n && i < kOutEntries; ++i) {
                const uint32_t s = p[1 + i];
                noc_async_write(
                    page + 128 * (1 + i),
                    get_noc_addr(s / 16, lists, (s % 16) * 128),
                    128
                );
            }
            const bool ok = flushed();
            pf.write += km::wall() - b;
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
    wr.run([&](const km_command_t& r) {
        return setup_writer_command(wr, r);
    });
}
#endif
