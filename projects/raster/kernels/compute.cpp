// Rasterizer compute: the unpack thread waits for list pages and hands each to the pack
// thread, which fills tiles (or runs the arithmetic check) on the vector unit.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_worker.hpp"
#include "raster.h"
#ifdef TRISC_PACK
#include "fill.hpp"
#endif

namespace {

    // A give-up into the writer's gave-up word (the first one is kept) and the stop
    // word.
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

    // Polls ready() for at most kWaitCycles. False: gave up (recorded at `where`).
    template <typename Ready>
    bool bounded(Ready ready, uint32_t where) {
        const uint32_t t0 = km::wall();
        while (!ready()) {
            if (km::wall() - t0 > kWaitCycles) {
                give_up(kGaveWait, where);
                return false;
            }
        }
        return true;
    }

    bool mail_from(ckernel::ThreadId thread, uint32_t where) {
        return bounded(
            [&] {
                return ckernel::mailbox_not_empty(thread);
            },
            where
        );
    }

}  // namespace

#ifdef TRISC_PACK
namespace {

    // The pack thread's view of the list: pages from the unpack thread, slot by slot
    // (nullptr: the wait for a page gave up).
    struct Pages {
        const volatile uint32_t* page = nullptr;
        uint32_t slot = kSlotsPerPage;
        uint32_t wait = 0;

        const volatile uint32_t* next() {
            if (slot == kSlotsPerPage) {
                if (page != nullptr) {
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
                }
                const uint32_t a = fill::wall();
                if (!mail_from(ckernel::ThreadId::UnpackThreadId, kWhereWaitPage)) {
                    return nullptr;
                }
                page = (
                    reinterpret_cast<const volatile uint32_t*>(
                        ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
                    )
                );
                asm volatile("fence" ::: "memory");
                wait += fill::wall() - a;
                slot = 0;
            }
            return page + kSlotWords * slot++;
        }

        // The page holding the end slot goes back with "no more".
        void end() {
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 0);
        }
    };

    uint32_t out_ptr() {
        return get_local_cb_interface(kCbOut).fifo_wr_ptr << 4;
    }

    // An item of `pages` at dst must end inside the output buffer; otherwise the
    // give-up goes to the writer's gave-up word and the stop word is set.
    bool item_fits(uint32_t dst, uint32_t pages) {
        if (dst + pages * kOutPageBytes <=
            (get_local_cb_interface(kCbOut).fifo_limit << 4)) {
            return true;
        }
        give_up(kGaveBadItem, kWhereBadOut);
        return false;
    }

    // Bounded cb_reserve_back on the output buffer.
    bool reserve_out(uint32_t pages) {
        const bool ok = bounded(
            [&] {
                const uint16_t acked = (
                    reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_acked_ptr(kCbOut)))
                );
                const uint16_t used = (
                    get_local_cb_interface(kCbOut).tiles_received - acked
                );
                return get_local_cb_interface(kCbOut).fifo_num_pages - used >= pages;
            },
            kWhereWaitOut
        );
        if (ok) {
            cb_reserve_back(kCbOut, pages);
        }
        return ok;
    }

    // Stores to L1 finished (a load of the last word written, addr + 4 (words - 1),
    // used), then pushed.
    void push_out(uint32_t addr, uint32_t words, uint32_t pages) {
        asm volatile("fence" ::: "memory");
        const uint32_t v = reinterpret_cast<volatile uint32_t*>(addr)[words - 1];
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
        cb_push_back(kCbOut, pages);
    }

    // Words [from, from + words) of the Dst window to L1 at dst: the vector extension
    // when the build has it.
    void copy_out(uint32_t dst, uint32_t from, uint32_t words) {
#if defined(__riscv_vector)
        fill::copy_words_rvv(
            reinterpret_cast<uint32_t*>(dst),
            const_cast<const uint32_t*>(fill::win() + from),
            words
        );
#else
        fill::copy_words(
            reinterpret_cast<volatile uint32_t*>(dst), fill::win() + from, words
        );
#endif
    }

    // Rows [row0, row1) of each plane to the same place in the item at dst.
    void copy_band(uint32_t dst, uint32_t row0, uint32_t row1) {
        const uint32_t at = kRowBytes / 4 * row0, words = kRowBytes / 4 * (row1 - row0);
        for (uint32_t p = 0; p < kPlanes; ++p) {
            copy_out(dst + 4 * (p * kPlaneWords + at), p * kPlaneWords + at, words);
        }
    }

    // False: an output item did not fit (recorded); the list is abandoned.
    bool fill_list(fill::Prof& pf) {
        constexpr uint32_t bg[3] = {0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A};
        Pages in;
        const uint32_t t0 = fill::wall();
        for (;;) {
            const volatile uint32_t* h = in.next();
            if (h == nullptr) {
                return false;
            }
            if (h[0] == kMarkEnd) {
                break;
            }
            const uint32_t tile = h[kHdrTile], n = h[kHdrCount];
            const int32_t row0 = h[kHdrRow0], row1 = h[kHdrRow1];
            if (!(0 <= row0 && row0 < row1 && row1 <= 32 && tile < 1024)) {
                give_up(kGaveBadItem, kWhereBadHeader);
                in.end();
                return false;
            }
            const uint32_t tx = tile % 32, ty = tile / 32;
            uint32_t a = fill::wall();
            fill::init_tile(tx, ty, bg);
            uint32_t b = fill::wall();
            pf.init += b - a;
            for (uint32_t i = 0; i < n; ++i) {
                const volatile uint32_t* rec = in.next();
                if (rec == nullptr) {
                    return false;
                }
                a = fill::wall();
                uint32_t w[kSlotWords];
#if defined(__riscv_vector)
                fill::copy_words_rvv(w, const_cast<const uint32_t*>(rec), kSlotWords);
#else
                fill::fetch(w, rec);
#endif
                const int32_t y0 = static_cast<int32_t>(w[kRecY0]) - 32 * ty;
                const int32_t y1 = static_cast<int32_t>(w[kRecY1]) - 32 * ty;
                const int32_t r0 = y0 < row0 ? row0 : y0;
                const int32_t r1 = y1 > row1 - 1 ? row1 - 1 : y1;
                fill::load_triangle(w);
                b = fill::wall();
                pf.tri += b - a;
                for (int32_t r = r0; r <= r1; ++r) {
                    fill::row(r);
                    pf.row_count++;
                }
                pf.rows += fill::wall() - b;
                pf.tris++;
            }
            a = fill::wall();
            ckernel::tensix_sync();
            b = fill::wall();
            pf.drain += b - a;
            if (!reserve_out(kPlanes)) {
                in.end();
                return false;
            }
            a = fill::wall();
            pf.wait_out += a - b;
            const uint32_t dst = out_ptr();
            if (!item_fits(dst, kPlanes)) {
                in.end();
                return false;
            }
            copy_band(dst, row0, row1);
            push_out(dst, (kPlanes - 1) * kPlaneWords + kRowBytes / 4 * row1, kPlanes);
            pf.out += fill::wall() - a;
            pf.tiles++;
        }
        in.end();
        pf.wait_in = in.wait;
        pf.total = fill::wall() - t0;
        return true;
    }

    bool arith_list(fill::Prof& pf) {
        Pages in;
        const uint32_t t0 = fill::wall();
        for (;;) {
            in.slot = kSlotsPerPage;
            const volatile uint32_t* p = in.next();
            if (p == nullptr) {
                return false;
            }
            if (p[0] == kMarkEnd) {
                break;
            }
            for (uint32_t k = 0; k < kArithIn; ++k) {
                if (k > 0) {
                    in.slot = kSlotsPerPage;
                    p = in.next();
                    if (p == nullptr) {
                        return false;
                    }
                }
                fill::copy_words(fill::win() + kArithLanes * k, p, kArithLanes);
            }
            fill::order(kArithLanes * kArithIn - 1);
            fill::arith_chunk();
            ckernel::tensix_sync();
            if (!reserve_out(kArithOut)) {
                in.end();
                return false;
            }
            const uint32_t dst = out_ptr();
            if (!item_fits(dst, kArithOut)) {
                in.end();
                return false;
            }
            fill::copy_words(
                reinterpret_cast<volatile uint32_t*>(dst),
                fill::win() + kArithLanes * kArithIn,
                kArithOut * kPlaneWords
            );
            push_out(dst, kArithOut * kPlaneWords, kArithOut);
            pf.tiles++;
        }
        in.end();
        pf.wait_in = in.wait;
        pf.total = fill::wall() - t0;
        return true;
    }

    // The profile item (`per` pages, so items stay aligned in the output CB): the pack
    // thread's counters at words 0 to 10 of its first page.
    void profile(const fill::Prof& pf, uint32_t per) {
        if (!reserve_out(per)) {
            return;
        }
        const uint32_t dst = out_ptr();
        if (!item_fits(dst, per)) {
            return;
        }
        volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(dst);
        const uint32_t v[11] = {
            pf.wait_in,
            pf.init,
            pf.tri,
            pf.rows,
            pf.drain,
            pf.out,
            pf.wait_out,
            pf.tiles,
            pf.tris,
            pf.row_count,
            pf.total
        };
        for (uint32_t k = 0; k < 32; ++k) {
            w[k] = k < 11 ? v[k] : 0;
        }
        push_out(dst, 32, per);
    }

}  // namespace
#endif

// FILL, ARITH and SPIN of one record; the frame program's compute kernel calls it as
// well.
void fill_compute_record(const km_record_t& r) {
    const uint32_t seg = km::seg(r);
    if (seg == kSegSpin) {
        const uint32_t t0 = km::wall();
        while (km::wall() - t0 < r.a0) {
            ;
        }
        return;
    }
    if (seg != kSegFill && seg != kSegArith) {
        return;
    }
#ifdef TRISC_UNPACK
    // Hands each page to the pack thread; frees it when the pack thread is done.
    for (;;) {
        const bool page = bounded(
            [] {
                const uint16_t received = (
                    reg_read(
                        reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(kCbIn))
                    )
                );
                const uint16_t acked = get_local_cb_interface(kCbIn).tiles_acked;
                return static_cast<uint16_t>(received - acked) >= 1;
            },
            kWhereWaitList
        );
        if (!page) {
            break;
        }
        cb_wait_front(kCbIn, 1);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(kCbIn, 0)
        );
        if (!mail_from(ckernel::ThreadId::PackThreadId, kWhereWaitDone)) {
            break;
        }
        const uint32_t more = (ckernel::mailbox_read(ckernel::ThreadId::PackThreadId));
        cb_pop_front(kCbIn, 1);
        if (more == 0) {
            break;
        }
    }
#endif
#ifdef TRISC_PACK
    fill::Prof pf;
    fill::setup();
    bool ok;
    if (seg == kSegFill) {
        fill::edges(true);
        ok = fill_list(pf);
    } else {
        ok = arith_list(pf);
    }
    fill::finish();
    if (ok) {
        profile(pf, seg == kSegFill ? kPlanes : kArithOut);
    }
#endif
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    cp.run([&](const km_record_t& r) {
        fill_compute_record(r);
    });
}
#endif
