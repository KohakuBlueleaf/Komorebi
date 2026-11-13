// Grid writer: the column broadcast on NoC1 (W blocks, kb x bn tiles, cast up the
// column) and Y: a block is written once the next block's first W block is handed over.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "grid.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto y_args = (
        TensorAccessorArgs<w_args.next_compile_time_args_offset()>()
    );
    km::Writer wr;
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbGridProf));
    const auto core = [&](uint32_t i) {
        return wr.arg(kArgCore + i);
    };
    const auto plan = [&](uint32_t i) {
        return core(kCorePlan + i);
    };
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) == kHandlerGridProf) {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = r.a0, .page_size = kProfBytes
            };
            noc_async_write(
                get_write_ptr(kCbGridProf), get_noc_addr(wr.me(), pages), kProfBytes
            );
            return flushed(wr, km::kStWriter, km::kFlushSent);
        }
        if (km::handler(r) != kHandlerGrid) {
            return true;
        }
        const uint32_t t0 = km::wall();
        const bool broadcasts = r.a3 != kGridProduct;
        const bool store = r.a3 == kGridFull;
        km::Broadcast ln(wr, r, kArgBroadcast);
        if (!ln.ok()) {
            return false;
        }
        if (core(kCoreRank) == 0) {
            return true;
        }
        const uint32_t mt = plan(kPlMt), kt = plan(kPlKt), nt = plan(kPlNt);
        const uint32_t bm = plan(kPlBm), bn = plan(kPlBn), kb = plan(kPlKb);
        const uint32_t nk = plan(kPlNk), nbh = plan(kPlNbh), nbw = plan(kPlNbw);
        const uint32_t sh = plan(kPlSh), sw = plan(kPlSw), sub = sh * sw;
        const uint32_t m0 = core(kCoreM0), n0 = core(kCoreN0);
        const uint32_t m_end = m0 + plan(kPlPcM) < mt ? m0 + plan(kPlPcM) : mt;
        const uint32_t n_end = n0 + plan(kPlPcN) < nt ? n0 + plan(kPlPcN) : nt;
        const auto w = TensorAccessor(w_args, r.a1);
        const auto y = TensorAccessor(y_args, r.a2);
        uint32_t read = 0, broadcast = 0, wait = 0, write = 0;

        // Output block (r0, c0) from kCbGridOut, sh x sw subblocks in compute's order.
        const auto put = [&](uint32_t r0, uint32_t c0) {
            for (uint32_t i = 0; i < bm; i += sh) {
                for (uint32_t j = 0; j < bn; j += sw) {
                    const uint32_t t1 = km::wall();
                    if (!wr.wait_front(kCbGridOut, sub)) {
                        return false;
                    }
                    const uint32_t t2 = km::wall();
                    uint32_t src = get_read_ptr(kCbGridOut);
                    for (uint32_t a = 0; a < sh; ++a) {
                        for (uint32_t b = 0; b < sw; ++b) {
                            const uint32_t row = r0 + i + a, col = c0 + j + b;
                            if (store && row < m_end && col < n_end) {
                                noc_async_write_page(row * nt + col, y, src);
                            }
                            src += kTileBytes;
                        }
                    }
                    if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    cb_pop_front(kCbGridOut, sub);
                    wait += t2 - t1;
                    write += km::wall() - t2;
                }
            }
            return true;
        };

        bool pending = false;
        uint32_t pr0 = 0, pc0 = 0, k = 0;
        for (uint32_t blk = 0; blk < nbh * nbw; ++blk) {
            const uint32_t r0 = m0 + blk / nbw * bm, c0 = n0 + blk % nbw * bn;
            for (uint32_t kblk = 0; broadcasts && kblk < nk; ++kblk, ++k) {
                const uint32_t t1 = km::wall();
                if (!ln.sender()) {
                    if (!ln.arrive(k)) {
                        return false;
                    }
                    broadcast += km::wall() - t1;
                } else {
                    const uint32_t base = ln.reserve(k);
                    if (base == 0) {
                        return false;
                    }
                    const uint32_t t2 = km::wall();
                    const uint32_t k0 = kblk * kb;
                    const uint32_t kc = kt - k0 < kb ? kt - k0 : kb;
                    for (uint32_t i = 0; i < kc; ++i) {
                        const uint32_t page = (k0 + i) * nt + c0;
                        for (uint32_t j = 0; j < bn && c0 + j < nt; ++j) {
                            noc_async_read_page(
                                page + j, w, base + (i * bn + j) * kTileBytes
                            );
                        }
                    }
                    if (!flushed(wr, km::kStWriter, km::kFlushReads)) {
                        return false;
                    }
                    const uint32_t t3 = km::wall();
                    if (!ln.cast(k)) {
                        return false;
                    }
                    read += t3 - t2;
                    broadcast += t2 - t1 + km::wall() - t3;
                }
                if (kblk == 0 && pending) {
                    if (!put(pr0, pc0)) {
                        return false;
                    }
                    pending = false;
                }
            }
            if (!broadcasts && pending && !put(pr0, pc0)) {
                return false;
            }
            pending = r.a3 != kGridBroadcasts;
            pr0 = r0;
            pc0 = c0;
        }
        if (pending && !put(pr0, pc0)) {
            return false;
        }
        if (broadcasts && !ln.finish()) {
            return false;
        }
        km::add_sat(prof + kGpWrRead, read);
        km::add_sat(prof + kGpWrBroadcast, broadcast);
        km::add_sat(prof + kGpWrWait, wait);
        km::add_sat(prof + kGpWrWrite, write);
        km::add_sat(prof + kGpWrBody, km::wall() - t0);
        return true;
    });
}
