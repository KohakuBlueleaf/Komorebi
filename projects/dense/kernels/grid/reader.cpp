// Grid reader: the row broadcast, on NoC0. The row's sender reads each A block (bm x kb
// tiles, row stride kb) from DRAM and casts it along the row; the others take it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "grid.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    km::Reader rd;
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbGridProf));
    for (uint32_t w = 0; w < kGpWords; ++w) {
        prof[w] = 0;
    }
    const auto core = [&](uint32_t i) {
        return rd.arg(kArgCore + i);
    };
    const auto plan = [&](uint32_t i) {
        return core(kCorePlan + i);
    };
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerGrid || r.a3 == kGridProduct) {
            return;
        }
        const uint32_t t0 = km::wall();
        km::Broadcast ln(rd, r, kArgBroadcast);
        if (!ln.ok() || !ln.included()) {
            return;
        }
        const uint32_t mt = plan(kPlMt), kt = plan(kPlKt), bm = plan(kPlBm);
        const uint32_t kb = plan(kPlKb), nk = plan(kPlNk);
        const uint32_t nbh = plan(kPlNbh), nbw = plan(kPlNbw);
        const uint32_t m0 = core(kCoreM0);
        const auto a = TensorAccessor(a_args, r.a0);
        uint32_t read = 0, broadcast = 0, k = 0;
        for (uint32_t blk = 0; blk < nbh * nbw; ++blk) {
            const uint32_t r0 = m0 + blk / nbw * bm;
            for (uint32_t kblk = 0; kblk < nk; ++kblk, ++k) {
                const uint32_t t1 = km::wall();
                if (!ln.sender()) {
                    if (!ln.arrive(k)) {
                        return;
                    }
                    broadcast += km::wall() - t1;
                    continue;
                }
                const uint32_t base = ln.reserve(k);
                if (base == 0) {
                    return;
                }
                const uint32_t t2 = km::wall();
                const uint32_t k0 = kblk * kb;
                const uint32_t kc = kt - k0 < kb ? kt - k0 : kb;
                for (uint32_t i = 0; i < bm && r0 + i < mt; ++i) {
                    const uint32_t page = (r0 + i) * kt + k0;
                    for (uint32_t j = 0; j < kc; ++j) {
                        noc_async_read_page(
                            page + j, a, base + (i * kb + j) * kTileBytes
                        );
                    }
                }
                if (!flushed(rd, km::kStReader, km::kFlushReads)) {
                    return;
                }
                const uint32_t t3 = km::wall();
                if (!ln.cast(k)) {
                    return;
                }
                read += t3 - t2;
                broadcast += t2 - t1 + km::wall() - t3;
            }
        }
        if (!ln.finish()) {
            return;
        }
        km::add_sat(prof + kGpRdRead, read);
        km::add_sat(prof + kGpRdBroadcast, broadcast);
        km::add_sat(prof + kGpRdBody, km::wall() - t0);
    });
}
