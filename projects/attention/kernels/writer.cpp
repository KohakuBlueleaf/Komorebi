// Attention writer (NoC1): its copy of the job table, each job's rows to O, and with
// BROADCAST_ON_WRITER first the job's K / V blocks (broadcast). PROF: the profile page
// out.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "attention.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

namespace {

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t KB = get_compile_time_arg_val(3);

    // The command's jobs, `passes` times: each job's blocks through `ln` (null: none),
    // then its rows written (full), popped (compute only) or none. False: gave up.
    template <typename Accessor>
    bool run_jobs(
        const km::Writer& wr,
        const km_command_t& r,
        uint32_t table,
        const Accessor& o,
        const Accessor& k,
        const Accessor& v,
        km::Broadcast* ln,
        volatile tt_l1_ptr uint32_t* prof,
        uint32_t t0
    ) {
        const uint32_t jobs = km::l1(table)[kTabJobs];
        const uint32_t passes = r.a6 == 0 ? 1 : r.a6;
        if (jobs > kJobsMax) {
            return false;
        }
        const uint32_t each = r.a5 == kModeBroadcastOnly ? 0 : 1;
        const bool writes = r.a5 == kModeFull;
        uint32_t c[kProfWords] = {};
        uint32_t n = 0;  // broadcast blocks of this command so far
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (uint32_t j = r.a7; j < jobs_end(r, jobs); ++j) {
                const volatile tt_l1_ptr uint32_t* w = (
                    km::l1(table + 4 * (kTabFirst + kJobWords * j))
                );
                if (ln != nullptr && !broadcast_job<DT, DV, KB>(*ln, k, v, w, n, c)) {
                    return false;
                }
                const uint32_t o0 = w[kJobO], rows = w[kJobRows] * each;
                const uint32_t stride = w[kJobStride];
                for (uint32_t i = 0; i < rows; ++i) {
                    const uint32_t t1 = km::wall();
                    if (!wr.wait_front(kCbOut, DV)) {
                        return false;  // nothing to report: the tile is stopping
                    }
                    const uint32_t t2 = km::wall();
                    const uint32_t src = get_read_ptr(kCbOut);
                    for (uint32_t col = 0; writes && col < DV; ++col) {
                        noc_async_write_page(
                            o0 + i * stride * DV + col, o, src + col * kTileBytes
                        );
                    }
                    if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    cb_pop_front(kCbOut, DV);
                    c[kPrWrWait] += t2 - t1;
                    c[kPrWrWrite] += km::wall() - t2;
                }
            }
        }
        if (ln != nullptr) {
            const uint32_t t1 = km::wall();
            ln->finish();
            c[kPrRdFinish] += km::wall() - t1;
            for (uint32_t i = kPrRdKvFree; i <= kPrRdFinish; ++i) {
                km::add_sat(
                    prof + i, c[i]
                );  // the broadcast's words, the reader's otherwise
            }
        }
        km::add_sat(prof + kPrWrWait, c[kPrWrWait]);
        km::add_sat(prof + kPrWrWrite, c[kPrWrWrite]);
        km::add_sat(prof + kPrWrBody, km::wall() - t0);
        return true;
    }

}  // namespace

void kernel_main() {
    constexpr auto tiles = TensorAccessorArgs<5>();  // K, V, O: bf16 tiles in DRAM
    km::Writer wr;
    // The table copy's page: reserved once, used in place, never pushed.
    cb_reserve_back(kCbWriterJobs, 1);
    const uint32_t table = get_write_ptr(kCbWriterJobs);
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbProf));
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) == kHandlerProf) {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = r.a0, .page_size = kProfBytes
            };
            const uint32_t rank = wr.me() - km::first(r);
            noc_async_write(
                get_write_ptr(kCbProf), get_noc_addr(rank, pages), kProfBytes
            );
            if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
                return false;
            }
            for (uint32_t w = 0; r.a1 != 0 && w < kProfWords; ++w) {
                prof[w] = 0;
            }
            return true;
        }
        if (km::handler(r) != kHandlerAttend) {
            return true;
        }
        const uint32_t t0 = km::wall();
        if (!read_table(wr, km::kStWriter, r.a4, wr.me() - km::first(r), table)) {
            return false;
        }
        const auto k = TensorAccessor(tiles, r.a1);
        const auto v = TensorAccessor(tiles, r.a2);
        const auto o = TensorAccessor(tiles, r.a3);
#ifdef BROADCAST_ON_WRITER
        if (r.a5 != kModeComputeOnly) {
            km::Broadcast ln(wr, r, kArgBroadcast);
            return (
                ln.ok() && ln.included() &&
                run_jobs(wr, r, table, o, k, v, &ln, prof, t0)
            );
        }
#endif
        return run_jobs(wr, r, table, o, k, v, nullptr, prof, t0);
    });
}
