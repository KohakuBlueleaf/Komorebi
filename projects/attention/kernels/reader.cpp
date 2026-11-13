// Attention reader (NoC0): the member's job table to compute, then per job its Q rows
// and, unless LANE_ON_WRITER, the job's K / V blocks (the lane's sender casts them).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "attention.h"
#include "komorebi/device/km_lane.hpp"
#include "komorebi/device/km_worker.hpp"

namespace {

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t KB = get_compile_time_arg_val(3);
    constexpr uint32_t B = get_compile_time_arg_val(4);  // lane window
    constexpr uint32_t P = DT * KB + KB * DV;            // pages of a K / V block

    // Rows `rows` of a job's Q (DT tiles each, `stride` tile rows apart) into the Q
    // buffer, pushed as one item of R x DT pages. False: gave up (recorded).
    template <typename Accessor>
    bool read_q(
        const km::Reader& rd,
        const Accessor& q,
        uint32_t q0,
        uint32_t rows,
        uint32_t stride,
        uint32_t* c
    ) {
        const uint32_t t0 = km::wall();
        if (!rd.reserve(kCbQ, R * DT)) {
            return false;
        }
        const uint32_t t1 = km::wall();
        const uint32_t base = get_write_ptr(kCbQ);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(
                    q0 + i * stride * DT + d, q, base + (i * DT + d) * kTileBytes
                );
            }
        }
        if (!flushed(rd, km::kStReader, km::kFlushReads)) {
            return false;
        }
        cb_push_back(kCbQ, R * DT);
        c[kPrRdQWait] += t1 - t0;
        c[kPrRdQRead] += km::wall() - t1;
        return true;
    }

    // Compute only: `pages` pushed as they are in the lane buffer (no read, no lane).
    bool push_block(const km::Reader& rd, uint32_t pages) {
        if (!rd.reserve(kCbKv, pages)) {
            return false;
        }
        cb_push_back(kCbKv, pages);
        return true;
    }

    // The record's jobs, `passes` times: Q rows, then the K / V blocks through the lane
    // `ln` (null: not this kernel's), or pushed as they are (compute only).
    template <typename Accessor>
    void run_jobs(
        const km::Reader& rd,
        const km_record_t& r,
        uint32_t table,
        const Accessor& tiles_q,
        const Accessor& tiles_k,
        const Accessor& tiles_v,
        km::Lane* ln,
        volatile tt_l1_ptr uint32_t* prof,
        uint32_t t0
    ) {
        const bool reads = r.a5 != kModeComputeOnly;
        const uint32_t jobs = km::l1(table)[kTabJobs];
        const uint32_t passes = r.a6 == 0 ? 1 : r.a6;
        if (jobs > kJobsMax) {
            return;
        }
        uint32_t c[kProfWords] = {};
        uint32_t n = 0;  // blocks of this record so far, over all jobs and passes
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (uint32_t j = r.a7; j < jobs_end(r, jobs); ++j) {
                const volatile tt_l1_ptr uint32_t* w = (
                    km::l1(table + 4 * (kTabFirst + kJobWords * j))
                );
                const uint32_t rows = reads ? w[kJobRows] : 0;
                if (!read_q(rd, tiles_q, w[kJobQ], rows, w[kJobStride], c)) {
                    return;
                }
                if (ln != nullptr &&
                    !lane_job<DT, DV, KB>(*ln, tiles_k, tiles_v, w, n, c)) {
                    return;
                }
                for (
                    uint32_t k_start = 0; !reads && k_start < w[kJobKeys];
                    k_start += KB, ++n
                ) {
                    if (!push_block(rd, P)) {
                        return;
                    }
                }
            }
        }
        const uint32_t t1 = km::wall();
        if (!reads) {
            const uint32_t pad = (B - n % B) % B;
            if (pad != 0 && !push_block(rd, pad * P)) {
                return;
            }
        } else if (ln != nullptr) {
            ln->finish();
        }
        c[kPrRdFinish] += km::wall() - t1;
        c[kPrRdBody] += km::wall() - t0;
        // Without the lane here, its words are the writer's.
        const uint32_t last = ln != nullptr || !reads ? kPrRdFinish : kPrRdQRead;
        for (uint32_t w = kPrRdBody; w <= last; ++w) {
            km::add_sat(prof + w, c[w]);
        }
    }

}  // namespace

void kernel_main() {
    constexpr auto tiles = TensorAccessorArgs<5>();  // Q, K, V: bf16 tiles in DRAM
    km::Reader rd;
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbProf));
    for (uint32_t w = 0; w < kProfWords; ++w) {
        prof[w] = 0;
    }
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegAttend) {
            return;
        }
        const uint32_t t0 = km::wall();
        if (!rd.reserve(kCbJobs, 1)) {
            return;
        }
        const uint32_t table = get_write_ptr(kCbJobs);
        if (!read_table(rd, km::kStReader, r.a4, rd.me() - km::first(r), table)) {
            return;
        }
        cb_push_back(kCbJobs, 1);  // compute pops it at the record's end
        const auto q = TensorAccessor(tiles, r.a0);
        const auto k = TensorAccessor(tiles, r.a1);
        const auto v = TensorAccessor(tiles, r.a2);
#ifndef LANE_ON_WRITER
        if (r.a5 != kModeComputeOnly) {
            km::Lane ln(rd, r, kArgLane);
            if (ln.ok() && ln.member()) {
                run_jobs(rd, r, table, q, k, v, &ln, prof, t0);
            }
            return;
        }
#endif
        run_jobs(rd, r, table, q, k, v, nullptr, prof, t0);
    });
}
