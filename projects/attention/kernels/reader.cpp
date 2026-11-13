// Attention reader, on the lane's NoC: the member's job table to compute, then per job
// its Q rows and the job's K / V blocks (the lane's sender reads and casts them).
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
    constexpr uint32_t V0 = DT * KB;  // first V page of a K / V block

    // Rows `rows` of a job's Q (DT tiles each, `stride` tile rows apart) into the Q
    // buffer, pushed as one item of R x DT pages. False: gave up (recorded).
    template <typename Accessor>
    bool read_q(
        const km::Reader& rd,
        const Accessor& q,
        uint32_t q0,
        uint32_t rows,
        uint32_t stride
    ) {
        if (!rd.reserve(kCbQ, R * DT)) {
            return false;
        }
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
        return true;
    }

    // Sender: K tiles [k_start, k_start + kb) of every head-dim tile (d-major) and
    // their V rows (key-major) into the block at `addr`.
    template <typename Accessor>
    void read_kv(
        const Accessor& k,
        const Accessor& v,
        uint32_t k0,
        uint32_t v0,
        uint32_t k_start,
        uint32_t kb,
        uint32_t addr
    ) {
        for (uint32_t j = 0; j < kb; ++j) {
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(
                    k0 + (k_start + j) * DT + d, k, addr + (d * KB + j) * kTileBytes
                );
            }
        }
        for (uint32_t j = 0; j < kb; ++j) {
            for (uint32_t c = 0; c < DV; ++c) {
                noc_async_read_page(
                    v0 + (k_start + j) * DV + c,
                    v,
                    addr + (V0 + j * DV + c) * kTileBytes
                );
            }
        }
    }

}  // namespace

void kernel_main() {
    constexpr auto tiles = TensorAccessorArgs<4>();  // Q, K, V: bf16 tiles in DRAM
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegAttend) {
            return;
        }
        if (!rd.reserve(kCbJobs, 1)) {
            return;
        }
        const uint32_t table = get_write_ptr(kCbJobs);
        if (!read_table(rd, km::kStReader, r.a4, rd.me() - km::first(r), table)) {
            return;
        }
        cb_push_back(kCbJobs, 1);  // compute pops it at the record's end
        km::Lane ln(rd, r, kArgLane);
        if (!ln.ok() || !ln.member()) {
            return;
        }
        const auto q = TensorAccessor(tiles, r.a0);
        const auto k = TensorAccessor(tiles, r.a1);
        const auto v = TensorAccessor(tiles, r.a2);
        const uint32_t jobs = km::l1(table)[kTabJobs];
        if (jobs > kJobsMax) {
            return;
        }
        uint32_t n = 0;  // blocks of this record so far, over all jobs
        for (uint32_t j = 0; j < jobs; ++j) {
            const volatile tt_l1_ptr uint32_t* w = (
                km::l1(table + 4 * (kTabFirst + kJobWords * j))
            );
            const uint32_t k0 = w[kJobK], v0 = w[kJobV], nk = w[kJobKeys];
            if (!read_q(rd, q, w[kJobQ], w[kJobRows], w[kJobStride])) {
                return;
            }
            for (uint32_t k_start = 0; k_start < nk; k_start += KB, ++n) {
                if (!ln.sender()) {
                    if (!ln.arrive(n)) {
                        return;
                    }
                    continue;
                }
                const uint32_t addr = ln.reserve(n);
                if (addr == 0) {
                    return;
                }
                const uint32_t kb = nk - k_start < KB ? nk - k_start : KB;
                read_kv(k, v, k0, v0, k_start, kb, addr);
                if (!ln.cast(n)) {
                    return;
                }
            }
        }
        // A lane with no block: its receivers never stamp, so the sender's finish()
        // would wait for stamps that never come; nothing is left to clear.
        if (n != 0) {
            ln.finish();
        }
    });
}
