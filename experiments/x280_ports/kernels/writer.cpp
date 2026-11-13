// X280 ports experiment, grid side: the writer gives TELL's word as the result, and
// with XP_F_READS reads DRAM pages until the reader's asks are done (xp.h).
#include <stdint.h>

#include "../xp.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kCbScratch = 16;
constexpr uint32_t kReadsMax = 1u << 24;  // limit of the reading loop, pages

void kernel_main() {
    constexpr auto reads_args = TensorAccessorArgs<0>();
    km::Writer wr;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    volatile tt_l1_ptr uint32_t* flags = km::l1(scratch + XP_SCRATCH_FLAGS);
    wr.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        if (handler == XP_HANDLER_TELL) {
            const uint32_t code = wr.poll(
                [&] {
                    return flags[0] == r.seq;
                },
                wr.args().wait_limit
            );
            if (code != km::kGaveNone) {
                wr.gave_up(km::kStWriter, code, km::kWhereData, km::slot(r), r.seq);
                return false;
            }
            wr.result(flags[1]);
            return true;
        }
        if (handler != XP_HANDLER_ASK || !(r.a2 & XP_F_READS)) {
            return true;
        }
        // DRAM pages from tensor a3 (a4 pages) while the reader asks; a limited loop
        // that leaves on the stop flag.
        const auto src = TensorAccessor(reads_args, r.a3);
        uint32_t n = 0;
        for (; n < kReadsMax; n++) {
            invalidate_l1_cache();
            if (flags[2] == r.seq || wr.stop()) {
                break;
            }
            noc_async_read_page(
                (wr.me() * 7 + n) % r.a4, src, scratch + XP_SCRATCH_READ
            );
            const uint32_t code = wr.poll(
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                wr.args().wait_limit
            );
            if (code != km::kGaveNone) {
                wr.gave_up(km::kStWriter, code, km::kWhereData, km::slot(r), r.seq);
                return false;
            }
        }
        flags[3] = n;
        flags[4] = r.seq;
        return true;
    });
}
