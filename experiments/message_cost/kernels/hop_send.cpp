// Hop table: `reps` puts of each size to each target, one at a time, put to write flush
// in cycles. The NoC0 kernel runs first, then lets the NoC1 kernel start.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "message_cost.h"

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0), page0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2), reps = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4), limit = get_arg_val<uint32_t>(5);
    const uint32_t sync_limit = get_arg_val<uint32_t>(6),
                   tag = get_arg_val<uint32_t>(7);
    const uint32_t buf = get_write_ptr(kCbBlock), times = get_write_ptr(kCbTimes);
    volatile uint32_t* sync = (
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(kCbSlot) + 4 * kSlotSync)
    );
    volatile uint32_t* t = reinterpret_cast<volatile uint32_t*>(times);
    if (!first &&
        !poll_timeout(
            [&] {
                return *sync == tag;
            },
            sync_limit
        )) {
        return;  // nothing written: the host sees zero pages
    }
    const InterleavedAddrGen<true> dst = {
        .bank_base_address = out, .page_size = reps * 4
    };
    const uint32_t sizes[2] = {64, 2048};
    for (uint32_t s = 0; s < 2; s++) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t xy = get_arg_val<uint32_t>(8 + k);
            const uint64_t to = get_noc_addr(xy & 0xFFFFu, xy >> 16, buf);
            bool ok = true;
            for (uint32_t i = 0; i < reps; i++) {
                const uint32_t t0 = wall_lo();
                noc_async_write(buf, to, sizes[s]);
                ok = ok &&
                    poll_timeout(
                         [] {
                             return ncrisc_noc_nonposted_writes_flushed(noc_index);
                         },
                         limit
                    );
                t[i] = ok ? wall_lo() - t0 : kFailed;
            }
            noc_async_write(times, get_noc_addr(page0 + s * n + k, dst), reps * 4);
            if (!poll_timeout(
                    [] {
                        return ncrisc_noc_nonposted_writes_flushed(noc_index);
                    },
                    limit
                )) {
                return;
            }
        }
    }
    *sync = tag;  // this launch's: a word left by an earlier launch never matches
}
