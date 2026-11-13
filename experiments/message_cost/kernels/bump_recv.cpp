// Bump table, receiver: after every sender checked in, per repetition a release time to
// all, then cycles until the port counts all (page 0) and late senders (page 1).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "message_cost.h"

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const uint32_t reps = get_arg_val<uint32_t>(2), lead = get_arg_val<uint32_t>(3);
    const uint32_t limit = get_arg_val<uint32_t>(4);
    const uint32_t tag = get_arg_val<uint32_t>(5) << 16;
    const uint32_t sync_limit = get_arg_val<uint32_t>(6);
    const uint32_t slot = get_write_ptr(kCbSlot), times = get_write_ptr(kCbTimes);
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(slot);
    volatile uint32_t* t = reinterpret_cast<volatile uint32_t*>(times);
    volatile uint32_t* lates = t + reps;
    auto flushed = [] {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    };
    bool ok = poll_timeout(
        [&] {
            for (uint32_t k = 0; k < n; k++) {
                if (w[kSlotCheckin + 4 * k] != tag) {
                    return false;
                }
            }
            return true;
        },
        sync_limit
    );
    for (uint32_t i = 0; i < reps; i++) {
        w[kSlotPort] = 0;
        const uint32_t release = wall_lo() + lead;
        w[kSlotOut] = release;
        w[kSlotOut + 1] = tag | (i + 1);
        for (uint32_t k = 0; ok && k < n; k++) {
            const uint32_t xy = get_arg_val<uint32_t>(7 + k);
            noc_async_write(
                slot + 4 * kSlotOut,
                get_noc_addr(xy & 0xFFFFu, xy >> 16, slot + 4 * kSlotRelease),
                8
            );
        }
        ok = ok && poll_timeout(flushed, limit);
        ok = ok &&
            poll_timeout(
                [&] {
                    return (w[kSlotPort] & 0xFFFFu) >= n;
                },
                limit + lead
            );
        t[i] = ok ? wall_lo() - release : kFailed;
        lates[i] = w[kSlotPort] >> 16;
    }
    const InterleavedAddrGen<true> dst = {
        .bank_base_address = out, .page_size = reps * 4
    };
    noc_async_write(times, get_noc_addr(0, dst), reps * 4);
    noc_async_write(times + reps * 4, get_noc_addr(1, dst), reps * 4);
    poll_timeout(flushed, limit);
}
