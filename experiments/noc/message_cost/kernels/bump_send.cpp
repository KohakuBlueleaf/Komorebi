// Bump table, sender: checks in with the receiver; per repetition waits for its release
// time, then increments the receiver's port (by kLate + 1 if the release had passed).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "message_cost.h"

void kernel_main() {
    const uint32_t xy = get_arg_val<uint32_t>(0), reps = get_arg_val<uint32_t>(1);
    const uint32_t limit = get_arg_val<uint32_t>(2), lead = get_arg_val<uint32_t>(3);
    const uint32_t tag = get_arg_val<uint32_t>(4) << 16;  // this launch's words
    const uint32_t me = get_arg_val<uint32_t>(5), sync_limit = get_arg_val<uint32_t>(6);
    const uint32_t slot = get_write_ptr(kCbSlot);
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(slot);
    const uint64_t port = get_noc_addr(xy & 0xFFFFu, xy >> 16, slot + 4 * kSlotPort);
    w[kSlotOut] = tag;
    noc_async_write(
        slot + 4 * kSlotOut,
        get_noc_addr(xy & 0xFFFFu, xy >> 16, slot + 4 * (kSlotCheckin + 4 * me)),
        4
    );
    if (!poll_timeout(
        [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        },
        limit
    )) {
        return;
    }
    for (uint32_t i = 0; i < reps; i++) {
        if (!poll_timeout(
            [&] {
                return w[kSlotRelease + 1] == (tag | (i + 1));
            },
            i == 0 ? sync_limit : limit + lead
        )) {
            break;
        }
        const uint32_t release = w[kSlotRelease];
        const bool late = static_cast<int32_t>(wall_lo() - release) > 0;
        poll_timeout(
            [&] {
                return static_cast<int32_t>(wall_lo() - release) >= 0;
            },
            lead
        );
        noc_semaphore_inc(port, late ? kLate + 1 : 1);
        if (!poll_timeout(
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            limit
        )) {
            break;
        }
    }
}
