// The dispatcher's second mover: multicasts the slot line the loop hands it in the word
// pair to rectangle 1, on this mover's NoC, while the loop writes the others.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_core.hpp"
#include "live.h"

void kernel_main() {
    const uint32_t control = get_arg_val<uint32_t>(kLvControl);
    const uint32_t block = get_arg_val<uint32_t>(kLvBlock);
    const uint32_t a = get_arg_val<uint32_t>(kLvRect + 3);
    const uint32_t b = get_arg_val<uint32_t>(kLvRect + 4);
    const uint32_t dests = get_arg_val<uint32_t>(kLvRect + 5);
    const uint32_t start = noc_index == 0 ? a : b, end = noc_index == 0 ? b : a;
    const uint64_t target = (
        get_noc_multicast_addr(
            start & 0xFFFFu, start >> 16, end & 0xFFFFu, end >> 16, 0
        )
    );
    volatile tt_l1_ptr uint32_t* hand = km::l1(block + kLbHand);
    uint32_t seen = 0;
    for (;;) {
        invalidate_l1_cache();
        const uint32_t token = hand[kHandLine];
        if (token == seen) {
            live_pace();
            continue;
        }
        if (token == kHandStop) {
            break;
        }
        const uint32_t line = control + KM_CONTROL_RING + (token & 0xFFu) * 64;
#if defined(LIVE_LOOPBACK)
        noc_async_write_multicast_loopback_src(line, target | line, 64, dests);
#else
        noc_async_write_multicast(line, target | line, 64, dests);
#endif
        noc_async_writes_flushed();
        hand[kHandDone] = token;
        seen = token;
    }
    noc_async_write_barrier();
}
