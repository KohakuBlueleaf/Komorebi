// Receiver: waits (bounded) for its counter to count `packets` arrivals, checks its
// slots against the pattern, writes its status and its first slots' copy to DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "link.h"

constexpr uint32_t kArgs = 12;  // the host's count comes as compile-time arg 0

inline uint32_t wall_lo() {
    return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

void kernel_main() {
    size_t i = 0;
    const uint32_t status_addr = get_arg_val<uint32_t>(i++);
    const uint32_t status_page = get_arg_val<uint32_t>(i++);
    const uint32_t page_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t status_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t slots_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t counter_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t packets = get_arg_val<uint32_t>(i++);
    const uint32_t payload = get_arg_val<uint32_t>(i++);
    const uint32_t window = get_arg_val<uint32_t>(i++);
    const uint32_t seed = get_arg_val<uint32_t>(i++);
    const uint32_t sender_id = get_arg_val<uint32_t>(i++);
    const uint64_t limit = static_cast<uint64_t>(get_arg_val<uint32_t>(i++)) << 20;

    volatile tt_l1_ptr uint32_t*
        status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_l1);
    volatile tt_l1_ptr uint32_t*
        counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_l1);
    volatile tt_l1_ptr uint32_t*
        slots = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_l1));
    for (uint32_t w = 0; w < kStatusBytes / 4; w++) {
        status[w] = 0;
    }
    status[kWordMagic] = kMagic;
    uint32_t code = kCodeOk, wrong = 0, copy = 0;
    uint32_t last = wall_lo();
    uint64_t total = 0, first = 0;  // cycles since the start; at the first arrival
    bool arrived = false;
    if (
        i != kArgs || get_compile_time_arg_val(0) != kArgs || payload == 0 ||
        payload % 16 != 0 || window == 0 || page_bytes < kStatusBytes + kCopyBytes
    ) {
        code = kCodeArgs;
    } else {
        for (;;) {
            invalidate_l1_cache();
            const uint32_t count = *counter;
            const uint32_t t = wall_lo();
            total += t - last;
            last = t;
            if (count != 0 && !arrived) {
                arrived = true;
                first = total;
            }
            if (count >= packets) {
                break;
            }
            if (total > limit) {
                code = kCodeCountTimeout;
                break;
            }
        }
        const uint32_t filled = packets < window ? packets : window;
        for (uint32_t s = 0; s < filled; s++) {
            for (uint32_t w = 0; w < payload / 4; w++) {
                wrong += slots[s * (payload / 4) + w] != pattern(seed, sender_id, s, w);
            }
        }
        copy = filled * payload < kCopyBytes ? filled * payload : kCopyBytes;
    }
    status[kWordCode] = code;
    status[kWordPackets] = *counter;
    status[kWordWrong] = wrong;
    status[kWordCyclesLo] = static_cast<uint32_t>(total - first);
    status[kWordCyclesHi] = static_cast<uint32_t>((total - first) >> 32);
    status[kWordFirst] = static_cast<uint32_t>(first >> 10);
    status[kWordCount] = *counter;
    const InterleavedAddrGen<true> out = {
        .bank_base_address = status_addr, .page_size = page_bytes
    };
    const uint64_t page = get_noc_addr(status_page, out);
    noc_async_write(status_l1, page, kStatusBytes);
    if (copy) {
        noc_async_write(slots_l1, page + kStatusBytes, copy);
    }
    uint32_t waited = 0, start = wall_lo();
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index) && waited < (1u << 30)) {
        waited = wall_lo() - start;
    }
}
