// Receiver reader (NoC0): fills every page but k from the input, waits (with a timeout)
// for the sender's bump, then pushes all S pages, also when the wait timed out.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "probe.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    const uint32_t in = get_arg_val<uint32_t>(0), flags = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);
    volatile uint32_t* f = reinterpret_cast<volatile uint32_t*>(flags);
    cb_reserve_back(kCbArrive, kPages);  // empty at start: returns at once
    const uint32_t base = get_write_ptr(kCbArrive);
    const auto src = TensorAccessor(in_args, in);
    for (uint32_t p = 0; p < kPages; p++) {
        if (p != k) {
            noc_async_read_page(p, src, base + p * kTileBytes);
        }
    }
    const bool read = probe_poll([] {
        return ncrisc_noc_reads_flushed(noc_index);
    });
    const bool seen = read && probe_poll([&] {
        invalidate_l1_cache();
        return f[kFlag] >= 1;
    });
    f[kRecv] = seen ? 1 : (read ? 2 : 3);
    cb_push_back(kCbArrive, kPages);
}
