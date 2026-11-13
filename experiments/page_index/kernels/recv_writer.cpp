// Receiver writer (NoC1): waits (with a timeout) for the two pages compute packed and
// writes them to output pages 0 and 1.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "probe.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t out = get_arg_val<uint32_t>(0), flags = get_arg_val<uint32_t>(1);
    volatile uint32_t* f = reinterpret_cast<volatile uint32_t*>(flags);
    if (!probe_poll([] {
        return cb_pages_available_at_front(kCbOut, 2);
    })) {
        f[kWrite] = 2;
        return;
    }
    cb_wait_front(kCbOut, 2);
    const auto dst = TensorAccessor(out_args, out);
    const uint32_t at = get_read_ptr(kCbOut);
    noc_async_write_page(0, dst, at);
    noc_async_write_page(1, dst, at + kTileBytes);
    const bool ok = probe_poll([] {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    });
    cb_pop_front(kCbOut, 2);
    f[kWrite] = ok ? 1 : 3;
}
