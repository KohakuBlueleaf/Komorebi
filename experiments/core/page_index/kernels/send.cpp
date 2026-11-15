// Sender (writer kernel, NoC1): reads input tile k into its own page 0, puts it into
// the receiver's page k, waits for the write, bumps the receiver's flag, waits for it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "probe.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    const uint32_t in = get_arg_val<uint32_t>(0), flags = get_arg_val<uint32_t>(1);
    const uint32_t xy = get_arg_val<uint32_t>(2), k = get_arg_val<uint32_t>(3);
    volatile uint32_t* f = reinterpret_cast<volatile uint32_t*>(flags);
    const uint32_t base = get_write_ptr(kCbArrive);
    const auto src = TensorAccessor(in_args, in);
    noc_async_read_page(k, src, base);
    if (!probe_poll([] {
        return ncrisc_noc_reads_flushed(noc_index);
    })) {
        f[kSend] = 2;
        return;
    }
    const uint32_t x = xy & 0xFFFFu, y = xy >> 16;
    noc_async_write(base, get_noc_addr(x, y, base + k * kTileBytes), kTileBytes);
    if (!probe_poll([] {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    })) {
        f[kSend] = 3;
        return;
    }
    noc_semaphore_inc(get_noc_addr(x, y, flags + 4 * kFlag), 1);
    if (!probe_poll([] {
        return ncrisc_noc_nonposted_atomics_flushed(noc_index);
    })) {
        f[kSend] = 4;
        return;
    }
    f[kSend] = 1;
}
