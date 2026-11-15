// Tile probes writer (RISCV_0): the output CB page to DRAM once compute pushed it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "probe.hpp"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const auto out = TensorAccessor(out_args, out_addr);

    cb_wait_front(probe::kCbOut, 1);
    noc_async_write_page(0, out, get_read_ptr(probe::kCbOut));
    noc_async_write_barrier();
    cb_pop_front(probe::kCbOut, 1);
}
