// Tile probes reader (RISCV_1): the one input page from DRAM into the input CB.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "probe.hpp"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const auto in = TensorAccessor(in_args, in_addr);

    cb_reserve_back(probe::kCbIn, 1);
    noc_async_read_page(0, in, get_write_ptr(probe::kCbIn));
    noc_async_read_barrier();
    cb_push_back(probe::kCbIn, 1);
}
