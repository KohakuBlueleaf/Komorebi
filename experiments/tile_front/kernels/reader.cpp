// SiLU body experiment reader (RISCV_1): the input tiles from DRAM into CB 0, one page
// each.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "front.hpp"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    const auto in = TensorAccessor(in_args, in_addr);
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_reserve_back(front::kCbIn, 1);
        noc_async_read_page(t, in, get_write_ptr(front::kCbIn));
        noc_async_read_barrier();
        cb_push_back(front::kCbIn, 1);
    }
}
