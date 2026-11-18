// Tile-cost writer (RISCV_0): the info page, once compute pushed it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "cost.hpp"

void kernel_main() {
    constexpr auto info_args = TensorAccessorArgs<0>();
    const auto info = TensorAccessor(info_args, get_arg_val<uint32_t>(0));
    cb_wait_front(cost::kCbInfo, 1);
    noc_async_write_page(0, info, get_read_ptr(cost::kCbInfo));
    noc_async_write_barrier();
    cb_pop_front(cost::kCbInfo, 1);
}
