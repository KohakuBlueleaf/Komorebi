// SiLU body experiment writer (RISCV_0): each output tile, then the info page, to DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "front.hpp"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto info_args = (
        TensorAccessorArgs<out_args.next_compile_time_args_offset()>()
    );
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t info_addr = get_arg_val<uint32_t>(1);
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto info = TensorAccessor(info_args, info_addr);
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_wait_front(front::kCbOut, 1);
        noc_async_write_page(t, out, get_read_ptr(front::kCbOut));
        noc_async_write_barrier();
        cb_pop_front(front::kCbOut, 1);
    }
    cb_wait_front(front::kCbInfo, 1);
    noc_async_write_page(0, info, get_read_ptr(front::kCbInfo));
    noc_async_write_barrier();
    cb_pop_front(front::kCbInfo, 1);
}
