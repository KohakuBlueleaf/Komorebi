// The epilogue program's writer (RISCV_0): each output tile, then the info page, to
// DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "layout.hpp"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto info_args = (
        TensorAccessorArgs<out_args.next_compile_time_args_offset()>()
    );
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const auto info = TensorAccessor(info_args, get_arg_val<uint32_t>(1));
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_wait_front(layout::kCbOut, 1);
        noc_async_write_page(t, out, get_read_ptr(layout::kCbOut));
        noc_async_write_barrier();
        cb_pop_front(layout::kCbOut, 1);
    }
    cb_wait_front(layout::kCbInfo, 1);
    noc_async_write_page(0, info, get_read_ptr(layout::kCbInfo));
    noc_async_write_barrier();
    cb_pop_front(layout::kCbInfo, 1);
}
