// Dst words probe writer (RISCV_0): two output pages an input tile (Dst tile 0's words,
// tile 1's after the vector unit's reordering), then the info page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "layout.hpp"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto info_args = (
        TensorAccessorArgs<out_args.next_compile_time_args_offset()>()
    );
    const uint32_t pages = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const auto info = TensorAccessor(info_args, get_arg_val<uint32_t>(1));
    for (uint32_t p = 0; p < pages; ++p) {
        cb_wait_front(dst_words::kCbOut, 1);
        noc_async_write_page(p, out, get_read_ptr(dst_words::kCbOut));
        noc_async_write_barrier();
        cb_pop_front(dst_words::kCbOut, 1);
    }
    cb_wait_front(dst_words::kCbInfo, 1);
    noc_async_write_page(0, info, get_read_ptr(dst_words::kCbInfo));
    noc_async_write_barrier();
    cb_pop_front(dst_words::kCbInfo, 1);
}
