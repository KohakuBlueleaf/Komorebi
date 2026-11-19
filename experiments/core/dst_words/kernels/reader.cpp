// Dst words probe reader (RISCV_1): each input page (one tile of 1,024 words) into the
// input buffer.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "layout.hpp"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_reserve_back(dst_words::kCbIn, 1);
        noc_async_read_page(t, in, get_write_ptr(dst_words::kCbIn));
        noc_async_read_barrier();
        cb_push_back(dst_words::kCbIn, 1);
    }
}
