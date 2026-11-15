// Formats probe reader (RISCV_1): the left tensor's pages into buffer A, the right
// tensor's into buffer B, each page one tile of the buffer's format.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "layout.hpp"

namespace {

    template <typename Accessor>
    void read(const Accessor& tensor, uint32_t page, uint32_t cb) {
        cb_reserve_back(cb, 1);
        noc_async_read_page(page, tensor, get_write_ptr(cb));
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    }

}  // namespace

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    const uint32_t a_pages = get_arg_val<uint32_t>(2);
    const uint32_t b_pages = get_arg_val<uint32_t>(3);
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    for (uint32_t i = 0; i < a_pages; ++i) {
        read(a, i, formats::kCbA);
    }
    for (uint32_t i = 0; i < b_pages; ++i) {
        read(b, i, formats::kCbB);
    }
}
