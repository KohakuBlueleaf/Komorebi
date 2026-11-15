// The epilogue program's reader (RISCV_1): per tile its two operands, one tile of A
// and one of B, from DRAM.
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
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const uint32_t tiles = get_arg_val<uint32_t>(2);
    for (uint32_t t = 0; t < tiles; ++t) {
        read(a, t, layout::kCbA);
        read(b, t, layout::kCbB);
    }
}
