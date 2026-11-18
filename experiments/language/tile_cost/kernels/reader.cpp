// Tile-cost reader (RISCV_1): the resident tiles of each input buffer, once.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "cost.hpp"

namespace {

    template <typename Accessor>
    void fill(const Accessor& tensor, uint32_t cb, uint32_t pages) {
        for (uint32_t i = 0; i < pages; ++i) {
            cb_reserve_back(cb, 1);
            noc_async_read_page(i, tensor, get_write_ptr(cb));
            noc_async_read_barrier();
            cb_push_back(cb, 1);
        }
    }

}  // namespace

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    constexpr auto alt_args = (
        TensorAccessorArgs<b_args.next_compile_time_args_offset()>()
    );
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const auto alt = TensorAccessor(alt_args, get_arg_val<uint32_t>(2));
    fill(a, cost::kCbA, get_arg_val<uint32_t>(3));
    fill(b, cost::kCbB, get_arg_val<uint32_t>(4));
    fill(alt, cost::kCbAlt, get_arg_val<uint32_t>(5));
}
