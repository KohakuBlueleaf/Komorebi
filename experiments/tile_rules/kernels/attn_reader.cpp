// Per-core attention reader (RISCV_1): the query row, the ones tile and the first
// row state once, then per step its block of K and V tiles (the hand body's order).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "rules.hpp"

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
    constexpr auto kv_args = TensorAccessorArgs<0>();
    constexpr auto q_args = (
        TensorAccessorArgs<kv_args.next_compile_time_args_offset()>()
    );
    constexpr auto st_args = (
        TensorAccessorArgs<q_args.next_compile_time_args_offset()>()
    );
    constexpr auto ones_args = (
        TensorAccessorArgs<st_args.next_compile_time_args_offset()>()
    );
    const auto kv = TensorAccessor(kv_args, get_arg_val<uint32_t>(0));
    const auto q = TensorAccessor(q_args, get_arg_val<uint32_t>(1));
    const auto st = TensorAccessor(st_args, get_arg_val<uint32_t>(2));
    const auto ones = TensorAccessor(ones_args, get_arg_val<uint32_t>(3));
    const uint32_t steps = get_arg_val<uint32_t>(4);
    const uint32_t block_pages = get_arg_val<uint32_t>(5);
    const uint32_t head_tiles = get_arg_val<uint32_t>(6);
    for (uint32_t h = 0; h < head_tiles; ++h) {
        read(q, h, rules::kCbQ);
    }
    read(ones, 0, rules::kCbOnes);
    read(st, 0, rules::kCbSt);
    for (uint32_t s = 0; s < steps; ++s) {
        for (uint32_t page = 0; page < block_pages; ++page) {
            read(kv, s * block_pages + page, rules::kCbKv);
        }
    }
}
