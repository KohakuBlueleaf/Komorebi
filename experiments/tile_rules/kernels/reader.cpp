// Tile-rules reader (RISCV_1): the P pages once, then per unit its tiles of A (and,
// for the hand norm, of Y again), with two inputs one tile of B, with three one of P.
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
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    constexpr auto p_args = (
        TensorAccessorArgs<b_args.next_compile_time_args_offset()>()
    );
    const uint32_t units = get_arg_val<uint32_t>(3);
    const uint32_t unit_tiles = get_arg_val<uint32_t>(4);
    const uint32_t inputs = get_arg_val<uint32_t>(5);
    const uint32_t p_pages = get_arg_val<uint32_t>(6);
    const uint32_t copy_y = get_arg_val<uint32_t>(7);
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const auto p = TensorAccessor(p_args, get_arg_val<uint32_t>(2));
    for (uint32_t i = 0; i < p_pages; ++i) {
        read(p, i, rules::kCbP);
    }
    for (uint32_t u = 0; u < units; ++u) {
        for (uint32_t t = 0; t < unit_tiles; ++t) {
            read(a, u * unit_tiles + t, rules::kCbA);
            if (copy_y != 0) {
                read(a, u * unit_tiles + t, rules::kCbY);
            }
        }
        if (inputs >= 2) {
            read(b, u, rules::kCbB);
        }
        if (inputs == 3) {  // a product's third operand (P carries no pages then)
            read(p, u, rules::kCbP);
        }
    }
}
