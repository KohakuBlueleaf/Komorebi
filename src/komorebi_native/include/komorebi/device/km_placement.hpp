// Where a tensor's pages live, on a core: the session writes a tensor's placement as
// compile arguments after its accessor's (IslandTensor.placement_args()).
#pragma once

#include <stdint.h>

#include "api/compile_time_args.h"
#include "km_island.hpp"

namespace km {

    // A tensor's placement from compile argument `Base` on: the islands, copy, then
    // (first page, pages) of each island. On one island: island 0 holds every page.
    template <uint32_t Base>
    struct Placement {
        static constexpr uint32_t kIslands = get_compile_time_arg_val(Base);
        static constexpr bool kCopy = get_compile_time_arg_val(Base + 1) != 0;

        static constexpr uint32_t next_compile_time_args_offset() {
            return Base + 2 + 2 * kIslands;
        }
    };

    // From island I on, the first whose part holds `page`; none: this island.
    template <uint32_t Base, uint32_t I>
    inline uint32_t placement_holder(uint32_t page) {
        if constexpr (I >= Placement<Base>::kIslands) {
            return island();
        } else {
            constexpr uint32_t first = get_compile_time_arg_val(Base + 2 + 2 * I);
            constexpr uint32_t pages = get_compile_time_arg_val(Base + 3 + 2 * I);
            return page - first < pages ? I : placement_holder<Base, I + 1>(page);
        }
    }

    // The island that holds page `page` of the tensor; a copy's (or a page outside the
    // tensor): this core's own island.
    template <uint32_t Base>
    inline uint32_t island_of(const Placement<Base>&, uint32_t page) {
        if constexpr (Placement<Base>::kCopy) {
            return island();
        } else {
            return placement_holder<Base, 0>(page);
        }
    }

}  // namespace km
