// Where a tensor's pages live: its reference word (`tensor.address`) is a bank address
// on one island, its entry of the tensor table over islands (KM_ISLAND_TENSORS).
#pragma once

#include <stdint.h>

#include "km_island.hpp"

namespace km {

    // A page's place: page `index` of the interleaved buffer at bank address `address`
    // on island `island`.
    struct PageAt {
        uint32_t island;
        uint32_t address;
        uint32_t index;
    };

    // Where page `page` of tensor `tensor` (its reference word) lives; a word that is no
    // entry is a bank address here: {this island, word, page}. A copy's: this island's.
    inline PageAt page_at(uint32_t tensor, uint32_t page) {
        const Islands view(island_area());
        const uint32_t table = view.tensors();
        if (tensor >= KM_TENSORS_MAX || table == 0) {
            return {view.here(), tensor, page};
        }
        const uint32_t islands = view.count();
        const volatile uint32_t* entry = (
            reinterpret_cast<const volatile uint32_t*>(
                table +
                tensor * (KM_TENSOR_HEADER_BYTES + islands * KM_TENSOR_PART_BYTES)
            )
        );
        const volatile uint32_t* parts = entry + KM_TENSOR_HEADER_BYTES / 4;
        constexpr uint32_t kPartWords = KM_TENSOR_PART_BYTES / 4;
        if (entry[3] & 1u) {
            return {view.here(), parts[view.here() * kPartWords], page};
        }
        const uint32_t steps = entry[1];
        const uint32_t inner = entry[2];
        const uint32_t block = page / (steps * inner);
        const uint32_t step = page / inner % steps;
        for (uint32_t island = 0; island < islands; island++) {
            const volatile uint32_t* part = parts + island * kPartWords;
            if (step - part[1] < part[2]) {
                return {
                    island,
                    part[0],
                    (block * part[2] + step - part[1]) * inner + page % inner
                };
            }
        }
        return {view.here(), 0, page};  // outside the tensor: no island holds it
    }

    // The island that holds page `page` of tensor `tensor`; one island: 0.
    inline uint32_t island_of(uint32_t tensor, uint32_t page) {
        return page_at(tensor, page).island;
    }

}  // namespace km
