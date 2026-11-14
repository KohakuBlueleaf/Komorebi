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

    constexpr uint32_t kPartWords = KM_TENSOR_PART_BYTES / 4;

    // The table entry word `tensor` names on this island, or nullptr: a bank address.
    inline const volatile uint32_t* tensor_entry(const Islands& view, uint32_t tensor) {
        const uint32_t table = view.tensors();
        if (tensor >= KM_TENSORS_MAX || table == 0) {
            return nullptr;
        }
        const uint32_t bytes = (
            KM_TENSOR_HEADER_BYTES + view.count() * KM_TENSOR_PART_BYTES
        );
        return reinterpret_cast<const volatile uint32_t*>(table + tensor * bytes);
    }

    // Where page `page` of tensor `tensor` (its reference word) lives; a word naming
    // no entry is a bank address here: {this island, word, page}; a copy: this island.
    inline PageAt page_at(uint32_t tensor, uint32_t page) {
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, tensor);
        if (entry == nullptr) {
            return {view.here(), tensor, page};
        }
        const volatile uint32_t* parts = entry + KM_TENSOR_HEADER_BYTES / 4;
        if (entry[3] & 1u) {
            return {view.here(), parts[view.here() * kPartWords], page};
        }
        const uint32_t steps = entry[1];
        const uint32_t inner = entry[2];
        const uint32_t block = page / (steps * inner);
        const uint32_t step = page / inner % steps;
        for (uint32_t island = 0; island < view.count(); island++) {
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

    // How many pages from `page` on (at most `most`) the island of `at`, page `page`'s
    // place, holds at consecutive indices of its part: from the entry, to the end of
    // its part's steps in this block (all of them: to the end).
    inline uint32_t page_run(
        uint32_t tensor, uint32_t page, uint32_t most, const PageAt& at
    ) {
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, tensor);
        if (entry == nullptr || (entry[3] & 1u) || at.address == 0) {
            return most;  // one buffer: every page at its own index
        }
        const uint32_t steps = entry[1];
        const uint32_t inner = entry[2];
        const volatile uint32_t* part = (
            entry + KM_TENSOR_HEADER_BYTES / 4 + at.island * kPartWords
        );
        if (part[1] == 0 && part[2] == steps) {
            return most;
        }
        const uint32_t step = page / inner % steps;
        const uint32_t left = (part[1] + part[2] - step) * inner - page % inner;
        return left < most ? left : most;
    }

    // Whether word `tensor` names a split tensor of which another island holds pages;
    // a bank address or a copy: false.
    inline bool held_elsewhere(uint32_t tensor) {
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, tensor);
        if (entry == nullptr || (entry[3] & 1u)) {
            return false;
        }
        const volatile uint32_t* parts = entry + KM_TENSOR_HEADER_BYTES / 4;
        for (uint32_t island = 0; island < view.count(); island++) {
            if (island != view.here() && parts[island * kPartWords + 2] != 0) {
                return true;
            }
        }
        return false;
    }

    // The island that holds page `page` of tensor `tensor`; one island: 0.
    inline uint32_t island_of(uint32_t tensor, uint32_t page) {
        return page_at(tensor, page).island;
    }

    // A page of km::SizedTensor or km::BankRange: on island `island`, index `index` of
    // the DRAM pages at bank address `address`, `pages` from it there (0: refused).
    struct DramRun {
        uint32_t island;
        uint32_t address;
        uint32_t index;
        uint32_t pages;
    };

    // Page `page` of `bytes` (64s) of the interleaved tensor of word `tensor`, and of
    // `most` how many follow it there: a copy here, else the one island holding it all.
    inline DramRun sized_page_at(
        uint32_t tensor, uint32_t bytes, uint32_t page, uint32_t most
    ) {
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, tensor);
        if (entry == nullptr) {
            // a bank address: every page here
            return {view.here(), tensor, page, most};
        }
        const volatile uint32_t* parts = entry + KM_TENSOR_HEADER_BYTES / 4;
        uint32_t holder = entry[3] == 1u ? view.here() : view.count();
        for (uint32_t island = 0; entry[3] == 0 && island < view.count(); island++) {
            const volatile uint32_t* part = parts + island * kPartWords;
            if (part[1] == 0 && part[2] == entry[1]) {
                holder = island;  // its part has every step: the whole tensor
            }
        }
        if (holder >= view.count() || bytes == 0 || bytes % 64 != 0) {
            return {view.here(), 0, page, 0};  // split over islands, or not whole lines
        }
        const volatile uint32_t* part = parts + holder * kPartWords;
        // the pages of `bytes` its part's bytes in each bank hold, over all the banks
        const uint32_t end = part[KM_TENSOR_PART_BANK_BYTES] / bytes * NUM_DRAM_BANKS;
        const uint32_t left = page < end ? end - page : 0;
        return {holder, part[0], page, left < most ? left : most};
    }

    // Page `page` of `bytes` (64s) of the range of one bank word `range` names: index
    // its bank at its bank address; a bank address: in bank `bank` here.
    inline DramRun bank_page_at(
        uint32_t range, uint32_t bank, uint32_t bytes, uint32_t page
    ) {
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, range);
        if (entry == nullptr) {
            return {view.here(), range + page * bytes, bank, 1};
        }
        const uint32_t holder = entry[0];
        if (
            entry[3] != KM_TENSOR_BANK || holder >= view.count() || bytes == 0 ||
            bytes % 64 != 0 || page >= entry[2] / bytes
        ) {
            return {view.here(), 0, bank, 0};
        }
        const volatile uint32_t*
            part = (entry + KM_TENSOR_HEADER_BYTES / 4 + holder * kPartWords);
        return {holder, part[0] + page * bytes, entry[1], 1};
    }

}  // namespace km
