// A page call's range against its tensor's pages (KM_GUARD): past the last page it is
// refused by name, bad_item in kWherePageRange; KM_GUARD 0 checks nothing.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_placement.hpp"

namespace km {

#if KM_GUARD
    // The bytes of DRAM tensor `word` (0: unknown, unchecked): its table entry's word 3
    // (copy | pages << 2), else the session's list {count; address, 64-byte lines}.
    inline uint64_t extent_bytes(uint32_t word) {
        const volatile uint32_t* area = island_area();
#if defined(KM_TENSOR_TABLE)
        const volatile uint32_t* entry = tensor_entry(Islands(area), word);
        if (entry == nullptr || (entry[3] & 3u) == KM_TENSOR_BANK) {
            return 0;
        }
        return static_cast<uint64_t>(entry[3] >> 2) * entry[0];
#else
        const uint32_t list = area[KM_ISLAND_TENSORS / 4];
        if (list == 0) {
            return 0;
        }
        const volatile tt_l1_ptr uint32_t* words = l1(list);
        uint32_t low = 0, high = words[0];
        while (low < high) {  // sorted by address
            const uint32_t middle = (low + high) / 2;
            const uint32_t at = words[1 + 2 * middle];
            if (at == word) {
                return static_cast<uint64_t>(words[2 + 2 * middle]) << 6;
            }
            if (at < word) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return 0;
#endif
    }

    // Whether pages first .. first + pages - 1 of `bytes` lie inside tensor `word`;
    // else recorded (the call's last page named) and false.
    template <typename Kernel>
    __attribute__((noinline)) bool page_range_ok(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        uint32_t word,
        uint32_t first,
        uint32_t pages,
        uint32_t bytes
    ) {
        const uint64_t extent = extent_bytes(word);
        const uint64_t end = (static_cast<uint64_t>(first) + pages) * bytes;
        if (extent == 0 || end <= extent) {
            return true;
        }
        const uint32_t page = first + pages - 1;
        core.control(KM_CONTROL_LIVE)[kLvLine] = 1u << 31 | page;
        core.gave_up(
            core.role() == kRoleWriter ? kStWriter : kStReader,
            kGaveBadItem,
            kWherePageRange,
            slot | (page & 0xFFu) << 8,
            sequence
        );
        return false;
    }
#else
    template <typename Kernel>
    inline bool page_range_ok(
        const Kernel&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t
    ) {
        return true;
    }
#endif

    // Whether `Form` is a range of one bank (km::BankRange: its entry bounds it).
    template <typename Form, typename = decltype(&Form::bank)>
    constexpr bool extent_banked(int) {
        return true;
    }

    template <typename Form>
    constexpr bool extent_banked(...) {
        return false;
    }

    // page_range_ok of a Tensor or a SizedTensor (a BankRange: true, its entry's).
    template <typename Kernel, typename Form>
    inline bool pages_inside(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Form& form,
        uint32_t first,
        uint32_t pages
    ) {
        if constexpr (extent_banked<Form>(0)) {
            return true;
        } else {
            return (
                page_range_ok(
                    core, slot, sequence, form.word, first, pages, form.page_bytes()
                )
            );
        }
    }

    // page_range_ok of tensor word `tensor` read through accessor arguments `args`.
    template <typename Kernel, typename Args>
    inline bool pages_inside(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        const Args& args
    ) {
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
        return page_range_ok(core, slot, sequence, tensor, first, pages, bytes);
    }

}  // namespace km
