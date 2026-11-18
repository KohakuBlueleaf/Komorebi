// A page call's range against its tensor's pages, and a page write's source (KM_GUARD):
// refused by name (kWherePageRange, kWherePageSource); KM_GUARD 0 checks nothing.
#pragma once

#include <stdint.h>

#include "../core/km_core.hpp"
#include "km_placement.hpp"

namespace km {

    // A caller's source line, as a page write's give-up names it (a type of its own,
    // so the page calls' overloads by word stay apart).
    struct SourceLine {
        uint32_t value;
    };

// A page write's last parameter, the caller's line, and its pass on: none under
// KM_GUARD 0, so the calls build as before.
#if KM_GUARD
#define KM_SOURCE_LINE , ::km::SourceLine line = {__builtin_LINE()}
#define KM_LINE , line
#else
#define KM_SOURCE_LINE
#define KM_LINE
#endif

#if KM_GUARD
#if defined(KM_TENSOR_TABLE)
    // The bytes of the entry whose part here is at bank address `word` and holds the
    // whole tensor (a copy, or a tensor this island holds); 0: no such entry.
    inline uint64_t extent_at_address(const Islands& view, uint32_t word) {
        if (word < KM_TENSORS_MAX) {
            return 0;
        }
        for (uint32_t tensor = 0; tensor < KM_TENSORS_MAX; tensor++) {
            const volatile uint32_t* entry = tensor_entry(view, tensor);
            if (entry == nullptr) {
                return 0;
            }
            const volatile uint32_t*
                part = (entry + KM_TENSOR_HEADER_BYTES / 4 + view.here() * kPartWords);
            const uint32_t placement = entry[3] & 3u;
            const bool whole = (
                placement == 1u ||
                (placement == 0u && part[1] == 0 && part[2] == entry[1])
            );
            if (whole && part[0] == word) {
                return static_cast<uint64_t>(entry[3] >> 2) * entry[0];
            }
        }
        return 0;
    }
#endif

    // The bytes of DRAM tensor `word` (0: unknown, unchecked): its entry's (word 3,
    // copy | pages << 2) or its part's (extent_at_address), else the session's list's.
    inline uint64_t extent_bytes(uint32_t word) {
        const volatile uint32_t* area = island_area();
#if defined(KM_TENSOR_TABLE)
        const Islands view(area);
        const volatile uint32_t* entry = tensor_entry(view, word);
        if (entry == nullptr) {
            return extent_at_address(view, word);
        }
        if ((entry[3] & 3u) == KM_TENSOR_BANK) {
            return 0;  // a range of one bank: bank_page_at bounds it by its entry
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

#if defined(KM_OVERLAY_PART)
    // An overlay part keeps no lines (the persistent kernel's state is its own).
    inline uint64_t extent_cached(uint32_t word) {
        return extent_bytes(word);
    }
#else
    // A CPU's last bounds (extent_bytes is a lookup): word + 1 (0: empty), the bounds'
    // generation it was read in (the island header's KM_ISLAND_BOUNDS), its bytes.
    struct ExtentLine {
        uint32_t tag;
        uint32_t generation;
        uint64_t bytes;
    };

    constexpr uint32_t kExtentLines = 4;
    inline ExtentLine extent_lines[kExtentLines];  // each CPU its own (zero at start)
    inline uint32_t extent_next;                   // the line the next miss fills

    // The bounds' generation: 0 at a load, one more each time a bound changes while the
    // run lives (a tensor table row written: write_table_row).
    inline uint32_t bounds_generation() {
        return island_area()[KM_ISLAND_BOUNDS];
    }

    // A miss: extent_bytes, kept in the next line for this generation.
    __attribute__((noinline)) inline uint64_t extent_missed(
        uint32_t word, uint32_t generation
    ) {
        const uint64_t bytes = extent_bytes(word);
        extent_lines[extent_next] = {word + 1, generation, bytes};
        extent_next = (extent_next + 1) % kExtentLines;
        return bytes;
    }

    // extent_bytes of `word`, from this CPU's lines while the generation holds.
    inline uint64_t extent_cached(uint32_t word) {
        const uint32_t generation = bounds_generation();
        for (uint32_t line = 0; line < kExtentLines; line++) {
            if (
                extent_lines[line].tag == word + 1 &&
                extent_lines[line].generation == generation
            ) {
                return extent_lines[line].bytes;
            }
        }
        return extent_missed(word, generation);
    }
#endif

    // A page range past its tensor: recorded for the command the kernel runs, false.
    template <typename Kernel>
    __attribute__((noinline)) bool page_range_refused(
        const Kernel& core, uint32_t page
    ) {
        core.control(KM_CONTROL_LIVE)[kLvLine] = 1u << 31 | page;  // the whole page
        core.gave_up(
            core.role() == kRoleWriter ? kStWriter : kStReader,
            kGaveBadItem,
            kWherePageRange,
            core.running_slot(),  // the page: kLvLine's (8 bits here would cut it)
            core.running_sequence()
        );
        return false;
    }

    // Whether pages first .. first + pages - 1 of `bytes` lie inside tensor `word` (no
    // pages always do); else recorded for the command the kernel runs and false.
    template <typename Kernel>
    inline bool page_range_ok(
        const Kernel& core,
        uint32_t word,
        uint32_t first,
        uint32_t pages,
        uint32_t bytes
    ) {
        const uint64_t extent = extent_cached(word);
        const uint64_t end = (static_cast<uint64_t>(first) + pages) * bytes;
        if (pages == 0 || extent == 0 || end <= extent) {
            return true;
        }
        return page_range_refused(core, first + pages - 1);
    }

    // A page write refused at the caller's `line`: its source or page bytes not a
    // multiple of 16 (bad_transfer in kWherePageSource).
    template <typename Kernel>
    __attribute__((noinline)) bool page_source_refused(
        const Kernel& core, uint32_t line
    ) {
        return core.gave_up_at(kGaveBadTransfer, kWherePageSource, line);
    }

    // Whether a page write from L1 `from`, `bytes` a page, keeps each page's low four
    // address bits (the NoC lands a write at its destination's); else refused.
    template <typename Kernel>
    inline bool page_source_ok(
        const Kernel& core, uint32_t from, uint32_t bytes, SourceLine line
    ) {
        return ((from | bytes) & 15u) == 0 || page_source_refused(core, line.value);
    }
#else
    template <typename Kernel>
    inline bool page_range_ok(const Kernel&, uint32_t, uint32_t, uint32_t, uint32_t) {
        return true;
    }

    template <typename Kernel>
    inline bool page_source_ok(const Kernel&, uint32_t, uint32_t) {
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
            return (page_range_ok(core, form.word, first, pages, form.page_bytes()));
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
        return page_range_ok(core, tensor, first, pages, bytes);
    }

}  // namespace km
