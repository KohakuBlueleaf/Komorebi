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

#if !defined(KM_TENSOR_TABLE)
    // The session's list's entry with the largest address at most `word` (address,
    // lines, lines in each bank), else nullptr (no list, or none so low).
    inline const volatile tt_l1_ptr uint32_t* listed_at(uint32_t word) {
        const uint32_t list = island_area()[KM_ISLAND_TENSORS / 4];
        if (list == 0) {
            return nullptr;
        }
        const volatile tt_l1_ptr uint32_t* words = l1(list);
        uint32_t low = 0, high = words[0];
        while (low < high) {  // sorted by address
            const uint32_t middle = (low + high) / 2;
            if (words[1 + 3 * middle] <= word) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low == 0 ? nullptr : words + 1 + 3 * (low - 1);
    }
#endif

    // The bytes of DRAM tensor `word` (0: unchecked): its entry's (word 3), its part's
    // or the session's list's; `bank`: the list's bytes in each bank (else 0).
    inline uint64_t extent_bytes(uint32_t word, uint64_t& bank) {
        bank = 0;
#if defined(KM_TENSOR_TABLE)
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, word);
        if (entry == nullptr) {
            return extent_at_address(view, word);
        }
        if ((entry[3] & 3u) == KM_TENSOR_BANK) {
            return 0;  // a range of one bank: bank_page_at bounds it by its entry
        }
        return static_cast<uint64_t>(entry[3] >> 2) * entry[0];
#else
        const volatile tt_l1_ptr uint32_t* entry = listed_at(word);
        if (entry != nullptr && entry[0] == word) {
            bank = static_cast<uint64_t>(entry[2]) << 6;
            return static_cast<uint64_t>(entry[1]) << 6;
        }
        return 0;
#endif
    }

#if !defined(KM_OVERLAY_PART)
    // `extent` / `bytes` for an extent of 4 GiB or more, by shifts (no 64-bit division
    // routine in the image).
    inline uint64_t extent_divided(uint64_t extent, uint32_t bytes) {
        uint64_t quotient = 0, rest = 0;
        for (int bit = 63; bit >= 0; bit--) {
            rest = rest << 1 | (extent >> bit & 1u);
            if (rest >= bytes) {
                rest -= bytes;
                quotient |= uint64_t{1} << bit;
            }
        }
        return quotient;
    }

    // A miss: the bound looked up (extent_bytes) in pages of `bytes`, at most the banks
    // times the pages its bytes in each bank hold, kept in the next line; unknown: all.
    __attribute__((noinline)) inline uint32_t bound_missed(
        uint32_t word, uint32_t bytes
    ) {
        uint64_t bank = 0;
        const uint64_t extent = extent_bytes(word, bank);
        uint32_t pages = 0xFFFFFFFFu;
        if (extent != 0 && bytes != 0) {
            uint64_t whole = (
                extent >> 32 == 0
                    ? static_cast<uint32_t>(extent) / bytes
                    : extent_divided(extent, bytes)
            );
            if (bank != 0 && bank >> 32 == 0) {  // a bank holds less than 4 GiB
                const uint64_t banked = (
                    static_cast<uint64_t>(static_cast<uint32_t>(bank) / bytes) *
                    NUM_DRAM_BANKS
                );
                whole = banked < whole ? banked : whole;
            }
            pages = whole < 0xFFFFFFFFu ? static_cast<uint32_t>(whole) : 0xFFFFFFFEu;
        }
        bound_lines[bound_next] = {word + 1, bytes, pages};
        bound_next = (bound_next + 1) % kBoundLines;
        return pages;
    }

    // The pages of `bytes` tensor `word` holds, from this CPU's lines (emptied at a
    // command whose bounds' generation moved: bounds_command).
    inline uint32_t bound_pages(uint32_t word, uint32_t bytes) {
        for (const BoundLine& line : bound_lines) {
            if (line.tag == word + 1 && line.bytes == bytes) {
                return line.pages;
            }
        }
        return bound_missed(word, bytes);
    }
#endif

    // Page `page` past its tensor: recorded for the command the kernel runs. False.
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
#if defined(KM_OVERLAY_PART)  // an overlay part keeps no lines: the lookup
        uint64_t bank = 0;
        const uint64_t extent = extent_bytes(word, bank);
        const uint64_t last = static_cast<uint64_t>(first) + pages - 1;
        const uint64_t end = (last + 1) * bytes;
        const uint64_t bank_end = (last / NUM_DRAM_BANKS + 1) * bytes;  // in its bank
        return (
            pages == 0 || extent == 0 ||
            (end <= extent && (bank == 0 || bank_end <= bank)) ||
            page_range_refused(core, first + pages - 1)
        );
#else
        const uint32_t bound = bound_pages(word, bytes);
        return (
            pages == 0 || (pages <= bound && first <= bound - pages) ||
            page_range_refused(core, first + pages - 1)
        );
#endif
    }

    // Whether pages first .. first + pages - 1 of `bytes` of the bank range at bank
    // address `word` end inside the bank bytes of the listed tensor holding `word`.
    template <typename Kernel>
    inline bool bank_range_ok(
        const Kernel& core,
        uint32_t word,
        uint32_t first,
        uint32_t pages,
        uint32_t bytes
    ) {
#if defined(KM_TENSOR_TABLE)  // over islands its entry bounds it (bank_page_at)
        return true;
#else
        const volatile tt_l1_ptr uint32_t* entry = listed_at(word);
        if (pages == 0 || entry == nullptr) {
            return true;  // in no listed tensor: unchecked
        }
        const uint64_t end = entry[0] + (static_cast<uint64_t>(entry[2]) << 6);
        const uint64_t last = word + (static_cast<uint64_t>(first) + pages) * bytes;
        return (
            word >= end || last <= end || page_range_refused(core, first + pages - 1)
        );
#endif
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
    inline bool bank_range_ok(const Kernel&, uint32_t, uint32_t, uint32_t, uint32_t) {
        return true;
    }

    template <typename Kernel>
    inline bool page_source_ok(const Kernel&, uint32_t, uint32_t) {
        return true;
    }
#endif

    // Whether `Form` is a range of one bank (km::BankRange).
    template <typename Form, typename = decltype(&Form::bank)>
    constexpr bool extent_banked(int) {
        return true;
    }

    template <typename Form>
    constexpr bool extent_banked(...) {
        return false;
    }

    // page_range_ok of a Tensor or a SizedTensor, bank_range_ok of a BankRange.
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
            return bank_range_ok(core, form.word, first, pages, form.page_bytes());
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
#if KM_GUARD && defined(KM_TENSOR_TABLE)
        // a sharded L1 tensor on islands: no entry, this island's rows its pages alone
        if constexpr (!Args::is_dram && Args::is_sharded) {
            const uint32_t volume = (
                TensorAccessor(args, tensor).dspec().tensor_volume()
            );
            if (pages != 0 && static_cast<uint64_t>(first) + pages > volume) {
                return page_range_refused(core, first + pages - 1);
            }
        }
#endif
        return page_range_ok(core, tensor, first, pages, bytes);
    }

}  // namespace km
