// An X280 thread's region as a class handler takes it: a range of one DRAM bank, in
// pages of the region's granularity, for the page calls, item streams and puts.
#pragma once

#include <stdint.h>

#include "komorebi/device/tensors/km_ask.hpp"

namespace km {

    // A region's two command words: its bank address and its bank.
    struct Region {
        uint32_t word;
        uint32_t bank;
    };

    // A region's pages: a DRAM form for read_pages, write_pages and put_bytes, and an
    // address generator for a stream of its pages or items.
    template <uint32_t PageBytes>
    struct RegionPages : BankRange {
        static constexpr bool kLocal = true;

        explicit RegionPages(const Region& region) :
            BankRange(region.word, region.bank, PageBytes) {
        }

        uint64_t get_noc_addr(
            uint32_t page, uint32_t offset = 0, uint8_t noc = noc_index
        ) const {
            return (
                get_noc_addr_from_bank_id<true>(bank, word + page * bytes + offset, noc)
            );
        }

        uint32_t get_aligned_page_size() const {
            return bytes;
        }
    };

}  // namespace km
