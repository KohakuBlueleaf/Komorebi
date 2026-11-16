// A tensor table over KM_TABLE_CACHE entries: its rows in the island's DRAM, each RISC
// that looks one up keeping KM_TABLE_CACHE of them in L1 (two ways a set).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_island.hpp"

namespace km {

    constexpr uint32_t kCacheSets = KM_TABLE_CACHE / 2;
    // a RISC's part: tags (word + 1, 0 empty), a way to take next a set, then the rows
    constexpr uint32_t kCacheHead = (
        (KM_TABLE_CACHE * 4 + kCacheSets * 4 + 63) / 64 * 64
    );

    // The bytes of a row in DRAM and in a cache slot: an entry in whole 64 B lines.
    inline uint32_t cache_row_bytes(const Islands& view) {
        const uint32_t bytes = (
            KM_TENSOR_HEADER_BYTES + view.count() * KM_TENSOR_PART_BYTES
        );
        return (bytes + 63) / 64 * 64;
    }

    // The cache of RISC `risc` (0 reader, 1 writer) after the table's head at `table`.
    inline uint32_t cache_part(uint32_t table, uint32_t row, uint32_t risc) {
        return table + KM_TABLE_HEAD_BYTES + risc * (kCacheHead + KM_TABLE_CACHE * row);
    }

    // Entry `tensor` from this RISC's cache, read from DRAM into the slot its set takes
    // next when it is not there; a pair of words looked up in turn both stay.
    inline const volatile uint32_t* cached_entry(
        const Islands& view, uint32_t table, uint32_t tensor
    ) {
        const uint32_t row = cache_row_bytes(view);
#if defined(COMPILE_FOR_NCRISC)
        const uint32_t part = cache_part(table, row, 1);
#else
        const uint32_t part = cache_part(table, row, 0);
#endif
        volatile tt_l1_ptr uint32_t*
            tags = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part));
        volatile tt_l1_ptr uint32_t* next = tags + KM_TABLE_CACHE;
        const uint32_t set = tensor % kCacheSets;
        uint32_t way = 0;
        while (way < 2 && tags[2 * set + way] != tensor + 1) {
            way++;
        }
        const bool hit = way < 2;
        way = hit ? way : next[set];
        const uint32_t slot = part + kCacheHead + (2 * set + way) * row;
        if (!hit) {
            const uint32_t dram = (
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table)
            );
            const InterleavedAddrGen<true> rows = {
                .bank_base_address = dram, .page_size = row
            };
            tags[2 * set + way] = 0;
            noc_async_read(rows.get_noc_addr(tensor), slot, row);
            noc_async_read_barrier();
            invalidate_l1_cache();
            tags[2 * set + way] = tensor + 1;
        }
        next[set] = way ^ 1u;
        return reinterpret_cast<const volatile uint32_t*>(slot);
    }

    // A row written while the run lives: word `tensor`'s slots in both RISCs' caches
    // emptied, so its next lookup reads the DRAM row the host wrote.
    inline void forget_entry(const Islands& view, uint32_t table, uint32_t tensor) {
        const uint32_t row = cache_row_bytes(view);
        const uint32_t set = tensor % kCacheSets;
        for (uint32_t risc = 0; risc < 2; risc++) {
            volatile tt_l1_ptr uint32_t*
                tags = (
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        cache_part(table, row, risc)
                    )
                );
            for (uint32_t way = 0; way < 2; way++) {
                if (tags[2 * set + way] == tensor + 1) {
                    tags[2 * set + way] = 0;
                }
            }
        }
    }

}  // namespace km
