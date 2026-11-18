// A tensor table over KM_TABLE_CACHE entries: its rows in the island's DRAM, each RISC
// that looks one up keeping KM_TABLE_CACHE of them in L1 (four ways a set).
#pragma once

#include <stdint.h>

#include <stddef.h>

#include "api/dataflow/dataflow_api.h"
#include "../core/km_core.hpp"
#include "../core/km_island.hpp"

namespace km {

    constexpr uint32_t kCacheWays = 4;
    constexpr uint32_t kCacheSets = KM_TABLE_CACHE / kCacheWays;
    static_assert(kCacheSets * kCacheWays == KM_TABLE_CACHE, "4 ways a set");
    // a RISC's part: tags (word + 1, 0 empty), each slot's last use, the clock of
    // uses, the misses (km.Session.table_misses reads them); then the rows
    constexpr uint32_t kCacheClock = 2 * KM_TABLE_CACHE;
    constexpr uint32_t kCacheMisses = kCacheClock + 1;
    constexpr uint32_t kCacheHead = ((kCacheMisses + 1) * 4 + 63) / 64 * 64;

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

    // This RISC's reads landed within the wait limit; else its command is given up
    // (`tablerow`) and the CPU stops here: no entry to answer with (KM_GUARD's way).
    inline void row_landed() {
#if defined(COMPILE_FOR_TRISC)
        noc_async_read_barrier();
#else
        const uint32_t control = get_arg_val<uint32_t>(0);
        const uint32_t limit = get_arg_val<uint32_t>(offsetof(Args, wait_limit) / 4);
#if defined(COMPILE_FOR_NCRISC)
        constexpr uint32_t role = kRoleWriter, word = kStWriter;
#else
        constexpr uint32_t role = kRoleReader, word = kStReader;
#endif
        const uint32_t code = poll_live(
            control,
            role,
            kWhereTableRow,
            false,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            limit,
            limit  // a stop meanwhile: the read still gets its limit to land
        );
        if (code == kGaveNone) {
            return;
        }
        volatile tt_l1_ptr uint32_t* live = l1(control + KM_CONTROL_LIVE);
        command_gave_up(
            control, word, code, kWhereTableRow, 0, live[kLvSequence + role]
        );
        live[kLvWhere + role] = kWhereTableRow;
        for (;;) {
            asm volatile("");
        }
#endif
    }

    // Entry `tensor` from this RISC's cache (set: the word mod kCacheSets), read from
    // DRAM into its set's least recently used slot when it is not there.
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
        volatile tt_l1_ptr uint32_t* used = tags + KM_TABLE_CACHE;
        const uint32_t first = tensor % kCacheSets * kCacheWays;
        const uint32_t now = tags[kCacheClock] + 1;
        tags[kCacheClock] = now;
        uint32_t slot = first, oldest = first;
        while (slot < first + kCacheWays && tags[slot] != tensor + 1) {
            oldest = now - used[slot] > now - used[oldest] ? slot : oldest;
            slot++;
        }
        const bool hit = slot < first + kCacheWays;
        slot = hit ? slot : oldest;
        const uint32_t at = part + kCacheHead + slot * row;
        if (!hit) {
            const uint32_t dram = (
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table)
            );
            const InterleavedAddrGen<true> rows = {
                .bank_base_address = dram, .page_size = row
            };
            tags[slot] = 0;
            noc_async_read(rows.get_noc_addr(tensor), at, row);
            row_landed();
            invalidate_l1_cache();
            tags[slot] = tensor + 1;
            tags[kCacheMisses] = tags[kCacheMisses] + 1;
        }
        used[slot] = now;
        return reinterpret_cast<const volatile uint32_t*>(at);
    }

    // A row written while the run lives: word `tensor`'s slots in both RISCs' caches
    // emptied, so its next lookup reads the DRAM row the host wrote.
    inline void forget_entry(const Islands& view, uint32_t table, uint32_t tensor) {
        const uint32_t row = cache_row_bytes(view);
        const uint32_t first = tensor % kCacheSets * kCacheWays;
        for (uint32_t risc = 0; risc < 2; risc++) {
            volatile tt_l1_ptr uint32_t*
                tags = (
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        cache_part(table, row, risc)
                    )
                );
            for (uint32_t slot = first; slot < first + kCacheWays; slot++) {
                if (tags[slot] == tensor + 1) {
                    tags[slot] = 0;
                }
            }
        }
    }

}  // namespace km
