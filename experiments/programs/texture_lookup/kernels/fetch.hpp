// The probe's DRAM reads on a data-movement CPU: pages into the table's slots, several
// in flight (MISS); real tiles' lookups with each miss read as it comes (REPLAY).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "lookup.h"
#include "table.hpp"

namespace texture {

    // Spins of the wait for reads in flight before it gives up (a read lands in
    // about 1,000 cycles; this bound is a few milliseconds).
    constexpr uint32_t kReadSpins = 1u << 20;

    // Waits for every read this CPU has in flight. False: they did not land in time.
    inline bool reads_landed() {
        for (uint32_t spin = 0; spin < kReadSpins; spin++) {
            if (ncrisc_noc_reads_flushed(noc_index)) {
                invalidate_l1_cache();
                return true;
            }
        }
        return false;
    }

    // MISS: `reps` times a1 pages of a2 bytes from the pool into slots 0 to a1 - 1,
    // issued together, timed from the first issue until all landed.
    template <typename Pool>
    inline void miss(
        const Tensors& at,
        const Pool& pool,
        uint32_t pool_pages,
        uint32_t me,
        const km_command_t& command
    ) {
        uint32_t* results = words(at.results);
        const uint32_t count = command.a1, bytes = command.a2;
        const uint32_t reps = command.a3 < kRepsMax ? command.a3 : kRepsMax;
        uint32_t error = 0;
        if (
            count == 0 || count > kInFlightMax || bytes == 0 || bytes > kPageBytes ||
            bytes % 64 != 0 || pool_pages < kInFlightMax
        ) {
            error = kErrorArgs;
        }
        uint32_t sum = 0;
        for (uint32_t rep = 0; error == 0 && rep < reps; rep++) {
            const uint32_t first = (me * 97 + rep * count) % pool_pages;
            const uint32_t start = km::wall();
            for (uint32_t page = 0; page < count; page++) {
                noc_async_read(
                    pool.get_noc_addr((first + page) % pool_pages),
                    at.slots + (page << kPageShift),
                    bytes
                );
            }
            if (!reads_landed()) {
                error = kErrorRead;
                break;
            }
            results[kResultReps + rep] = km::wall() - start;
            for (uint32_t page = 0; page < count; page++) {
                sum += words(at.slots + (page << kPageShift))[bytes / 4 - 1];
            }
        }
        results[kResultSum] = sum;
        results[kResultError] = error;
        results[kResultRun] = command.a9;
        fence();
    }

    // REPLAY: this core's items (key: pool page key % pool_pages) through a0 slots in
    // form a1, a2 passes; a miss reads its page into the slot it replaces, then waits.
    template <typename Pool>
    inline void replay(
        const Tensors& at,
        const Pool& pool,
        uint32_t pool_pages,
        const km_command_t& command
    ) {
        uint32_t* results = words(at.results);
        const uint32_t* replay_words = words(at.replay);
        const uint32_t items = replay_words[0];
        const uint32_t passes = command.a2;
        const bool keep_last = command.a1 == kFormHashLast;
        Table table;
        uint32_t error = 0;
        if (
            !table.clear(command.a0) || items > kItemsMax || passes == 0 ||
            passes * items > 2 * kItemsMax || pool_pages == 0
        ) {
            error = kErrorArgs;
        }
        uint32_t* out = words(at.out);
        uint32_t* record = results + kResultItems;
        uint32_t total_misses = 0;
        for (uint32_t pass = 0; error == 0 && pass < passes; pass++) {
            for (uint32_t item = 0; error == 0 && item < items; item++) {
                const uint32_t* word = (
                    replay_words + kReplayWordsAt + replay_words[1 + 2 * item]
                );
                const uint32_t count = replay_words[2 + 2 * item];
                uint32_t sum = 0, misses = 0, last_key = kNone, base = at.slots;
                const uint32_t start = km::wall();
                for (uint32_t index = 0; index < count; index++) {
                    const uint32_t key = word[index] >> kOffsetBits;
                    if (!keep_last || key != last_key) {
                        uint32_t slot = table.find(key);
                        if (slot == kNone) {
                            slot = table.victim(key);
                            noc_async_read(
                                pool.get_noc_addr(key % pool_pages),
                                at.slots + (slot << kPageShift),
                                kPageBytes
                            );
                            if (!reads_landed()) {
                                error = kErrorRead;
                                break;
                            }
                            table.tag[slot] = key + 1;
                            misses++;
                        }
                        table.use[slot] = ++table.clock;
                        last_key = key;
                        base = at.slots + (slot << kPageShift);
                    }
                    const uint32_t texel = words(base)[word[index] & kOffsetMask];
                    out[index & (kOutWords - 1)] = texel;
                    sum += texel;
                }
                record[0] = km::wall() - start;
                record[1] = misses;
                record[2] = sum;
                record += 3;
                total_misses += misses;
            }
        }
        results[kResultMisses] = total_misses;
        results[kResultError] = error;
        results[kResultRun] = command.a9;
        fence();
    }

}  // namespace texture
