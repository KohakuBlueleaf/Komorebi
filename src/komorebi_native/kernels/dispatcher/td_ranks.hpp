// A dynamic run's X280 ranks, out's side (the X280's messages_run_line): the run line
// {run, KM_X280_RUN_*, group, handler} into their thread's area (KMD_WORK_RUN).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "td_sources.hpp"

namespace td {

    // In L1 at kTbRanks (kRanksBytes): the line first, 16-byte aligned.
    struct Ranks {
        uint32_t line[4];   // KM_X280_RUN_*
        uint32_t count;     // X280 ranks, 0 to KM_X280_RANKS_MAX (0: none)
        uint32_t index;     // their thread's: cluster - 1 (0 without them)
        uint32_t counters;  // a part's counter tiles (over islands, the counts below)
        uint32_t unstored;  // 1: the line waits until the thread serves
        uint32_t told;      // the run whose START was stored and whose END is owed

        void init(uint32_t word) {
            count = (word >> 4) & 0xFu;
            index = count != 0 && (word & 0xFu) != 0 ? (word & 0xFu) - 1u : 0u;
            counters = (word >> 8) & 0xFFu;
            unstored = told = 0;
        }

        // The line to the thread's area (an X280 thread's; none: nothing to tell).
        void store(const ThreadArea& work) {
            if (work.remote != 0) {
                noc_async_write(
                    reinterpret_cast<uint32_t>(line),
                    work.remote + KMD_WORK_RUN(work.event_depth),
                    16
                );
            }
            unstored = 0;
        }

        // The handler word's counts over islands `low` to `high` (kmd_x280_counts; 3
        // `entries` an island): `here`'s message ranks << 8, the group's << 16.
        uint32_t counts(
            const uint32_t* entries,
            uint32_t low,
            uint32_t high,
            uint32_t here,
            uint32_t group
        ) const {
            const uint32_t first = group & 0xFFFFu, end = first + (group >> 16);
            uint32_t total = 0, own = 0;
            for (uint32_t island = low; island <= high; island++) {
                const uint32_t range = entries[3 * island];
                const uint32_t start_at = range & 0xFFFFu;
                const uint32_t stop_at = start_at + (range >> 16);
                const uint32_t start = first > start_at ? first : start_at;
                const uint32_t stop = end < stop_at ? end : stop_at;
                const uint32_t held = (
                    stop > start + counters ? stop - start - counters : 0u
                );
                total += held;
                own = island == here ? held : own;
            }
            return own << 8 | total << 16;
        }

        // Run `run` of group `group` and handler `handler` starts: its line stored now
        // if the thread serves, else once it does (Lines::answered).
        void start(
            const ThreadArea& work,
            bool serving,
            uint32_t run,
            uint32_t group,
            uint32_t handler
        ) {
            if (count == 0) {
                return;
            }
            line[KM_X280_RUN_GROUP] = group;
            line[KM_X280_RUN_HANDLER] = handler;
            told = run;
            put(work, serving, run, KM_X280_RUN_START);
        }

        // Run `run` ends, by its end command or otherwise: its END, once.
        void end(const ThreadArea& work, bool serving, uint32_t run) {
            if (count != 0 && told == run && run != 0) {
                told = 0;
                put(work, serving, run, KM_X280_RUN_END);
            }
        }

        void put(const ThreadArea& work, bool serving, uint32_t run, uint32_t state) {
            line[KM_X280_RUN_SEQUENCE] = run;
            line[KM_X280_RUN_STATE] = state;
            asm volatile("" ::: "memory");
            if (serving) {
                store(work);
            } else {
                unstored = 1;
            }
        }
    };

    static_assert(sizeof(Ranks) <= kRanksBytes, "the X280 ranks' line");

}  // namespace td
