// Commands to the thread (kmx_program), out's side, as the X280's work_port.h: a line
// per command in the thread's in-lines (KMD_WORK_IN), its answer, its deadline.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_sources.hpp"

namespace td {

    // The thread's lines: KMD_WORK_IN_DEPTH at most unanswered, each its command's
    // request, source, sequence and times; a line is kept in the area until stored.
    struct Lines {
        uint32_t next = 1;      // the next line's number, from 1
        uint32_t busy = 0;      // bit slot: a line unanswered
        uint32_t late = 0;      // bit slot: its TIMEOUT sent, its answer owes none
        uint32_t result = 0;    // bit slot: KM_FLAG_RESULT (the answer word as ticks)
        uint32_t unstored = 0;  // bit slot: kept here, not yet at an X280 thread
        bool serving = false;   // the thread answered line 0: it takes lines
        uint32_t number[KMD_WORK_IN_DEPTH] = {};
        uint32_t request[KMD_WORK_IN_DEPTH] = {};
        uint32_t sequence[KMD_WORK_IN_DEPTH] = {};
        uint8_t source[KMD_WORK_IN_DEPTH] = {};
        uint64_t submitted[KMD_WORK_IN_DEPTH] = {};
        uint64_t deadline[KMD_WORK_IN_DEPTH] = {};

        // Line `slot` to an X280 thread's area (a thread here reads it where it is).
        static void store(const ThreadArea& work, uint32_t slot) {
            work.mirror(KMD_WORK_IN(work.event_depth) + slot * 64, 64);
        }

        // `pending` as line `next`, from source `from` for request `asked`: its words,
        // its number in word 0 and the landing words; stored once the thread serves.
        void put(
            const ThreadArea& work,
            const uint32_t* pending,
            uint32_t from,
            uint32_t asked,
            uint32_t sequence_number,
            uint64_t now,
            uint64_t ticks
        ) {
            const uint32_t at = next, slot = at & (KMD_WORK_IN_DEPTH - 1u);
            volatile tt_l1_ptr uint32_t* line = work.in_line(at);
            line[1] = pending[1] & 0xFF00FFFFu;
            for (uint32_t word = 2; word < 16; word++) {
                line[word] = word % 4 == 0 ? 0 : pending[word];
            }
            asm volatile("" ::: "memory");
            line[0] = line[4] = line[8] = line[12] = at;
            if (serving) {
                store(work, slot);
            } else {
                unstored |= 1u << slot;
            }
            next = at + 1;
            number[slot] = at;
            request[slot] = asked;
            sequence[slot] = sequence_number;
            source[slot] = static_cast<uint8_t>(from);
            submitted[slot] = now;
            deadline[slot] = now + ticks;
            busy |= 1u << slot;
            const uint32_t flags = (pending[1] >> 8) & 0xFFu;
            result = (
                (result & ~(1u << slot)) | (flags & KM_FLAG_RESULT ? 1u << slot : 0u)
            );
        }

        bool full() const {
            return (busy & (1u << (next & (KMD_WORK_IN_DEPTH - 1u)))) != 0;
        }

        // The auxiliary byte of a line's event: its cluster (1) and its sequence.
        uint32_t auxiliary(uint32_t slot) const {
            return (
                1u |
                (sequence[slot] & KM_EVENT_SEQUENCE_MASK) << KM_EVENT_SEQUENCE_SHIFT
            );
        }
    };

}  // namespace td
