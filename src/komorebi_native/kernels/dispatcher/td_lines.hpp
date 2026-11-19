// Commands to the thread (kmx_program), out's side, as the X280's work_port.h: a line
// per command in the thread's in-lines (KMD_WORK_IN), its answer, its deadline.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_sources.hpp"

namespace td {

    // The thread's lines, at most KMD_WORK_IN_DEPTH unanswered: request, source,
    // sequence, times. Out keeps them in L1 (OutState); init() before the first use.
    struct Lines {
        uint32_t next;      // the next line's number, from 1
        uint32_t busy;      // bit slot: a line unanswered
        uint32_t late;      // bit slot: its TIMEOUT sent, its answer owes none
        uint32_t result;    // bit slot: KM_FLAG_RESULT (the answer word as ticks)
        uint32_t unstored;  // bit slot: kept here, not yet at an X280 thread
        bool serving;       // the thread answered line 0: it takes lines
        uint32_t number[KMD_WORK_IN_DEPTH];
        uint32_t request[KMD_WORK_IN_DEPTH];
        uint32_t sequence[KMD_WORK_IN_DEPTH];
        uint8_t source[KMD_WORK_IN_DEPTH];
        uint64_t submitted[KMD_WORK_IN_DEPTH];
        uint64_t deadline[KMD_WORK_IN_DEPTH];

        // No line yet; the times are written with each line.
        void init() {
            next = 1;
            busy = late = result = unstored = 0;
            serving = false;
        }

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

        // The thread's answer to line `number` (0: it serves, the kept lines go to
        // it): the slot whose DONE is owed, else KMD_WORK_IN_DEPTH (late, or none).
        __attribute__((noinline)) uint32_t
        answered(const ThreadArea& work, uint32_t number) {
            if (number == 0) {
                for (uint32_t back = KMD_WORK_IN_DEPTH; !serving && back > 0; back--) {
                    const uint32_t slot = (next - back) & (KMD_WORK_IN_DEPTH - 1u);
                    if (unstored & (1u << slot)) {
                        store(work, slot);
                    }
                }
                unstored = 0;
                serving = true;
                return KMD_WORK_IN_DEPTH;
            }
            const uint32_t slot = number & (KMD_WORK_IN_DEPTH - 1u), bit = 1u << slot;
            if (!(busy & bit) || this->number[slot] != number) {
                return KMD_WORK_IN_DEPTH;
            }
            busy &= ~bit;
            const bool was_late = (late & bit) != 0;
            late &= ~bit;
            return was_late ? KMD_WORK_IN_DEPTH : slot;
        }

        // Line `slot`'s time at `now`: in its source's clock, a thread's (thread_ticks
        // by `scale`) or the wall clock.
        uint32_t waited(uint32_t slot, uint64_t now, uint32_t scale) const {
            const uint32_t time = static_cast<uint32_t>(now - submitted[slot]);
            return source[slot] == kSourceThread ? thread_ticks(time, scale) : time;
        }

        // The DONE's ticks of line `slot`: the answer's word with KM_FLAG_RESULT, else
        // its time.
        uint32_t ticks(
            uint32_t slot, uint32_t word, uint64_t now, uint32_t scale
        ) const {
            return result & (1u << slot) ? word : waited(slot, now, scale);
        }

        // A line past its deadline at `now`, marked late: its slot (its TIMEOUT owed),
        // else KMD_WORK_IN_DEPTH.
        __attribute__((noinline)) uint32_t late_one(uint64_t now) {
            for (uint32_t slot = 0; slot < KMD_WORK_IN_DEPTH; slot++) {
                const uint32_t bit = 1u << slot;
                if ((busy & ~late & bit) && now > deadline[slot]) {
                    late |= bit;
                    return slot;
                }
            }
            return KMD_WORK_IN_DEPTH;
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
