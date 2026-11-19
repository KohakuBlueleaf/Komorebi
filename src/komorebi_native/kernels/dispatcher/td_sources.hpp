// The Tensix dispatcher's sources of commands: one command ring each in this core's L1,
// under the X280 dispatcher's protocol with the host. A source added later is a row.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_slots.hpp"

namespace td {

    // One source's ring: command i (from 1) is the 64-byte line i % depth, taken once
    // its four landing words show i.
    struct Source {
        volatile tt_l1_ptr uint32_t* ring = nullptr;
        uint32_t depth = 0;  // lines, a power of two
        uint32_t next = 1;   // the number of the next command to take

        void init(volatile tt_l1_ptr uint32_t* lines, uint32_t lines_depth) {
            ring = lines;
            depth = lines_depth;
            next = 1;
        }

        // The line command `next` lands in.
        volatile tt_l1_ptr uint32_t* line() const {
            return ring + (next & (depth - 1u)) * 16;
        }

        // Whether command `next` has landed: all four landing words show its number.
        bool landed() const {
            volatile tt_l1_ptr uint32_t* at = line();
            return at[0] == next && at[4] == next && at[8] == next && at[12] == next;
        }

        // Copies the landed command into `pending` and steps on; returns its number.
        uint32_t take(uint32_t* pending) {
            volatile tt_l1_ptr uint32_t* at = line();
            for (uint32_t word = 0; word < 16; word++) {
                pending[word] = at[word];
            }
            return next++;
        }
    };

    // A thread's area, laid out as the X280's work area (KMD_WORK_*), wherever its
    // writer runs: commands taken, events written, the events, then the lines to it.
    struct ThreadArea {
        uint32_t base = 0;         // L1 address, 64-byte aligned
        uint32_t event_depth = 0;  // events of 16 bytes, a power of two

        // Written by out: the thread's commands taken (a line of the ring is free
        // again).
        volatile tt_l1_ptr uint32_t* consumed() const {
            return km::l1(base + KMD_WORK_CONSUMED);
        }

        // Written by in: events written into the ring below, the last one landed first.
        volatile tt_l1_ptr uint32_t* told() const {
            return km::l1(base + KMD_WORK_TOLD);
        }

        // Event `index` (from 1): {index, request, status, ticks}, as the host's ring.
        volatile tt_l1_ptr uint32_t* event(uint32_t index) const {
            return km::l1(base + KMD_WORK_EVENTS + (index & (event_depth - 1u)) * 16);
        }

        // Command line `number` (from 1) to the thread, until it answers it.
        volatile tt_l1_ptr uint32_t* in_line(uint32_t number) const {
            return (
                km::l1(
                    base + KMD_WORK_IN(event_depth) + (number % KMD_WORK_IN_DEPTH) * 64
                )
            );
        }

        uint32_t bytes() const {
            return KMD_WORK_IN(event_depth) + KMD_WORK_IN_DEPTH * 64;
        }
    };

    // The sources this dispatcher serves, by index: the host 0, the thread (the user's
    // code on this core, or a core that writes its ring over the NoC) 1, the engines 4.
    constexpr uint32_t kSourceHost = 0;
    constexpr uint32_t kSourceThread = 1;
    constexpr uint32_t kSourceEngine = 4;  // 2 and 3: threads on X280 clusters 2 and 3
    constexpr uint32_t kSources = 5;

    // The source of a slot's record: its event goes back there.
    inline uint32_t record_source(uint32_t flags) {
        return (
            flags & kRecordThread
                ? kSourceThread
                : flags & kRecordEngine
                    ? kSourceEngine
                    : kSourceHost
        );
    }

    static_assert(
        kTaWork + KMD_WORK_IN(kTaEventDepth) + KMD_WORK_IN_DEPTH * 64 == kTaState
    );
    static_assert(kTeWork + KMD_WORK_EVENTS + kTeEventDepth * 16 == kTeTable);

}  // namespace td
