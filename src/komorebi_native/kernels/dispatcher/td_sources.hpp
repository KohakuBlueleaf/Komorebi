// The Tensix dispatcher's sources of commands: one command ring each in this core's L1,
// under the X280 dispatcher's protocol with the host. A source added later is a row.
#pragma once

#include <stdint.h>

#include "komorebi/device/core/km_core.hpp"

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
        uint32_t take(volatile tt_l1_ptr uint32_t* pending) {
            volatile tt_l1_ptr uint32_t* at = line();
            for (uint32_t word = 0; word < 16; word++) {
                pending[word] = at[word];
            }
            return next++;
        }
    };

    // The sources this dispatcher serves, by index; the host is source 0.
    constexpr uint32_t kSourceHost = 0;
    constexpr uint32_t kSources = 1;

}  // namespace td
