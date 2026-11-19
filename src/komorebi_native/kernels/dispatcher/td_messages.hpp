// The Tensix dispatcher's look at a dynamic run's status lines (the X280's
// messages_sample), for "in": the lines land in the block at kTbLines.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

    // `group`'s lines looked at twice: 1 all of run `run`, idle, unchanged, sent equal
    // to consumed; 0 not yet; -1 a line that gave up (`gave` its core).
    inline int sample_lines(
        uint32_t block, uint32_t run, uint32_t group, uint32_t& gave
    ) {
        volatile tt_l1_ptr uint32_t* lines = km::l1(block + kTbLines);
        volatile tt_l1_ptr uint32_t* first = km::l1(block + kTbSample);
        const uint32_t from = group & 0xFFFFu, end = from + (group >> 16);
        invalidate_l1_cache();
        for (uint32_t core = from; core < end; core++) {
            volatile tt_l1_ptr uint32_t* line = lines + 4 * core;
            if (line[KM_MESSAGES_LINE_RUN] != run) {
                return 0;
            }
            const uint32_t state = line[KM_MESSAGES_LINE_STATE];
            if (state == KM_MESSAGES_GAVE_UP) {
                gave = core;
                return -1;
            }
            if (state != KM_MESSAGES_IDLE) {
                return 0;
            }
            first[2 * core] = line[KM_MESSAGES_LINE_SENT];
            first[2 * core + 1] = line[KM_MESSAGES_LINE_CONSUMED];
        }
        invalidate_l1_cache();
        uint64_t sent = 0, consumed = 0;
        for (uint32_t core = from; core < end; core++) {
            volatile tt_l1_ptr uint32_t* line = lines + 4 * core;
            if (
                line[KM_MESSAGES_LINE_RUN] != run ||
                line[KM_MESSAGES_LINE_STATE] != KM_MESSAGES_IDLE ||
                line[KM_MESSAGES_LINE_SENT] != first[2 * core] ||
                line[KM_MESSAGES_LINE_CONSUMED] != first[2 * core + 1]
            ) {
                return 0;
            }
            sent += first[2 * core];
            consumed += first[2 * core + 1];
        }
        return sent == consumed ? 1 : 0;
    }

}  // namespace td
