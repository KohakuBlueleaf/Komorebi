// The Tensix dispatcher's look at a dynamic run's status lines (the X280's
// messages_sample), for "in": the lines land in the block at kTbLines.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

    // Line `index` of a run of `count` cores from `from`: its cores, then its X280
    // ranks' (KM_X280_RANK_CORE), read as other cores (the X280's messages_line).
    inline uint32_t run_line(uint32_t from, uint32_t count, uint32_t index) {
        return index < count ? from + index : KM_X280_RANK_CORE(index - count);
    }

    // `group`'s lines, then `x280` X280 ranks', twice: 1 all of run `run`, idle and
    // unchanged (sums in `sent`, `consumed`); 0 not yet; -1 `gave` gave up.
    inline int sample_lines(
        uint32_t block,
        uint32_t run,
        uint32_t group,
        uint32_t x280,
        uint32_t& gave,
        uint64_t& sent,
        uint64_t& consumed
    ) {  // fmt: skip
        volatile tt_l1_ptr uint32_t* lines = km::l1(block + kTbLines);
        volatile tt_l1_ptr uint32_t* first = km::l1(block + kTbSample);
        const uint32_t from = group & 0xFFFFu, count = group >> 16;
        invalidate_l1_cache();
        for (uint32_t index = 0; index < count + x280; index++) {
            const uint32_t core = run_line(from, count, index);
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
        sent = consumed = 0;
        for (uint32_t index = 0; index < count + x280; index++) {
            const uint32_t core = run_line(from, count, index);
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
        return 1;
    }

}  // namespace td
