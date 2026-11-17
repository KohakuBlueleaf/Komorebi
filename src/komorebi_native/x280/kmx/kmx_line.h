/* A link service line's last word as an X280 rank seals it: km_pages.hpp line_end in
 * C (tests/core/device/test_line_end.py folds lines with both). */
#pragma once

#include <stdint.h>

#include "komorebi/abi/km_abi.h"

/* Line `line`'s last word (KM_ASK_END) for number `number`: the number folded with
 * words KM_ASK_NUMBER + 1 to KM_ASK_END - 1, each after a rotation by 5. */
static inline uint32_t kmx_line_end(const uint32_t* line, uint32_t number) {
    uint32_t check = number;
    for (uint32_t w = KM_ASK_NUMBER + 1u; w < KM_ASK_END; w++) {
        check = (check << 5 | check >> 27) ^ line[w];
    }
    return check;
}
