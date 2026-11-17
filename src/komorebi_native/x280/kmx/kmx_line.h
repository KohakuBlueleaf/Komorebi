/* A line of the link service as an X280 rank writes it into its ring: the same last
 * word the link cores check (km_pages.hpp line_end), in C. A host test folds lines
 * with both and compares them. */
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
