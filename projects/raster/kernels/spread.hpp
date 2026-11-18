// The option RASTER_SPREAD on the movers: a record's 21 broadcast numbers into a tile,
// one row group (8 lanes) each, in the Dst window's order; the reader and the writer
// both use it.
#pragma once

#include <stdint.h>

#include "raster.h"

// Record word of each spread vector (the fill's kConst order: a, b, c, w, colour a, b,
// c).
constexpr uint8_t kSpreadWord[kSpreadVectors] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 18, 21, 16, 19, 22, 17, 20, 23
};

// A pair of slots is 4 rows x (even, odd); row k of slot s holds vector 4 s + k (its 8
// lanes), which SFPTRANSP on the pack thread copies to all 32 lanes. Stores in address
// order (volatile), so the store queue joins them into 16 bytes.
inline void spread_tile(volatile uint32_t* tile, const volatile uint32_t* record) {
    uint32_t values[kSpreadSlotVectors];
#pragma GCC unroll 21
    for (uint32_t vector = 0; vector < kSpreadVectors; ++vector) {
        values[vector] = record[kSpreadWord[vector]];
    }
    for (uint32_t vector = kSpreadVectors; vector < kSpreadSlotVectors; ++vector) {
        values[vector] = 0;
    }
    for (uint32_t vector = 0; vector < kSpreadSlotVectors; vector += 8) {
        volatile uint32_t* block = tile + 8 * vector;
#pragma GCC unroll 4
        for (uint32_t row = 0; row < 4; ++row) {
            const uint32_t even = values[vector + row], odd = values[vector + 4 + row];
#pragma GCC unroll 8
            for (uint32_t word = 0; word < 16; word += 2) {
                block[16 * row + word] = even;
                block[16 * row + word + 1] = odd;
            }
        }
    }
}
