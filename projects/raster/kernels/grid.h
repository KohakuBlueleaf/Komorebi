// The frame's grid of tiles in a8 of every command but SHOW's (kernels/README.md): a
// tile is ty TX + tx, the frame rendered 32 TX x 32 TY. Included before the namespaces.
#pragma once

#include <stdint.h>

constexpr uint32_t kGridDefault = 32;
constexpr uint32_t kGridAcrossMax = 64;  // SETUP's box word: 6 bits of tile column
constexpr uint32_t kGridDownMax = 32;    // its pixel rows: 10 bits

struct Grid {
    uint32_t across, down;
};

// Bits 0 to 7: 0, or 1 + the X280 thread's regions' bank.
inline uint32_t bank_word(uint32_t a8) {
    return a8 & 0xFF;
}

// Bits 8 to 15 TX, 16 to 23 TY; 0 there is 32 (the 1,024 x 1,024 frame).
inline Grid grid_of(uint32_t a8) {
    const uint32_t across = a8 >> 8 & 0xFF, down = a8 >> 16 & 0xFF;
    return {across ? across : kGridDefault, down ? down : kGridDefault};
}
