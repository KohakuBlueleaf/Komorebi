// The overlay shift experiment: what the two programs, their overlays and Python agree
// on. Both programs run the same kernels; each has overlays of its own.
#pragma once

#include <stdint.h>

// TAG, handler 1 of each program: a0 the overlay to call in the program's own numbers,
// a1 a seed, a2 the lines' tensor (its word), a3 the line's first page.
constexpr uint32_t kHandlerTag = 1;
constexpr uint32_t kCbLine = 0;  // the reader's line before it goes to DRAM
// first: times (1), xor (2); second: plus (1), its 3 in the run.
constexpr uint32_t kOverlayTimes = 1;
constexpr uint32_t kOverlayXor = 2;
constexpr uint32_t kOverlayPlus = 1;

// What TAG hands the overlay it names.
struct TagArgument {
    uint32_t value;
};
