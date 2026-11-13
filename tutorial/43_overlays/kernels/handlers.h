// The numbers that the kernels, the overlays and Python agree on.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerApply = 1;  // a0 the overlay, a1 and a2 the numbers
constexpr uint32_t kOverlayAdd = 1;
constexpr uint32_t kOverlayMultiply = 2;

// The argument both overlays take.
struct Pair {
    uint32_t left;
    uint32_t right;
};
