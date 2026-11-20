// Bit counts the device code can call on every chip.
#pragma once

#include <stdint.h>

namespace km {

    // The set bits of `x`. On Wormhole (no count instruction) the builtin is a call to
    // libgcc's __popcountsi2, which an overlay's part cannot reach.
    constexpr uint32_t popcount(uint32_t x) {
#if defined(ARCH_WORMHOLE)
        uint32_t count = 0;
        for (; x != 0; x &= x - 1u) {
            count++;
        }
        return count;
#else
        return static_cast<uint32_t>(__builtin_popcount(x));
#endif
    }

}  // namespace km
