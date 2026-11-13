// The call between the persistent kernel and an overlay: the entry at the slot's
// first word takes these arguments and returns the overlay's result word.
#pragma once

#include <stdint.h>

namespace ov {

    struct Args {
        uint32_t arg;               // the command's argument
        uint32_t (*mix)(uint32_t);  // a function of the persistent kernel
    };

    using Entry = uint32_t (*)(Args*);

    // The persistent kernel's mix(), mirrored by the host's reference.
    inline uint32_t mix(uint32_t x) {
        return ((x << 7) | (x >> 25)) ^ 0x5A5A5A5Au;
    }

}  // namespace ov

// Counts the overlay calls; defined by the persistent kernel, incremented by every
// overlay through the persistent image's symbol (its address in the core's own memory).
extern "C" volatile uint32_t ov_calls;
