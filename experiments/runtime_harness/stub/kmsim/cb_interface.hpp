// Stub of a buffer's local interface, as both kernel APIs show it (pointers and limit
// as byte addresses on every side; the card's compute side counts 16-byte units).
#pragma once

#include <stdint.h>

#include "kmsim/sim.hpp"

using LocalCBInterface = kmsim::CbView;

// Page count and size, this side's pointers and the limit, and this thread's counts
// of pages pushed (producer) and popped (consumer). Read-only; the stub keeps them.
inline kmsim::CbView get_local_cb_interface(uint32_t cb) {
    return kmsim::cb_view(cb);
}
