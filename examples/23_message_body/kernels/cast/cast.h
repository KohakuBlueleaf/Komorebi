// A cast of a body: its handler, buffer, command arguments, the fold every receiver
// makes of each body and the per-core area.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerCast = 1;
constexpr uint32_t kCbBody = 0;  // the broadcast's buffer: window blocks of one body
constexpr uint32_t kBodyBytes = 4096;
constexpr uint32_t kBodyWords = 1024;

// CAST: a0 casts, a1 bodies_address (a DRAM tensor of a2 bodies, one a page), a2
// bodies, a3 1: compute folds every word of every body it receives (0: it only pops).

// Per-core area at compute argument 2: the fold of the bodies received, then results.
constexpr uint32_t kResCasts = 1024;  // bodies received
constexpr uint32_t kResRun = 1025;    // the command's sequence
constexpr uint32_t kAreaWords = 1040;
static_assert(kResCasts == kBodyWords && kResRun < kAreaWords, "the area");

// The fold of body word `word` into `fold`: changes with every word and with the order.
inline uint32_t cast_fold(uint32_t fold, uint32_t word) {
    return ((fold << 1) | (fold >> 31)) ^ word;
}
