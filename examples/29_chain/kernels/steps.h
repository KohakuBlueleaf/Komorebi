// Chain example: the handler, buffers and command words its kernels share.
#pragma once

#include <stdint.h>

// Every handler: a1 the input, a2 the output, a3 ADD's second input (SUM's input
// tiles), a4 the chain mode or 0, a5 SCALE's factor, a6 a group's output tiles.
constexpr uint32_t kHandlerScale = 1;  // output tile = factor x input tile
constexpr uint32_t kHandlerAdd = 2;    // output tile = input tile + second input's tile
constexpr uint32_t kHandlerSum = 3;    // the group's input tiles summed, at its root
constexpr uint32_t kChainMode = 1;

// One of this program's handlers (a chain command names any of them).
inline bool step_handler(uint32_t handler) {
    return handler >= kHandlerScale && handler <= kHandlerSum;
}

constexpr uint32_t kCbIn = 0;         // a tile read, worked on in place
constexpr uint32_t kCbOther = 1;      // ADD's second tile
constexpr uint32_t kCbArrive = 2;     // SUM's tree: slot r, round r's part
constexpr uint32_t kCbOut = 16;       // a tile for the writer
constexpr uint32_t kArrivePages = 7;  // a group of up to 128 cores
constexpr uint32_t kOutPages = 2;
constexpr uint32_t kTileBytes = 4096;  // 1024 int32 words
constexpr uint32_t kTileWords = 1024;
