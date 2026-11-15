// Chain example: the handler, buffers and command words its kernels share.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_group.hpp"

// a1 the input, a2 the output, a3 ADD's second input (SUM: tiles a group; TOTAL: the
// scratch), a4 the chain mode, a5 the factor, a6 tiles a core, a7 SUM's group, a8
// below.
constexpr uint32_t kHandlerScale = 1;  // output tile = factor x input tile
constexpr uint32_t kHandlerAdd = 2;    // input + second input's tile (a8 1: page 0)
constexpr uint32_t kHandlerSum = 3;    // the group's input tiles summed, at its root
constexpr uint32_t kHandlerTotal = 4;  // a copy's page 0 summed over the islands
constexpr uint32_t kChainMode = 1;

// One of this program's handlers (a chain command names any of them).
inline bool step_handler(uint32_t handler) {
    return handler >= kHandlerScale && handler <= kHandlerTotal;
}

// SUM's and TOTAL's group of core `me`: groups of a7 cores (0: one of every core).
inline km::Group step_group(const km_command_t& command, uint32_t me) {
    const uint32_t size = command.a7 != 0 ? command.a7 : km::count(command);
    const uint32_t local = me - km::first(command);
    return {km::first(command) + local / size * size, 1, size, local % size};
}

constexpr uint32_t kCbIn = 0;         // a tile read, worked on in place
constexpr uint32_t kCbOther = 1;      // ADD's second tile
constexpr uint32_t kCbArrive = 2;     // SUM's tree: slot r, round r's part
constexpr uint32_t kCbOut = 16;       // a tile for the writer
constexpr uint32_t kArrivePages = 7;  // a group of up to 128 cores
constexpr uint32_t kOutPages = 2;
constexpr uint32_t kTileBytes = 4096;  // 1024 int32 words
constexpr uint32_t kTileWords = 1024;
