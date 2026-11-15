// Casters probe: handlers, buffers and arguments its kernels share, and the word every
// core expects at each place of each block.
#pragma once

#include <stdint.h>

#include "counted.hpp"

// CAST: a0 results_address, a1 blocks, a2 first (the first block's index over the
// jobs), a3 result_page (the page of the group's first core).
constexpr uint32_t kHandlerCast = 1;

constexpr uint32_t kCbBlocks = 0;      // the broadcast's buffer: B blocks of P pages
constexpr uint32_t kCbResult = 1;      // a core's result row, written to DRAM
constexpr uint32_t kResultWords = 16;  // wrong, first block, first place, blocks, ...
constexpr uint32_t kNone = 0xFFFFFFFFu;

// Reader arguments: the broadcast's words. Writer: none. Compute: included, B, P.
constexpr uint32_t kArgBroadcast = 0;

// The word at `place` (its index in the block) of block `block`.
inline uint32_t expected(uint32_t block, uint32_t place) {
    uint32_t hash = block * 0x9E3779B1u ^ place * 0x85EBCA77u;
    hash ^= hash >> 15;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12;
    return hash;
}
