// Which mover owns block k of a stream, from one host-chosen word. Free of tt-metal
// headers: the movers and compute both read it.
#pragma once

#include <stdint.h>

namespace movement {

    // Mover A is the reader role's CPU (RISCV_0, NoC 0); mover B the writer role's
    // (RISCV_1, NoC 1).
    constexpr uint32_t kMoverA = 0;
    constexpr uint32_t kMoverB = 1;
    constexpr uint32_t kPatternLengthMax = 24;

    // A pattern word: its length (1 to 24) in the low byte, then one bit per position,
    // set when mover B owns the block at that position ("AAAB": length 4, bits 0b1000).
    struct Pattern {
        uint32_t length;
        uint32_t owners;
    };

    inline Pattern pattern_of(uint32_t word) {
        return {word & 0xFFu, word >> 8};
    }

    inline bool valid(const Pattern& pattern) {
        return pattern.length >= 1 && pattern.length <= kPatternLengthMax;
    }

    inline uint32_t owner(const Pattern& pattern, uint32_t block) {
        return (pattern.owners >> (block % pattern.length)) & 1u;
    }

}  // namespace movement
