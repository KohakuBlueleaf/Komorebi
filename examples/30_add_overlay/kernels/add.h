// What the kernels, the code added to the live run and Python agree on: the one
// handler of the load, the buffers, the words of each command.
#pragma once

#include <stdint.h>

// RUN, the one handler the load has: a0 an overlay to call (0: none), a1 a seed, a2
// the lines' tensor (its word), a3 the overlay's steps.
constexpr uint32_t kHandlerRun = 1;
constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbLine = 2;  // the reader's line before it goes to DRAM
constexpr uint32_t kLineWords = 16;
constexpr uint32_t kTileValues = 1024;
// bfloat16 bits of 1 to 7: input value j of tile t on core c is entry (c + t + j) % 7.
constexpr uint32_t kSmallValues = 7;

// What RUN hands the overlay it names: the word to mix and the loop's steps.
struct MixArgument {
    uint32_t value;
    uint32_t steps;
};

// SCALE, the handler added while the run is live, takes the command itself: a3 tiles
// a core, a4 the scalar (float32 bits), a6 the tiles' tensor (its word).
