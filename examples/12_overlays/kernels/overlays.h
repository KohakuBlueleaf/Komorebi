// What the kernels, the overlays and Python agree on: the handler, the two overlays'
// numbers, the buffers and each overlay's argument.
#pragma once

#include <stdint.h>

// RUN: a0 the overlay, a1 a seed, a2 the lines' tensor, a3 tiles, a4 the scalar
// (float32 bits), a5 steps, a6 the tiles' tensor (each its word).
constexpr uint32_t kHandlerRun = 1;
constexpr uint32_t kOverlayMix = 1;  // a reader part: a mixing loop on a word
constexpr uint32_t kOverlayScale = 2;  // reader, compute, writer parts: tiles times a scalar
constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbLine = 2;  // the reader's line before it goes to DRAM
constexpr uint32_t kLineWords = 16;
constexpr uint32_t kTileValues = 1024;

// Mix's argument: the word to mix and the steps of the loop.
struct MixArgument {
    uint32_t value;
    uint32_t steps;
};

// Scale's argument: the core (it picks the input values), tiles, the scalar.
struct ScaleArgument {
    uint32_t core;
    uint32_t tiles;
    uint32_t scalar;
};

// Scale's writer argument: the core (it picks the pages), tiles, the pages' tensor
// (its word), the command's slot and sequence (for the page calls).
struct DrainArgument {
    uint32_t core;
    uint32_t tiles;
    uint32_t tensor;
    uint32_t slot;
    uint32_t sequence;
};

// bfloat16 bits of 1 to 7: input value j of tile t on core c is entry (c + t + j) % 7.
constexpr uint32_t kSmallValues = 7;
