// Gate example: the handler, the three buffer pairs and the block size its kernels
// share.
#pragma once

#include <stdint.h>

// GATE: a0 a_address, a1 b_address, a2 out_address, a3 tiles (per core, a multiple of
// kBlock), a4 pattern word: out = silu(a) * b over the core's tiles.
constexpr uint32_t kHandlerGate = 1;

// Tiles a block holds: the tiles one pass of the vector unit keeps in its registers.
constexpr uint32_t kBlock = 8;

// The three pairs, each mover's half of a (bfloat16 pages), of b and of out; the
// kernels name them as km::BufferPair kA, kB and kOut.
constexpr uint32_t kAMoverA = 0;
constexpr uint32_t kAMoverB = 1;
constexpr uint32_t kBMoverA = 2;
constexpr uint32_t kBMoverB = 3;
constexpr uint32_t kOutMoverA = 16;
constexpr uint32_t kOutMoverB = 17;
