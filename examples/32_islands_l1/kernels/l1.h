// An L1 tensor moved across islands: the handler and the buffers its kernels share.
#pragma once

#include <stdint.h>

// MOVE: a0 input, a1 output (rank r's tile at page r), a2 the shift s, a3 the port:
// rank r's held tile into rank r + s's landing (mod the group), written out there.
constexpr uint32_t kHandlerMove = 1;

constexpr uint32_t kCbHeld = 0;     // the L1 tensor: each rank's tile, at one address
constexpr uint32_t kCbLanding = 1;  // where a moved tile lands, at one address
constexpr uint32_t kTileBytes = 4096;  // a float32 tile
