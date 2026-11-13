// Micro-op stream example: the handler, buffers and command arguments its kernels
// share.
#pragma once

#include <stdint.h>

// MAP: a0 input, a1 input tiles T from tile a4, a2 output, a3 output page of member 0.
constexpr uint32_t kHandlerMap = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
