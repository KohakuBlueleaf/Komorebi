// The two programs' handlers and buffers, as in example 16: sum adds a core's
// partition of a1 tiles from tile a6 of the input (a0); widen writes them as float32.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerSum = 1;
constexpr uint32_t kHandlerWiden = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbWideIn = 1;    // widen's input: another index, other pages
constexpr uint32_t kCbWideOut = 16;  // widen's output: float32 at sum's index
constexpr uint32_t kBlock = 8;       // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
