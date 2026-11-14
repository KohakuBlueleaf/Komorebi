// Two programs example: each program's handlers, the buffers both declare (one shared
// buffer each when combined) and the argument words of a command.
#pragma once

#include <stdint.h>

// sum: SUM adds the core's partition of a1 tiles from tile a6 of the input (a0) into
// one tile, written to page a3 + rank of the output (a2). EMPTY does nothing.
constexpr uint32_t kHandlerSum = 1;
// scale: SCALE doubles the core's partition of a1 tiles from tile a6 of the input (a0)
// into pages a3 + their places of the output (a2). EMPTY does nothing.
constexpr uint32_t kHandlerScale = 1;
constexpr uint32_t kHandlerEmpty = 2;
// widen: WIDEN reads the core's partition of a1 tiles from tile a6 of the input (a0)
// and writes each as float32 to page a3 + its place of the output (a2).
constexpr uint32_t kHandlerWiden = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbWideIn = 1;    // widen's input: another index, other pages
constexpr uint32_t kCbWideOut = 16;  // widen's output: float32 at sum's index
constexpr uint32_t kBlock = 8;       // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
