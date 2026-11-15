// The card's two ceilings: every rank reading its share of a tensor on both movers
// (nothing kept), and the matrix unit multiplying resident tiles (nothing read).
#pragma once

#include <stdint.h>

// READ: a0 the tensor, a1 its pages, a2 pages a chunk; each rank's partition in chunks,
// even ones on the reader, odd on the writer, into one L1 (each waited for).
constexpr uint32_t kHandlerRead = 1;
// MATMUL: a0 passes. Each pass multiplies kCbA's bm x kb tiles by kCbB's kb x bn into
// Dst a subblock at a time (no pack): bm * bn * kb tile products a pass.
constexpr uint32_t kHandlerMatmul = 2;

constexpr uint32_t kCbRead = 0;   // the reader's chunk
constexpr uint32_t kCbWrite = 1;  // the writer's chunk
constexpr uint32_t kCbA = 2;      // bm x kb resident tiles
constexpr uint32_t kCbB = 3;      // kb x bn resident tiles
constexpr uint32_t kCbOut = 16;   // unused: the pack's format
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kChunkPages = 16;  // the chunk buffers' pages
// The matrix unit's resident block: 8 x 8 output tiles, K blocks of 8, subblock 2 x 4.
constexpr uint32_t kBm = 8;
constexpr uint32_t kBn = 8;
constexpr uint32_t kKb = 8;
constexpr uint32_t kSh = 2;
constexpr uint32_t kSw = 4;
