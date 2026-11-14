// The card's two ceilings for the dense project's bounds: every rank reading its share
// of a tensor from the banks on both movers (nothing kept), and the matrix unit
// multiplying resident tiles (nothing read).
#pragma once

#include <stdint.h>

// READ: a0 the tensor (bank address, tile pages), a1 its pages, a2 pages a chunk. Each
// rank's partition of the pages in chunks, even chunks on the reader, odd ones on the
// writer, each into the same L1 (a chunk's reads waited for before the next).
constexpr uint32_t kHandlerRead = 1;
// MATMUL: a0 passes. Each pass multiplies kCbA's bm x kb tiles by kCbB's kb x bn into
// Dst a subblock at a time (no pack): bm * bn * kb tile products a pass.
constexpr uint32_t kHandlerMatmul = 2;

constexpr uint32_t kCbRead = 0;    // the reader's chunk
constexpr uint32_t kCbWrite = 1;   // the writer's chunk
constexpr uint32_t kCbA = 2;       // bm x kb resident tiles
constexpr uint32_t kCbB = 3;       // kb x bn resident tiles
constexpr uint32_t kCbOut = 16;    // unused: the pack's format
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kChunkPages = 16;  // the chunk buffers' pages
// The matrix unit's resident block: 8 x 8 output tiles, K blocks of 8, subblock 2 x 4.
constexpr uint32_t kBm = 8;
constexpr uint32_t kBn = 8;
constexpr uint32_t kKb = 8;
constexpr uint32_t kSh = 2;
constexpr uint32_t kSw = 4;
