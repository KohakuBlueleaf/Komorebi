// SHOW, after FILL: each core writes one page of the display tensor for each tile of
// its directory, 32 x 32 pixels of R, G, B, 255; a band writes only its rows' bytes.
#pragma once

#include <stdint.h>

#include "grid.h"

// SHOW: a0 directory_address (FILL's), a1 display_address, a2 profile_address (0:
// none); the block holds the thresholds of the 8-bit rule (reference/png.py to_srgb8).
constexpr uint32_t kHandlerShow = 4;
// a4 kShowRaw: plane 0 word for word (SUN's map); tile (row r, column c of 32 x 32) to
// page a5 + r a6 + c. a8 is the block's word: no grid, a tile's page is its number.
constexpr uint32_t kShowRaw = 1;
// The reader's colour planes to the writer: a tile's three planes an item.
constexpr uint32_t kCbShow = 13;
constexpr uint32_t kShowPlanes = 3;
constexpr uint32_t kShowPages = 6;  // two items: an item never wraps
constexpr uint32_t kShowPageBytes = 4096;
constexpr uint32_t kShowFirstPlane = 1;  // red, green, blue: the frame's planes 1 to 3
constexpr uint32_t kShowTableWords = 256;
constexpr uint32_t kShowAlpha = 0xFF000000;
// The writer's buckets (show_writer.hpp): the bits from kShowBucketBase (under the
// first threshold) to kShowOne (1.0, over the last) in ranges of 2^kShowBucketShift.
constexpr uint32_t kShowBucketBase = 0x35800000;
constexpr uint32_t kShowOne = 0x3F800000;
constexpr uint32_t kShowBucketShift = 16;
constexpr uint32_t kShowBuckets = (kShowOne - kShowBucketBase) >> kShowBucketShift;
// The writer's profile page (64 words): cycles converting, waiting for the planes,
// writing, in all; items, rows converted.
constexpr uint32_t kShowProfBytes = 256;
constexpr uint32_t kShowProfConvert = 0;
constexpr uint32_t kShowProfWait = 1;
constexpr uint32_t kShowProfWrite = 2;
constexpr uint32_t kShowProfTotal = 3;
constexpr uint32_t kShowProfItems = 4;
constexpr uint32_t kShowProfRows = 5;
// A frame table or block the writer cannot use: its gave-up word, where.
constexpr uint32_t kWhereShowBad = 28;
