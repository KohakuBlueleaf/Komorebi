// SHOW: the frame's colour as an image for a display. One command to the fill's cores
// after FILL: each core, for each item of its directory (as FILL), reads the tile's
// three colour planes from the frame and writes one page of the display tensor (page =
// tile number): 32 x 32 pixels row by row, 4 bytes a pixel R, G, B, 255. A band writes
// only its rows' bytes of the page, so the bands of a tile give the page of the whole
// tile.
#pragma once

#include <stdint.h>

// SHOW: a0 directory_address (FILL's), a1 display_address (pages of kShowPageBytes),
// a2 profile_address (pages of kShowProfBytes, one per core; 0: none); the block (a7,
// a8): kShowTableWords words, word k (1 to 255) the bits of the smallest float32 whose
// 8-bit value is k (reference/png.py to_srgb8), word 0 unused. The writer converts.
constexpr uint32_t kHandlerShow = 4;
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
