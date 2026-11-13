// Small triangles as vector lanes, on one core: one command fills one tile, its larger
// triangles row by row (fill.hpp), its small ones 32 at a time, one per lane.
#pragma once

#include <stdint.h>

// TILE: a0 input address, a1 first input page, a2 input pages, a3 output address, a4
// first output page. Input: the header page, the row records (32 a page), lane pages.
constexpr uint32_t kHandlerTile = 1;
constexpr uint32_t kLanesCbIn = 0;
constexpr uint32_t kLanesCbOut = 16;
constexpr uint32_t kLanesPageBytes = 4096;
constexpr uint32_t kLanesInPages = 4;
constexpr uint32_t kLanesOutPages = 6;  // the five planes (raster.h), then the profile

// Header page words: the tile (ty * 32 + tx), the row records, the lane pages.
constexpr uint32_t kHeadTile = 0;
constexpr uint32_t kHeadRowRecords = 1;
constexpr uint32_t kHeadLanePages = 2;
constexpr uint32_t kMaxRecords = 65536;
constexpr uint32_t kTileCount = 1024;

// Lane page: word s of lane l at 64 (s >> 1) + 2 l + (s & 1), the order of the Dst
// window. Words 0 to 23 are the record's (raster.h); then, as float32 unless named:
constexpr uint32_t kLanePx = 24;      // px of the box's first column in the tile
constexpr uint32_t kLanePxEnd = 25;   // px past its last column
constexpr uint32_t kLanePy = 26;      // py of its first row in the tile
constexpr uint32_t kLanePyEnd = 27;   // py past its last row
constexpr uint32_t kLaneTri = 28;     // the triangle number
constexpr uint32_t kLaneCorner = 29;  // first column | first row << 8 (int)
constexpr uint32_t kLaneWords = 30;
// Page word 1023: the page's offsets, columns | rows << 4 (each 1 to 4).
constexpr uint32_t kLaneShape = 1023;
constexpr uint32_t kMaxSide = 4;
constexpr uint32_t kMaxOffsets = 16;

// Dst vectors: the lane words from 160, w (0 outside) of offset k at 192 + k, its
// colours at 208 + 3 k; below 160 the tile's planes (fill.hpp).
constexpr uint32_t kVecLaneIn = 160;
constexpr uint32_t kVecLaneInside = 190;  // per lane: bit k, inside at offset k
constexpr uint32_t kVecLaneW = 192;
constexpr uint32_t kVecLaneColour = 208;

// Profile page (the sixth output page): the pack thread's cycles and counts.
constexpr uint32_t kProfTotal = 0;
constexpr uint32_t kProfInit = 1;
constexpr uint32_t kProfRows = 2;        // row records: their set-up and rows
constexpr uint32_t kProfCopyIn = 3;      // lane pages into Dst
constexpr uint32_t kProfVector = 4;      // the lanes on the vector unit
constexpr uint32_t kProfResolve = 5;     // fragments against the planes, scalar
constexpr uint32_t kProfOut = 6;         // planes out
constexpr uint32_t kProfWait = 7;        // waiting for input pages
constexpr uint32_t kProfRowRecords = 8;  // counts from here
constexpr uint32_t kProfRowCount = 9;
constexpr uint32_t kProfLanePages = 10;
constexpr uint32_t kProfOffsets = 11;
constexpr uint32_t kProfFragments = 12;  // inside fragments found
constexpr uint32_t kProfWins = 13;
// 64 accesses timed once a tile: dependent window loads, independent ones, window
// stores, dependent L1 loads.
constexpr uint32_t kProfWindowChain = 14;
constexpr uint32_t kProfWindowLoads = 15;
constexpr uint32_t kProfWindowStores = 16;
constexpr uint32_t kProfL1Chain = 17;
constexpr uint32_t kProfSetup = 18;      // row records' set-up (inside kProfRows)
constexpr uint32_t kProfBitsWrong = 19;  // broadcast build: lanes that differ, of 2,048
constexpr uint32_t kProfBroadcastWrite = 20;  // of kProfSetup: the RISC-V's copies
constexpr uint32_t kProfWords = 21;

// Give-ups (raster.h's word and codes): a header or a lane page's shape out of range.
constexpr uint32_t kWhereBadLanes = 30;
