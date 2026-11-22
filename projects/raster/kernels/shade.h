// SHADE, between FILL and SHOW on a shaded frame: each core colours its tiles in
// place from materials and textures (reference/raster.py rasterize_shaded).
#pragma once

#include <stdint.h>

// SHADE: a0 directory, a1 lists (FILL's), a2 texture pages, a3 table, a4 profile (0:
// none) addresses. FILL's planes in, the colour planes written back.
constexpr uint32_t kHandlerShade = 5;

// Buffers: the reader's planes, the table, the pack thread's page requests, the
// reader's answers, the page slots (plain memory), the pack thread's own space.
constexpr uint32_t kCbShadePlanes = 14;
constexpr uint32_t kShadePlanePages = 10;  // two items of five planes
constexpr uint32_t kCbShadeTable = 15;
constexpr uint32_t kShadeTableBytes = 16384;
constexpr uint32_t kCbShadeAsk = 20;
constexpr uint32_t kShadeAskBytes = 1024;
constexpr uint32_t kShadeAskPages = 2;
constexpr uint32_t kCbShadeDone = 21;
constexpr uint32_t kShadeDoneBytes = 64;
constexpr uint32_t kShadeDonePages = 2;
constexpr uint32_t kCbShadeSlots = 22;
constexpr uint32_t kTexPageBytes = 4096;  // 32 x 32 texels, RGBA8, rows from the top
constexpr uint32_t kShadeSlots = 128;     // and one more page: the white texel
constexpr uint32_t kCbShadeWork = 23;
constexpr uint32_t kShadeWorkBytes = 8192;
constexpr uint32_t kShadeColourPages = 3;  // red, green, blue out (kCbOut)
// An item's pages in kCbOut: FILL's five (the first three used), so that the ring
// stays where FILL's items expect it and no item wraps (raster.h kOutPages).
constexpr uint32_t kShadeOutPages = 5;

// Table words: counts; TEXEL_LINEAR; a material: kind | (1 + albedo's level 0 entry, 0:
// none) << 8, base rgb; then an entry: first page, width | height << 16.
constexpr uint32_t kTabMaterials = 0;
constexpr uint32_t kTabEntries = 1;
constexpr uint32_t kTabPages = 2;
constexpr uint32_t kTabGamma = 16;
constexpr uint32_t kTabMaterial = 272;
constexpr uint32_t kMaterialWords = 4;
constexpr uint32_t kEntryWords = 2;
// kinds (reference/raster.py): masked and lit are shaded as textured in this step
constexpr uint32_t kKindFlat = 0;
constexpr uint32_t kKindTextured = 1;
constexpr uint32_t kKindMasked = 2;
constexpr uint32_t kKindLit = 3;
constexpr uint32_t kKindEmissive = 4;

// Record word (SETUP writes it on a shaded frame): material | albedo level << 16.
constexpr uint32_t kRecShade = 30;
constexpr uint32_t kShadeLevelShift = 16;

// An ask page: word 0 the pages to read (kShadeAskEnd: the command's end), then the
// texture pages, the page for slot i at word 1 + i.
constexpr uint32_t kShadeAskEnd = 0xFFFFFFFF;

// The profile page (64 words) of a core: the pack thread's counters.
constexpr uint32_t kShadeProfBytes = 256;
constexpr uint32_t kShadeProfTiles = 0;
constexpr uint32_t kShadeProfRecords = 1;
constexpr uint32_t kShadeProfRows = 2;    // record rows run
constexpr uint32_t kShadeProfPasses = 3;  // passes of rows, each its own pages
constexpr uint32_t kShadeProfPages = 4;   // texture pages read
constexpr uint32_t kShadeProfTexels = 5;  // texels fetched (pixels with a texture)
constexpr uint32_t kShadeProfWaitIn = 6;  // cycles: list pages and planes
constexpr uint32_t kShadeProfCopyIn = 7;  // planes into Dst
constexpr uint32_t kShadeProfDivide = 8;  // u, v, light per pixel
constexpr uint32_t kShadeProfRecord = 9;  // the records' rows
constexpr uint32_t kShadeProfScan = 10;   // pages found, texel addresses
constexpr uint32_t kShadeProfWaitPages = 11;
constexpr uint32_t kShadeProfGather = 12;  // texels and their linear values to Dst
constexpr uint32_t kShadeProfOut = 13;     // colour, planes to the writer
constexpr uint32_t kShadeProfTotal = 14;

// Gave-up places (the writer's word): a wait of the unpack or pack thread, a bad
// header or table, the reader's loop.
constexpr uint32_t kWhereShadeMail = 29;
constexpr uint32_t kWhereShadeFront = 30;
constexpr uint32_t kWhereShadeBad = 31;
constexpr uint32_t kWhereShadeReader = 32;
