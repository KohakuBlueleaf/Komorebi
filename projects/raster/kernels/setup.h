// The rasterizer's per-triangle passes on the cores: COUNT and SETUP over each core's
// share of the mesh, 32 triangles per block as the vector unit's lanes.
#pragma once

#include <stdint.h>

// LANES: a0 mesh, a1 output (a page of 32 records per block), a2 blocks, a3 camera,
// a4 and a6 the window of blocks written (first, count: page b - a4 of the output),
// a5 cores sharing the blocks. Every block's records as the vector unit leaves them
// (words 24 to 31 still raw, see kRawX0). For the exactness check.
constexpr uint32_t kSegLanes = 1;
// COUNT: a0 mesh, a1 counters (a page of kCountBytes per core), a2 blocks, a3 camera,
// a5 cores. Each core counts, over its share, the records of every tile and their rows
// in it (two arrays of 1,024 words).
constexpr uint32_t kSegCount = 2;
// SETUP: a0 mesh, a1 lists (2 KB pages of 16 slots), a2 blocks, a3 camera, a4 slots
// (a page of kSlotsBytes per core: the list slot of each tile's next record), a5
// cores, a6 the lists' slots (a slot past them is not written: a give-up). Each kept
// record goes once to every tile it touches, at that tile's next slot.
constexpr uint32_t kSegSetup = 3;

// Buffers: mesh pages in (the camera's page first), record pages out; the tiles'
// slots (SETUP), the counters (COUNT), the pack thread's own space, entries out
// (SETUP: in each page slot 0 holds the count and the target slots, then records).
constexpr uint32_t kCbMesh = 0;
constexpr uint32_t kCbSlots = 1;
constexpr uint32_t kCbWork = 2;
constexpr uint32_t kCbRec = 16;
constexpr uint32_t kCbCount = 17;
constexpr uint32_t kCbOut = 18;
constexpr uint32_t kMeshPages = 4;
constexpr uint32_t kRecPages = 2;
constexpr uint32_t kSlotsBytes = 4096;
constexpr uint32_t kCountBytes = 8192;
constexpr uint32_t kWorkBytes = 12288;  // records of a block, of pieces, crossing corners
constexpr uint32_t kOutBytes = 4096;
constexpr uint32_t kOutPages = 4;
constexpr uint32_t kOutEntries = 31;
constexpr uint32_t kOutLast = 0x80000000;  // in an out page's count: the last page
constexpr uint32_t kListPageBytes = 2048;
constexpr uint32_t kTiles = 1024;
constexpr uint32_t kTilesX = 32;
constexpr uint32_t kBlockSlots = 18;   // x, y, z of corners 0 to 2, then their colours
constexpr uint32_t kMeshBytes = 2304;  // kBlockSlots vectors of 32 words
constexpr uint32_t kRecBytes = 4096;   // 32 records of 32 words
static_assert(kMeshBytes == kBlockSlots * 32 * 4);
constexpr uint32_t kCamWords = 16;  // the scene's camera (ref/scenes.py)

// The image, as ref/scenes.py.
constexpr uint32_t kWidth = 1024;
constexpr uint32_t kHeight = 1024;

// Record words 24 to 31 as the vector unit leaves them: the box's first and last
// column and row each as float32 (1.5 x 2^23 + n), the doubled area before the
// corners are ordered, and the corners' camera distances.
constexpr uint32_t kRawX0 = 24;
constexpr uint32_t kRawX1 = 25;
constexpr uint32_t kRawY0 = 26;
constexpr uint32_t kRawY1 = 27;
constexpr uint32_t kRawArea = 28;
constexpr uint32_t kRawZ = 29;
constexpr uint32_t kMagic = 0x4B400000;  // float32(1.5 x 2^23)

// Waits on the other compute threads give up after this many cycles (0.8 s), with
// kGaveWait and where (1: unpack, a mesh page; 2: unpack, the pack thread's hand-back;
// 3: pack, the next page; 4: unpack, the slots page) in the writer's gave-up word. A
// record for a slot past the lists gives up with kGaveBadSlot (where: the tile).
constexpr uint32_t kWaitCycles = 0x40000000;
constexpr uint32_t kGaveWait = 4;
constexpr uint32_t kGaveBadSlot = 5;
// A kept record whose box leaves the image, or a clipped polygon that is not 3 or 4
// corners, gives up with these codes (where: the low bits of the triangle number).
constexpr uint32_t kGaveBadBox = 6;
constexpr uint32_t kGaveBadClip = 7;
