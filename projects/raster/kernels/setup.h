// The per-triangle passes: COUNT and SETUP over each core's share of the mesh, 32
// triangles a block; segments from 11, apart from the fill's (raster.h): one program.
#pragma once

#include <stdint.h>

// LANES: every block's records as the vector unit leaves them, for the exactness check.
// Each segment's record words: kernels/README.md.
constexpr uint32_t kSegLanes = 11;
// COUNT: per core, over its share, the records of every tile and their rows in it (two
// arrays of 1,024 words), and the histograms kept for HIST.
constexpr uint32_t kSegCount = 12;
// SETUP: each kept record goes once to every item (a tile, or a band of its rows) it
// reaches, at that item's next slot.
constexpr uint32_t kSegSetup = 13;
// HIST (writer only): each core writes, per heavy tile, the 32 counts COUNT kept for it
// (kCbHist).
constexpr uint32_t kSegHist = 14;
// PLACE (writer only, a8 required): core c copies its directory page (c < a6) and its
// share of the marks (a list slot's 128 bytes, then its slot number) to their slots.
constexpr uint32_t kSegPlace = 15;
// a8, every segment: 0, or 1 + the DRAM bank of the X280 thread's regions, whose small
// arrays are then bank addresses (pages contiguous); mesh and lists stay tensors.

// Buffers, apart from the fill's (raster.h) so that one program can hold both.
constexpr uint32_t kCbMesh = 4;    // mesh pages in, the camera's page first
constexpr uint32_t kCbSlots = 5;   // SETUP: the list slot of each item's next record
constexpr uint32_t kCbWork = 6;    // the pack thread's own space
constexpr uint32_t kCbRec = 17;    // record pages out
constexpr uint32_t kCbCount = 18;  // COUNT's counters
constexpr uint32_t kCbOut = 19;  // entries out: slot 0 the count and slots, then records
constexpr uint32_t kCbHist = 7;   // COUNT to HIST: per tile 16 uint16 counts of starts
                                  // in each 2-row group, then 16 of ends
constexpr uint32_t kCbItems = 8;  // SETUP's items page
constexpr uint32_t kMeshPages = 4;
constexpr uint32_t kRecPages = 2;
constexpr uint32_t kSlotsBytes = 8192;  // at most 2,048 items
constexpr uint32_t kItemsBytes = 8192;
constexpr uint32_t kItemRows = 4096;
constexpr uint32_t kCountBytes = 8192;
constexpr uint32_t kWorkBytes = 12288;  // records of a block, of pieces, crossing corners
constexpr uint32_t kOutBytes = 4096;
constexpr uint32_t kHistBytes = 65536;  // kTiles x 64
constexpr uint32_t kHeavyBytes = 4096;  // HIST's heavy tiles: at most 1,023
constexpr uint32_t kDirPageBytes = 4096;
// A list slot and its number, padded to 64 bytes: a DRAM read must start on a
// 64-byte boundary (one at 16 mod 64 returned the line's start).
constexpr uint32_t kMarkBytes = 192;
constexpr uint32_t kMarkBatch = 21;  // marks per copy through one out page
static_assert(kMarkBatch * kMarkBytes <= kOutBytes);
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

// Record words 24 to 31 as the vector unit leaves them: the box's columns and rows as
// float32 (1.5 x 2^23 + n), the doubled area unordered, the corners' camera distances.
constexpr uint32_t kRawX0 = 24;
constexpr uint32_t kRawX1 = 25;
constexpr uint32_t kRawY0 = 26;
constexpr uint32_t kRawY1 = 27;
constexpr uint32_t kRawArea = 28;
constexpr uint32_t kRawZ = 29;
constexpr uint32_t kMagic = 0x4B400000;  // float32(1.5 x 2^23)

// Waits give up after kWaitCycles (0.8 s) with kGaveWait, where: 1, 2 unpack's mesh
// page, hand-back; 3 pack's next page; 4 unpack's slots or items page.
constexpr uint32_t kWaitCycles = 0x40000000;
constexpr uint32_t kGaveWait = 4;
constexpr uint32_t kGaveBadSlot = 5;  // a slot past the lists or the slots page; where: tile
// A kept record whose box leaves the image, or a clipped polygon that is not 3 or 4
// corners, gives up with these codes (where: the low bits of the triangle number).
constexpr uint32_t kGaveBadBox = 6;
constexpr uint32_t kGaveBadClip = 7;
// A histogram count past 65,535 (where: the tile); HIST's heavy tiles past its page or
// out stride; PLACE without the thread's bank.
constexpr uint32_t kGaveBadHist = 8;
constexpr uint32_t kGaveBadHeavy = 9;
constexpr uint32_t kGaveBadPlace = 10;  // PLACE without a8

// Profile (a9 not 0): 32 words per pass per core, at 128 x (segment - kSegCount) of its
// page; the threads leave wall-clock counters in kCbSProf and the writer sends them.
constexpr uint32_t kCbSProf = 9;
constexpr uint32_t kSProfBytes = 256;
constexpr uint32_t kSProfPageBytes = 4096;
constexpr uint32_t kSProfReader = 0;  // reserve wait, read, total, bytes, start
constexpr uint32_t kSProfPack = 8;  // mail wait, vector, scalar, total, kept, clipped,
                                    // entries, out wait
constexpr uint32_t kSProfWriter = 16;  // wait, write, total, bytes, start, pages
