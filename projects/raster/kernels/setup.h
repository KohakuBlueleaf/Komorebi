// The per-triangle passes: COUNT and SETUP over each core's partition of the mesh, 32
// triangles a block; handlers from 11, apart from the fill's (raster.h): one program.
#pragma once

#include <stdint.h>

// LANES: every block's records as the vector unit leaves them, for the exactness check.
// Each handler's command words: kernels/README.md.
constexpr uint32_t kHandlerLanes = 11;
// COUNT: per core, over its partition, the records of every tile and their rows in it
// (two arrays of 1,024 words), and the histograms kept for HIST.
constexpr uint32_t kHandlerCount = 12;
// SETUP: each kept record goes once to every item (a tile, or a band of its rows) it
// reaches, at that item's next slot.
constexpr uint32_t kHandlerSetup = 13;
// HIST (writer only): each core writes, per heavy tile, the 32 counts COUNT kept for it
// (kCbHist).
constexpr uint32_t kHandlerHist = 14;
// PLACE (writer only, bank required): directory pages, end slots and list headers from
// the middle pass's words in the items region.
constexpr uint32_t kHandlerPlace = 15;
// SUM (writer only): the counters' sums over the cores, 16 tiles a slice: each core's
// entries before it per tile into its slots page, the totals for the X280 thread.
constexpr uint32_t kHandlerSum = 16;
// HSUM (writer only): the same for HIST's histograms, one heavy tile at a time: each
// core's counts before it, summed over the groups, into its page at prefixes_address.
constexpr uint32_t kHandlerHsum = 17;
// bank (a8), every handler: 0, or 1 + the DRAM bank of the X280 thread's regions, whose
// small arrays are then bank addresses (pages contiguous); mesh and lists stay tensors.

// blocks_word (a2) with kRanges (COUNT, SETUP; bank): blocks [first_block, end_block)
// at kRangesAt + 64 x core of the camera's region; costs at kCostsAt + 128 x core.
constexpr uint32_t kRanges = 0x80000000;
constexpr uint32_t kRangesAt = 64;
constexpr uint32_t kCostsAt = 16384;
// COUNT's costs: the range's blocks, then the cost summed after j x blocks / 31, j 1-31
constexpr uint32_t kCostKnots = 31;
// A block's predicted cost: per block, kept triangle or piece, entry, clipped one; a
// fit of SETUP's cycles, whose ranges COUNT shares (README "what was sped up")
constexpr uint32_t kCostBlock = 4050;
constexpr uint32_t kCostKept = 400;
constexpr uint32_t kCostEntry = 40;
constexpr uint32_t kCostClipped = 9250;

// Buffers, apart from the fill's (raster.h) so that one program can hold both.
constexpr uint32_t kCbMesh = 4;    // mesh pages in, the camera's page first
constexpr uint32_t kCbSlots = 5;   // SETUP: the list slot of each item's next record
constexpr uint32_t kCbWork = 6;    // the pack thread's own space
constexpr uint32_t kCbRec = 17;    // record pages out
constexpr uint32_t kCbCount = 18;  // COUNT's counters
constexpr uint32_t kCbOut = 19;    // to the writer: the count, box words, then records
constexpr uint32_t kCbHist = 7;    // COUNT to HIST: per tile 16 uint16 counts of starts
                                   // in each 2-row group, then 16 of ends
constexpr uint32_t kCbItems = 8;   // SETUP's items page
constexpr uint32_t kMeshPages = 4;
constexpr uint32_t kRecPages = 2;
constexpr uint32_t kSlotsBytes = 8192;  // at most 2,048 items
constexpr uint32_t kItemsBytes = 8192;
constexpr uint32_t kItemRows = 4096;
// SETUP with kSetupPrefix in list_slots (a6): each core's slots page from the prefixes
// and the items' base words (first slot | 1 + heavy index << kBaseSlotBits).
constexpr uint32_t kSetupPrefix = 0x80000000;
constexpr uint32_t kBaseAt = 8192;
constexpr uint32_t kBaseSlotBits = 23;
constexpr uint32_t kHprefixAddrAt = 16384;
// Shaded SETUP: then the shade tensor (0: none), per mesh block 32 texel areas of the
// albedo, 32 material words, 32 of the normal, 32 of the specular; corners u, v, light.
constexpr uint32_t kShadeAddrAt = 16388;
static_assert(kShadeAddrAt == kHprefixAddrAt + 4);
constexpr uint32_t kShadeBlockBytes = 512;
// material | the albedo's, normal's, specular's levels << 16, << 20, << 24 (shade.h)
// | the shade block's bit 28 (lit: no tangent frame, shade.h kShadeNoFrame)
constexpr uint32_t kRecShadeWord = 30;
constexpr uint32_t kShadeWordKept = 0x10000000;
constexpr uint32_t kHprefixPageBytes = 32768;  // 128 bytes per heavy tile, 256 of them
constexpr uint32_t kSumsHeavyAt = 8192;  // in SUM's totals: HSUM's, 128 bytes per tile
// Then PLACE's words: frame words, per item and per fill core (README.md); each part on
// 64 bytes, as a DRAM read must start (one at 16 mod 64 returned the line's start).
constexpr uint32_t kFrameWordsAt = 16448;
static_assert(kShadeAddrAt + 4 <= kFrameWordsAt);
constexpr uint32_t kItemWordsAt = 16512;
constexpr uint32_t kItemWords = 2;
constexpr uint32_t kCoreWordsAt = 32896;
constexpr uint32_t kCoreWords = 4;
constexpr uint32_t kMaxFillCores = 248;
constexpr uint32_t kItemsRegionBytes = kCoreWordsAt + kCoreWords * 4 * kMaxFillCores;
// The list and directory words PLACE writes, as raster.h's (card_thread.py checks).
constexpr uint32_t kLayDirIds = 3;
constexpr uint32_t kLayDirFrame = 1000;
constexpr uint32_t kLayItemEmpty = 0x80000000;
constexpr uint32_t kLayRow0Shift = 16;
constexpr uint32_t kLayRow1Shift = 22;
constexpr uint32_t kLayMarkTile = 0xFFFFFFFE;
constexpr uint32_t kLayMarkEnd = 0xFFFFFFFF;
constexpr uint32_t kLayHdrTile = 1;
constexpr uint32_t kLayHdrCount = 2;
constexpr uint32_t kLayHdrRow0 = 3;
constexpr uint32_t kLayHdrRow1 = 4;
constexpr uint32_t kCountBytes = 8192;
constexpr uint32_t kWorkBytes = 12288;  // records of a block, of pieces, crossing corners
constexpr uint32_t kWorkCostsAt = 11776;  // then COUNT's costs (kCostsAt), 128 bytes
constexpr uint32_t kOutBytes = 4096;
constexpr uint32_t kHistBytes = 65536;  // kTiles x 64
constexpr uint32_t kHeavyBytes = 4096;  // HIST's heavy tiles: at most 1,023
constexpr uint32_t kDirPageBytes = 4096;
constexpr uint32_t kOutPages = 4;
constexpr uint32_t kOutEntries = 31;
constexpr uint32_t kOutLast = 0x80000000;  // in an out page's count: the last page
// COUNT: box words a page (after the count)
constexpr uint32_t kOutWords = 1023;
constexpr uint32_t kListPageBytes = 2048;
constexpr uint32_t kTiles = 1024;
constexpr uint32_t kTilesX = 32;
constexpr uint32_t kBlockSlots = 18;   // x, y, z of corners 0 to 2, then their colours
constexpr uint32_t kMeshBytes = 2304;  // kBlockSlots vectors of 32 words
constexpr uint32_t kMeshPageBytes = 2816;  // a mesh page: the mesh's block, the shade's
static_assert(kMeshPageBytes == kMeshBytes + kShadeBlockBytes);
constexpr uint32_t kRecBytes = 4096;  // 32 records of 32 words
static_assert(kMeshBytes == kBlockSlots * 32 * 4);
constexpr uint32_t kCamWords = 16;  // the scene's camera (reference/scenes.py)

// The image, as reference/scenes.py.
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
// out stride (HSUM: past its pages); PLACE without the thread's bank.
constexpr uint32_t kGaveBadHist = 8;
constexpr uint32_t kGaveBadHeavy = 9;
constexpr uint32_t kGaveBadPlace = 10;  // PLACE, SUM or HSUM without a bank (where: 0, the handler)

// Profile (profile_address, a9, not 0): 32 words per pass per core, at 128 x (handler -
// kHandlerCount) of its page; the threads leave wall-clock counters, the writer sends.
constexpr uint32_t kCbSProf = 9;
constexpr uint32_t kSProfBytes = 256;
constexpr uint32_t kSProfPageBytes = 4096;
constexpr uint32_t kSProfReader = 0;  // reserve wait, read, total, bytes, start
constexpr uint32_t kSProfPack = 8;  // mail wait, vector, scalar, total, kept, clipped,
                                    // entries, out wait
constexpr uint32_t kSProfWriter = 16;  // wait, write, total, bytes, start, pages
