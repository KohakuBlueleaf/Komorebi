// Ray tracing on the resident workers: segments, buffers, and the layouts of the ray
// packets, the hit pages, the directory and the tree image that host and kernels share.
#pragma once

#include <stdint.h>

// LOAD: a0 tree image, a1 its bytes. Every core copies the tree into its tree buffer.
constexpr uint32_t kSegLoad = 1;
// TRACE: a0 directory, a1 packets, a2 hits, a3 mode (kModeNearest or kModeAny), a4
// profile. Each core walks the packets its directory entry names.
constexpr uint32_t kSegTrace = 2;
// ARITH: a0 directory, a1 input pages, a2 output pages, a4 profile. The slab and
// triangle tests on per-lane inputs (the arithmetic check).
constexpr uint32_t kSegArith = 3;
// RECIP: a0 directory, a2 output pages, a3 exponent field, a4 profile. Reciprocals of
// every mantissa with that exponent, generated on the card.
constexpr uint32_t kSegRecip = 4;
// TREELETS: as TRACE with the top tree loaded; a5 treelet images, a6 list entries, a7
// slot bytes, a8 slots, a9 policy. Top tree walk per packet, then treelet visits.
constexpr uint32_t kSegTreelets = 5;
constexpr uint32_t kPolicyRounds = 0;  // round r: the r-th treelet of every packet
constexpr uint32_t kPolicyOnce = 1;
// PROBE: as TRACE (one packet per page), a3 the form, a5 passes; each pass tests the
// packet against both children of every inner node of the loaded tree.
constexpr uint32_t kSegProbe = 6;
constexpr uint32_t kProbeImm = 0;   // the slab tests, box words as immediates
constexpr uint32_t kProbeData = 1;  // the same, box words loaded from Dst
constexpr uint32_t kProbeMask = 2;  // immediates, plus the entry test and lane read-back
constexpr uint32_t kProbeStack = 3;  // plus the stack and counters of the walk
constexpr uint32_t kProbeFast = 4;   // fused multiply-add, no exactness (report only)
constexpr uint32_t kProbeSync = 5;   // the entry test and lane read-back alone
// MATH: as ARITH; division and square root on per-lane inputs (their check).
constexpr uint32_t kSegMath = 7;

constexpr uint32_t kModeNearest = 0;
constexpr uint32_t kModeAny = 1;

// Buffers: input pages (packets or check inputs), the tree, directory scratch of the
// reader and the writer, output pages.
constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbTree = 1;
constexpr uint32_t kCbDirReader = 2;
constexpr uint32_t kCbDirWriter = 3;
constexpr uint32_t kCbOut = 16;
// Treelets: work pages (reader to compute), the end of the top walk (compute to
// reader), packet states, the lists region and the treelet slots (raw L1).
constexpr uint32_t kCbWork = 4;
constexpr uint32_t kCbSignal = 5;
constexpr uint32_t kCbState = 6;
constexpr uint32_t kCbLists = 7;
constexpr uint32_t kCbSlots = 8;
constexpr uint32_t kWorkBytes = 1024;
constexpr uint32_t kWorkPages = 4;
constexpr uint32_t kSignalBytes = 64;
constexpr uint32_t kDirReaderBytes = 128;  // directory entry, then the reader's counters
constexpr uint32_t kInPageBytes = 4096;
constexpr uint32_t kInPages = 4;
constexpr uint32_t kOutPageBytes = 4096;
constexpr uint32_t kOutPages = 4;
constexpr uint32_t kDirBytes = 64;
constexpr uint32_t kTreePageBytes = 4096;

// Directory entry of a core (one 64-byte page each): first page, pages.
constexpr uint32_t kDirFirst = 0;
constexpr uint32_t kDirCount = 1;

// A vector is 32 words, word l for lane l. Packet page: these vectors, then a header.
constexpr uint32_t kVec = 32;
constexpr uint32_t kPkO = 0;      // origin x, y, z
constexpr uint32_t kPkD = 3;      // direction x, y, z
constexpr uint32_t kPkInv = 6;    // 1 / direction as the reference clamps it
constexpr uint32_t kPkBest = 9;   // tmax, or -1.0 for a lane without a ray
constexpr uint32_t kPkHead = 10;  // header vector: word 0 the direction signs
constexpr uint32_t kPkVectors = 11;
constexpr uint32_t kPacketBytes = kPkVectors * kVec * 4;
// Direction signs: bit k set when the packet's rays point to negative k on the whole.
constexpr uint32_t kHeadSigns = 0;

// Hit page of a packet: t (tmax on a miss), scene triangle index (-1: none), u, v,
// box tests and triangle tests per ray; word 0 of the last vector the packet's cycles.
constexpr uint32_t kHitT = 0;
constexpr uint32_t kHitTri = 1;
constexpr uint32_t kHitU = 2;
constexpr uint32_t kHitV = 3;
constexpr uint32_t kHitNodes = 4;
constexpr uint32_t kHitLeaves = 5;
constexpr uint32_t kHitCycles = 6;
constexpr uint32_t kHitVectors = 7;
constexpr uint32_t kHitBytes = kHitVectors * kVec * 4;
constexpr uint32_t kHitPagePackets = 64;  // packets per DRAM page: host reads are per page

// Tree image (32-bit words): header, the reference tree's nodes (box min, max, child
// words), an order word per node, triangle records (v0, e1, e2, scene index).
constexpr uint32_t kTreeNodes = 0;     // header: node count
constexpr uint32_t kTreeNodeOff = 1;   // word offset of the nodes
constexpr uint32_t kTreeOrderOff = 2;  // word offset of the order words
constexpr uint32_t kTreeTriOff = 3;    // word offset of the triangle records
constexpr uint32_t kTreeHeadWords = 8;
constexpr uint32_t kNodeWords = 8;
constexpr uint32_t kTriWords = 10;
constexpr uint32_t kLeafBit = 0x80000000;
// Order word of an inner node: axis in bits 0-1; bit 2 set when child 1 is the nearer
// child for rays pointing to positive axis.
constexpr uint32_t kOrderFlip = 4;
// Top tree image: the nodes above the treelets, then one node per treelet (its root's
// box, child words kTreeletBit | treelet); a table of first page and pages per treelet.
constexpr uint32_t kTreeTreelets = 4;
constexpr uint32_t kTreeTableOff = 5;
constexpr uint32_t kTreeletBit = 0x40000000;

// Work page: slot address, treelet, packet count (kWorkEnd: no more work; kWorkNext:
// the batch's end, a rewalk follows), then 16-bit packet numbers.
constexpr uint32_t kWorkSlot = 0;
constexpr uint32_t kWorkTreelet = 1;
constexpr uint32_t kWorkCount = 2;
constexpr uint32_t kWorkIds = 4;
constexpr uint32_t kWorkMax = (kWorkBytes / 4 - kWorkIds) * 2;
constexpr uint32_t kWorkEnd = 0xFFFFFFFF;
constexpr uint32_t kWorkNext = 0xFFFFFFFE;

// Packet state in L1: Dst slots 0 to 14 (ray, results, counts), then a header vector
// (signs, cycles so far, kStDeferred: waiting for list room).
constexpr uint32_t kStHead = 15;
constexpr uint32_t kStCycles = 1;  // header word: the packet's cycles so far
constexpr uint32_t kStDeferred = 2;
constexpr uint32_t kStVectors = 16;
constexpr uint32_t kStWords = kStVectors * kVec;
// Lists region: header words, then per packet its first list entry and entry count,
// then the 16-bit list entries (treelets in walk order), then the reader's scratch.
constexpr uint32_t kLsDone = 0;  // work pages compute has finished
constexpr uint32_t kLsMore = 1;  // packets whose lists did not fit: another batch
constexpr uint32_t kLsHead = 16;

// Arithmetic check pages: the walk's inputs per lane (ray, 1 / d, box, triangle, best,
// current index) in; entry, exit, triangle values and updated results out.
constexpr uint32_t kArO = 0;
constexpr uint32_t kArD = 3;
constexpr uint32_t kArInv = 6;
constexpr uint32_t kArBoxMin = 9;
constexpr uint32_t kArBoxMax = 12;
constexpr uint32_t kArBest = 15;
constexpr uint32_t kArV0 = 16;
constexpr uint32_t kArE1 = 19;
constexpr uint32_t kArE2 = 22;
constexpr uint32_t kArTri = 25;
constexpr uint32_t kArCur = 26;
constexpr uint32_t kArInVectors = 27;
// Outputs: entry, min(exit, best), entry or +inf (the walk's stack value); det, 1 /
// det, u, v, t; best, index, u and v after the nearest-hit update.
constexpr uint32_t kArEntry = 0;
constexpr uint32_t kArExit = 1;
constexpr uint32_t kArStack = 2;
constexpr uint32_t kArDet = 3;
constexpr uint32_t kArDetInv = 4;
constexpr uint32_t kArU = 5;
constexpr uint32_t kArV = 6;
constexpr uint32_t kArT = 7;
constexpr uint32_t kArNewBest = 8;
constexpr uint32_t kArNewTri = 9;
constexpr uint32_t kArNewU = 10;
constexpr uint32_t kArNewV = 11;
constexpr uint32_t kArOutVectors = 12;
// Division and square root check pages: a, b, x in; a / b and sqrt(x) out.
constexpr uint32_t kMaA = 0;
constexpr uint32_t kMaB = 1;
constexpr uint32_t kMaX = 2;
constexpr uint32_t kMaInVectors = 3;
constexpr uint32_t kMaQ = 0;
constexpr uint32_t kMaS = 1;
constexpr uint32_t kMaOutVectors = 2;
// RECIP: an output page holds this many reciprocals.
constexpr uint32_t kRecipPerPage = 1024;

// Profile page of a core (a4, one 4096-byte page per core): the pack thread's counters
// from word 0, the reader's at kProfReader, the writer's at kProfWriter.
constexpr uint32_t kProfReader = 32;
constexpr uint32_t kProfWriter = 48;
constexpr uint32_t kRdWords = 16;  // reader counters after the directory entry
constexpr uint32_t kRdCounters = 16;
// Reader counters: packet pages (wait, read, total, pages), then the treelet schedule.
constexpr uint32_t kRdBin = 4;       // cycles binning packets by treelet
constexpr uint32_t kRdFetch = 5;     // cycles reading treelets
constexpr uint32_t kRdSlotWait = 6;  // cycles waiting for a slot to be free
constexpr uint32_t kRdWorkWait = 7;  // cycles waiting for a free work page
constexpr uint32_t kRdSignal = 8;    // cycles waiting for the top walk to end
constexpr uint32_t kRdReads = 9;     // treelet reads
constexpr uint32_t kRdPages = 10;    // treelet pages read
constexpr uint32_t kRdHits = 11;     // treelets found in a slot
constexpr uint32_t kRdItems = 12;    // work pages
constexpr uint32_t kRdVisits = 13;   // packet-treelet visits
constexpr uint32_t kRdRounds = 14;
constexpr uint32_t kRdGroups = 15;  // treelet groups (one per treelet and round)

// Every FIFO buffer moves one page at a time and each item fits its page; the tree
// buffer takes the whole image in one push and is sized to it by the host.
static_assert(kPacketBytes <= kInPageBytes && kArInVectors * kVec * 4 <= kInPageBytes);
static_assert(kHitBytes <= kOutPageBytes && kArOutVectors * kVec * 4 <= kOutPageBytes);
static_assert(kRecipPerPage * 4 <= kOutPageBytes && 32 * 4 <= kOutPageBytes);
static_assert(
    4 * (kRdWords + kRdCounters) <= kDirReaderBytes && kDirBytes <= 4 * kRdWords
);
static_assert(kWorkIds * 4 + kWorkMax * 2 <= kWorkBytes && kHitBytes % 16 == 0);
static_assert(
    kProfWriter + 4 <= kOutPageBytes / 4 && kProfReader + kRdCounters <= kProfWriter
);
