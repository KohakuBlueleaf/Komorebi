// The block down a spanning tree of the islands and the results added up it, S cores an
// island, core s with core s of its neighbours, in the runtime's form (README.md).
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerTree = 1;   // a0 word, a1..a6 parents, a7 x / y, a8 the sum
constexpr uint32_t kHandlerStamp = 3;  // a0 0: the writer's ticks, 1: the home reader's
// a0: home [0, 5), phase [5, 12), S [12, 17)
constexpr uint32_t kSync = 1;  // first a page up the tree, untimed
constexpr uint32_t kDown = 2;  // x: the home's pages down, each island's into its copy (a7)
constexpr uint32_t kGo = 4;    // one page down in place of the block
constexpr uint32_t kUp = 8;    // y (a7) up, added at each island; the home writes a8
constexpr uint32_t kAck = 16;  // one page up in place of the block
constexpr uint32_t kNone = 31;

// An index where the class Moe holds bfloat16 or nothing (one format in the load);
// kCbSum the first from 16, where the runtime's start-up packs
constexpr uint32_t kCbOut = 0;      // n tiles: the block down (the home's, sent on)
constexpr uint32_t kCbDown = 1;     // n tiles: the parent's block
constexpr uint32_t kCbUp = 2;       // TREE_KIDS n tiles: a child's sum a slot
constexpr uint32_t kCbOwn = 3;      // n tiles: the island's own y slice (compute)
constexpr uint32_t kCbTiny = 6;     // one page
constexpr uint32_t kCbX = 7;        // n tiles: the home's x slice (its writer sends it)
constexpr uint32_t kCbSum = 17;     // n tiles: own + children (compute)
constexpr uint32_t kCbUpTiny = 18;  // TREE_KIDS pages: a child's sync or ack page
constexpr uint32_t kCbStamp = 22;   // 64 B
// one consumer a buffer: a dataflow core and compute each keep their own count of the
// pages taken, so two consumers of one buffer would leave it full
constexpr uint32_t kPageBytes = 2048;
constexpr uint32_t kPortDown = 1;
constexpr uint32_t kPortUp = 2;
constexpr uint32_t kPortSync = 3;
