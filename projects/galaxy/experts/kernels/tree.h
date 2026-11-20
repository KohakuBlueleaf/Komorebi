// The block down a tree of the islands and the islands' results summed up it, in the
// runtime's form: the command words and buffers (README.md, "The tree kernels").
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

// Indices where Moe and the attention's classes hold bf16 or nothing (26-31 are the
// runtime's); kCbSum the first from 16 (the start-up packs it)
constexpr uint32_t kCbOut = 0;      // n tiles: the block down (the home's, sent on)
constexpr uint32_t kCbDown = 1;     // n tiles: the parent's block
constexpr uint32_t kCbSum = 18;     // n tiles: own + children (compute)
constexpr uint32_t kCbUpTiny = 20;  // TREE_KIDS pages: a child's sync or ack page
constexpr uint32_t kCbUp = 21;      // TREE_KIDS n tiles: a child's sum a slot
constexpr uint32_t kCbOwn = 22;     // n tiles: the island's own y slice (compute)
constexpr uint32_t kCbStamp = 23;   // 64 B
constexpr uint32_t kCbTiny = 24;    // one page
constexpr uint32_t kCbX = 25;       // n tiles: the home's x slice (its writer sends it)
// one consumer a buffer: a dataflow core and compute each keep their own count of the
// pages taken, so two consumers of one buffer would leave it full
constexpr uint32_t kPageBytes = 2048;
constexpr uint32_t kPortDown = 1;
constexpr uint32_t kPortUp = 2;
constexpr uint32_t kPortSync = 3;
