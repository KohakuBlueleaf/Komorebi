// The gate's hand kernel as a program of its own (kernel_handler.py --hand): the names
// the class's generated header gives its hand files, at the class's values.
#pragma once

#include <stdint.h>

// GATE: a0 a, a1 b, a2 out, a3 tiles, a4 pattern.
constexpr uint32_t kHandlerGate = 1;

// The class's BLOCK: tiles a block, one acquire of a float32 Dst's half.
constexpr uint32_t kConstBlock = 4;

// The three declared pairs' halves, at the class's indices, and the pages of each half
// (each kernel names them as km::BufferPair kPairAIn, kPairBIn and kPairResult).
constexpr uint32_t kPairAInMoverA = 4;
constexpr uint32_t kPairAInMoverB = 5;
constexpr uint32_t kPairBInMoverA = 6;
constexpr uint32_t kPairBInMoverB = 7;
constexpr uint32_t kPairResultMoverA = 18;
constexpr uint32_t kPairResultMoverB = 19;
constexpr uint32_t kPairAInPages = 16;
constexpr uint32_t kPairBInPages = 16;
constexpr uint32_t kPairResultPages = 16;
