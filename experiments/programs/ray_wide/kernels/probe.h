// The wide-node probe's handler, forms, record layouts and result words (C++ and the
// host read this file).
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerWideProbe = 1;

// Forms of the probe, each adding a step to the one before.
constexpr uint32_t kFormLoad = 0;   // child boxes into the lanes
constexpr uint32_t kFormTest = 1;   // and the slab test, waited for
constexpr uint32_t kFormOut = 2;    // and the entry distances to the core
constexpr uint32_t kFormStack = 3;  // and the core's stack step
constexpr uint32_t kFormCheck = 4;  // as kFormOut, the distances kept for the host
constexpr uint32_t kFormRay = 5;    // rays into their lanes alone
constexpr uint32_t kForms = 6;

// How child boxes reach the lanes: the core copies each group's record, the core
// gathers every lane's word at once, or unpack and math copy a node tile (one ray).
constexpr uint32_t kMethodWindow = 0;
constexpr uint32_t kMethodGather = 1;
constexpr uint32_t kMethodUnpack = 2;
constexpr uint32_t kCbNodes = 0;  // node tiles for the unpacker, float32
constexpr uint32_t kTileWords = 1024;
constexpr uint32_t kTilesMax = 32;

// An operation: per group (node record word, ray, children). A node record: the six
// box words, lanes / groups children each. A ray: origin, reciprocal, best, a spare.
constexpr uint32_t kGroupsMax = 4;
constexpr uint32_t kOpWords = 3 * kGroupsMax;
constexpr uint32_t kRayWords = 8;
constexpr uint32_t kLanes = 32;
constexpr uint32_t kStackDepth = 64;

// Result words of a core.
constexpr uint32_t kResultCycles = 0;
constexpr uint32_t kResultOps = 1;
constexpr uint32_t kResultVisits = 2;
constexpr uint32_t kResultEntered = 3;
constexpr uint32_t kResultPushes = 4;
constexpr uint32_t kResultRayLoads = 5;
constexpr uint32_t kResultRun = 6;
constexpr uint32_t kResultError = 7;
constexpr uint32_t kResultWords = 16;
