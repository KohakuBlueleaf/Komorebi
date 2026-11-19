// The Tensix dispatcher's arguments, its block in L1 (the host ring) and its area of
// the host channel; the host reads the same constants (komorebi.session.tensix).
#pragma once

#include <stdint.h>

// Runtime arguments of the dispatcher kernel.
constexpr uint32_t kTdControl = 0;     // control block address (every core's)
constexpr uint32_t kTdRanks = 1;       // the workers: ranks 0 to ranks - 1
constexpr uint32_t kTdCores = 2;       // DRAM address of the core table
constexpr uint32_t kTdBlock = 3;       // L1 address of the block below, 64-byte aligned
constexpr uint32_t kTdHostDepth = 4;   // host ring lines, a power of two, at most 64
constexpr uint32_t kTdEventDepth = 5;  // event ring, a power of two, at most 128
constexpr uint32_t kTdArea = 6;        // channel offset of the area below
constexpr uint32_t kTdDepth = 7;       // slots used, 1 to KM_RING_SLOTS
constexpr uint32_t kTdDeadline = 8;    // per command, submit to ack, 1024-tick units
constexpr uint32_t kTdBootLimit = 9;   // start-up wait, 1024-tick units
constexpr uint32_t kTdIdleLimit = 10;  // nothing taken nor busy this long: 0 never
constexpr uint32_t kTdGroups = 11;     // start-up groups: READY words to wait for
constexpr uint32_t kTdRects = 12;      // multicast rectangles (0: unicast to each)
// Rectangle r at kTdRect + 3 r: NoC 0 start and end (x | y << 16), receivers.
constexpr uint32_t kTdRect = 13;
constexpr uint32_t kTdRectsMax = 4;
constexpr uint32_t kTdLoopback = 25;  // 1: a rectangle's multicast counts its sender
constexpr uint32_t kTdArgs = 26;

// The block in the dispatcher's L1: the host writes the ring and the read line.
constexpr uint32_t kTbRing = 0;        // host_depth lines of 64 bytes
constexpr uint32_t kTbRead = 4096;     // word 0: events the host has read
constexpr uint32_t kTbWords = 4160;    // staging of the host words (KMD_HOST_*)
constexpr uint32_t kTbEvents = 4224;   // staging of the events, 16 bytes each
constexpr uint32_t kTbPending = 6272;  // the command taken and not yet submitted
constexpr uint32_t kTbResult = 6336;   // staging of the result (kThResult)
constexpr uint32_t kTbBytes = 6592;

// The area in the host channel: the host words and the event ring as the X280's host
// buffer (KMD_HOST_*), then the result block (KMD_R_*), then kTr* words.
constexpr uint32_t kThResult = 4096;
constexpr uint32_t kThBytes = 8192;
constexpr uint32_t kTrAbandoned = 32;  // commands in flight at the finish
constexpr uint32_t kTrWords = 64;
constexpr uint32_t kTdHostDepthMax = 64;
constexpr uint32_t kTdEventDepthMax = 128;

// Empty cycles between two polls of L1 on a pass that found nothing.
constexpr uint32_t kTdPace = 16;
// Passes between two looks at the dispatcher's own stop flag while commands flow.
constexpr uint32_t kTdStopLook = 64;
