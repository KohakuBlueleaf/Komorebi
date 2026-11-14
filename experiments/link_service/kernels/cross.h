// The link experiment: its dynamic handler, the message words, the RUN command's
// arguments and each core's results, shared by the kernels and the host.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerCross = 1;  // the dynamic handler

// Message words (64 bytes); word 0 is the runtime's, word 1 the route word the link
// reads (island << 16 | rank).
constexpr uint32_t kMessageWords = 16;
constexpr uint32_t kMsgRoute = 1;
constexpr uint32_t kMsgOrigin = 2;  // the rank that seeded it, on island 0
constexpr uint32_t kMsgTrips = 3;   // round trips left
constexpr uint32_t kMsgId = 4;      // origin * messages + index

// RUN command: a0 origins (island 0's first a0 ranks but the link core), a1 messages
// each, a2 round trips each, a3 mode, a4 the link core's rank, a5 entries a packet.
constexpr uint32_t kModeLocal = 0;   // partner: rank + ranks / 2 of the same island
constexpr uint32_t kModeAcross = 1;  // partner: the same rank of island 1, by the link

// Per-core results, 16 words at the address of compute argument 0 (the link core's
// region at argument 2).
constexpr uint32_t kResConsumed = 0;  // messages ended here (an origin)
constexpr uint32_t kResIdsLo = 1;     // sum of their ids, 64 bits
constexpr uint32_t kResIdsHi = 2;
constexpr uint32_t kResVisits = 3;    // messages handled here as a partner
constexpr uint32_t kResFirst = 4;     // wall clock (low word) at the first seed
constexpr uint32_t kResLast = 5;      // wall clock (low word) at the last end
constexpr uint32_t kResRejected = 6;  // seeds refused for lack of quota
constexpr uint32_t kResWrong = 7;     // messages whose words were not as sent
constexpr uint32_t kResRun = 8;       // the run's sequence: the results are this run's
constexpr uint32_t kResWords = 16;
constexpr uint32_t kResLinkWords = 52;  // the link core: its region's words follow
