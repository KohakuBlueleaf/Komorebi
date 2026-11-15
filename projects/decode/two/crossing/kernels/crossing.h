// A crossing of the two-card decode alone: its handlers and stamp words.
#pragma once

#include <stdint.h>

// CROSS: a0 the activation (a copy, 2 x a2 pages), a1 the stamps (a copy, a page a
// flat rank), a2 a card's half in pages, a3 roots, a4 their stride, a5 a root's pages,
// a6 the way (1: V1, 2: V2), a7 the run's salt.
constexpr uint32_t kHandlerCross = 1;
// GATHER: a0 .. a2 as CROSS: km::all_gather of the 2 x a2 pages, each half written.
constexpr uint32_t kHandlerGather = 2;

constexpr uint32_t kWayLeader = 1;  // V1: roots mark the leader, it sends one line
constexpr uint32_t kWayRoots = 2;   // V2: each root puts its block straight across

constexpr uint32_t kPageBytes = 2048;  // a bf16 tile page
constexpr uint32_t kStampBytes = 64;   // a rank's stamps page

// The stamps (wall clock, low word): entry, block written, leader marked, all marked,
// line out, far half counted, released, own line confirmed, the run's salt.
enum CrossStamp : uint32_t {
    kXsEntry,
    kXsWritten,
    kXsMarked,
    kXsMet,
    kXsPosted,
    kXsArrived,
    kXsReleased,
    kXsConfirmed,
    kXsSalt,
    kXsWords = 16
};
