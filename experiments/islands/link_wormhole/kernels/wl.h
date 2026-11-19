// What the host and the two ends of a link_wormhole run share: the L1 layout (the same
// on both ends, so an end names its partner's places by its own), the words, the codes.
#pragma once

#include <stdint.h>

// The L1 layout, offsets from the area's base (the host's, 64-byte aligned).
constexpr uint32_t kStatus = 0x000;   // the status words below
constexpr uint32_t kSender = 0x100;   // the link sender's words (teardown, cursor)
constexpr uint32_t kHeaders = 0x200;  // four 64-byte packet headers
constexpr uint32_t kCounts = 0x400;   // the words the far end adds to or writes
constexpr uint32_t kPing = 0x800;     // a ping's 64 bytes, sent from here
constexpr uint32_t kPong = 0x840;     // where the partner's ping lands
constexpr uint32_t kStage = 0x900;    // a line's 128 bytes, sent from here
constexpr uint32_t kLanding = 0x980;  // where the partner's line lands
constexpr uint32_t kSource = 0x1000;  // the pages sent, `window` slots of `page` bytes
constexpr uint32_t kLineBytes = 128;  // a command's line packet (KM_LINK_PACKET_BYTES)
constexpr uint32_t kPingBytes = 64;

// The count words (4 bytes each, 16 apart), at kCounts.
constexpr uint32_t kCountPing = 0;     // pings landed here
constexpr uint32_t kCountLine = 4;     // lines landed here (their count line)
constexpr uint32_t kCountDone = 8;     // the far end's DONE word for a line
constexpr uint32_t kCountPages = 12;   // counted pages landed here
constexpr uint32_t kCountCredit = 16;  // slots the receiver of pages has freed

// Status words.
constexpr uint32_t kMagic = 0x574C4E4Bu;  // "WLNK"
constexpr uint32_t kWordMagic = 0;
constexpr uint32_t kWordCode = 1;
constexpr uint32_t kWordStep = 2;     // the phase it was in at the end
constexpr uint32_t kWordWrong = 3;    // words that differed from the pattern
constexpr uint32_t kWordEarly = 4;    // counts or words that came before their data
constexpr uint32_t kWordRounds = 5;   // round trips done (ping, then line)
constexpr uint32_t kWordPingMin = 6;  // ping round trip ticks: min, sum low, max
constexpr uint32_t kWordPingSum = 7;
constexpr uint32_t kWordPingMax = 8;
constexpr uint32_t kWordLineMin = 9;  // line and DONE round trip ticks
constexpr uint32_t kWordLineSum = 10;
constexpr uint32_t kWordLineMax = 11;
constexpr uint32_t kWordPages = 12;   // pages sent or received
constexpr uint32_t kWordSpanLo = 13;  // pages: ticks from the first to the last
constexpr uint32_t kWordSpanHi = 14;
constexpr uint32_t kWordLines = 15;  // lines done
constexpr uint32_t kStatusWords = 16;

constexpr uint32_t kCodeOk = 0;
constexpr uint32_t kCodeArgs = 1;
constexpr uint32_t kCodeSpace = 2;  // no slot in the router in time
constexpr uint32_t kCodeWait = 3;   // a count that did not come in time
constexpr uint32_t kCodeClose = 4;  // the router did not let the connection go

// Phases, a bit each.
constexpr uint32_t kPhasePing = 1;
constexpr uint32_t kPhaseLine = 2;
constexpr uint32_t kPhasePages = 4;

// Word w of page p: both ends compute it.
inline uint32_t pattern(uint32_t seed, uint32_t page, uint32_t w) {
    return seed ^ (page * 0x9E3779B1u) ^ (w * 0x27D4EB2Fu);
}
