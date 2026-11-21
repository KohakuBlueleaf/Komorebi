// A posted write into another island's L1: the handler and the buffers it uses.
#pragma once

#include <stdint.h>

// ROUNDS: a0 output, a1 posted (0: confirmed), a2 bytes, a3 rounds; group rank 0 sends,
// rank 1 checks each row and answers; posted: then the order and the fence.
constexpr uint32_t kHandlerRounds = 1;

constexpr uint32_t kCbRows = 0;     // the rows sent: kRowsHeld of kRowBytes
constexpr uint32_t kCbLanding = 1;  // where they land: two rows
constexpr uint32_t kCbWords = 2;    // the small words: 16 B a use
constexpr uint32_t kCbResults = 3;  // 16 B a round: call, one way, round trip, bad
constexpr uint32_t kPage = 4096;
constexpr uint32_t kRowBytes = 20480;
constexpr uint32_t kRowsHeld = 3;
constexpr uint32_t kRoundsMax = 1000;
constexpr uint32_t kResultPages = kRoundsMax * 16 / kPage + 1;

// the output's pages: the sender's rounds, then the receiver's words and last row
constexpr uint32_t kOutWords = kResultPages;
constexpr uint32_t kOutRow = kOutWords + 1;
constexpr uint32_t kOutPages = kOutRow + kRowBytes / kPage;
