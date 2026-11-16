// Far order: the program's handlers and the words of the sender's and watcher's area.
#pragma once

#include <stdint.h>

// CLEAR (reader): the rank's area set to 0.
constexpr uint32_t kHandlerClear = 1;
// PAIR (reader, one command over the flat ranks from min(a0, a4) to max(a0, a4)): rank
// a4 sends a1 rounds to rank a0, which watches; a2 the data tensor (a3 pages, held by
// a0's island); a5 the most each may run, milliseconds at 1,350 MHz.
constexpr uint32_t kHandlerPair = 2;
// REPORT (reader, one rank): its area written into page a1 of tensor a0.
constexpr uint32_t kHandlerReport = 3;
// The area (buffer kCbArea, one page).
constexpr uint32_t kAreaCounter = 0;  // watcher: the adds that arrived
constexpr uint32_t kAreaReady = 1;    // sender: 1 once the watcher runs
constexpr uint32_t kAreaDone = 2;     // sender: rounds sent; watcher: pages checked
constexpr uint32_t kAreaEarly = 3;    // watcher: pages read before their round's put
constexpr uint32_t kAreaStale = 4;    // watcher: reads a later round may have reached
constexpr uint32_t kAreaFirst = 5;    // watcher: the first early read's round, word 0
constexpr uint32_t kAreaGaveUp = 7;   // 1: a call gave up; 2: the time ran out
constexpr uint32_t kAreaSeen = 8;     // watcher: rises of the counter seen
constexpr uint32_t kAreaBytes = 4096;
// The L1 buffers (circular buffer indices; only their addresses are used).
constexpr uint32_t kCbArea = 0;
constexpr uint32_t kCbPage = 1;
constexpr uint32_t kTicksAMillisecond = 1350000u;

// Word `word` of round `round`'s page: no round's page equals another's.
inline uint32_t order_word(uint32_t round, uint32_t word) {
    return word == 0 ? round + 1 : (round + 1) * 0x9E3779B1u ^ word;
}
