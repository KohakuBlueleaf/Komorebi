// Link operations: the program's handlers.
#pragma once

#include <stdint.h>

// CLEAR (reader): the rank's area and its count at port a0 set to 0.
constexpr uint32_t kHandlerClear = 1;
// OPS (reader, one rank): a1 calls to rank a0, kind a5 (kKind*): km::write_l1 of a4
// bytes into ring slot call % a3, port a2 (kNoPort: none); km::add_word of 1 (counter).
constexpr uint32_t kHandlerOps = 2;
// REPORT (reader, a group of one rank): its area, its count at port a2 in word
// kAreaPort, written into page a1 of tensor a0.
constexpr uint32_t kHandlerReport = 3;
constexpr uint32_t kKindWrite = 1;
constexpr uint32_t kKindAdd = 2;
constexpr uint32_t kKindPosted = 4;  // the adds posted, km::put_wait after the calls
constexpr uint32_t kNoPort = 0xFFFFFFFFu;
// The area (buffer kCbArea, one page): counter word, port count, then the ring.
constexpr uint32_t kAreaCounter = 0;
constexpr uint32_t kAreaPort = 1;
constexpr uint32_t kAreaDone = 2;  // OPS: calls done (the sender's own area)
constexpr uint32_t kAreaRing = 64;
constexpr uint32_t kAreaBytes = 4096;
// The L1 buffers (circular buffer indices; only their addresses are used).
constexpr uint32_t kCbArea = 0;
constexpr uint32_t kCbSource = 1;

// Word `word` of call `call`'s bytes, to rank `rank`.
inline uint32_t ops_word(uint32_t call, uint32_t word, uint32_t rank) {
    return word == 0 ? call : call ^ word << 24 ^ rank << 12;
}
