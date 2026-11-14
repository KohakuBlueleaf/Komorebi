// Ranges of a page across islands: the program's handler.
#pragma once

#include <stdint.h>

// READ (reader): rank r reads a4 ranges of a3 bytes of a0, a5 in flight (km::ask_bytes;
// 1: km::read_bytes) by ranges_pick, checks words and guards. Report: a1 page a7 + r.
constexpr uint32_t kHandlerRead = 1;
constexpr uint32_t kRangesMagic = 0x52414E47u;  // "RANG": word 0 of a report
constexpr uint32_t kRangesMark = 0xA5A5A5A5u;   // a slot's words before its range lands
constexpr uint32_t kRangesNone = 0xFFFFFFFFu;
// report words: magic, rank, ranges read, words wrong, guards changed, first wrong
// range (kRangesNone), read cycles, gave up, last page and offset; bytes after
constexpr uint32_t kRangesRank = 1;
constexpr uint32_t kRangesDone = 2;
constexpr uint32_t kRangesWrong = 3;
constexpr uint32_t kRangesGuard = 4;
constexpr uint32_t kRangesFirst = 5;
constexpr uint32_t kRangesCycles = 6;
constexpr uint32_t kRangesFailed = 7;
constexpr uint32_t kRangesPage = 8;
constexpr uint32_t kRangesOffset = 9;
constexpr uint32_t kRangesCopy = 64;  // bytes
// A slot: a guard line, the range at the slot's second line plus the offset's low 6
// bits, then a guard line after it (kRangesSlot bytes for kRangesLargest at most).
constexpr uint32_t kRangesLargest = 2048;
constexpr uint32_t kRangesSlot = 64 + 64 + kRangesLargest + 64;
constexpr uint32_t kRangesInFlight = 32;  // slots in the buffer
// The L1 buffer (a circular buffer index; only its address is used): the slots, then
// a page for the report.
constexpr uint32_t kCbRanges = 0;

// Word w of page p of the source, the host's pattern.
constexpr uint32_t ranges_word(uint32_t p, uint32_t w) {
    return (p << 16 | w) ^ 0x9E3779B9u;
}

// The page and the offset (a multiple of 4 that keeps `bytes` inside a page of
// `page` bytes) of range `n` of rank `rank`, from `seed`.
struct RangeAt {
    uint32_t page;
    uint32_t offset;
};

constexpr RangeAt ranges_pick(
    uint32_t seed,
    uint32_t rank,
    uint32_t n,
    uint32_t pages,
    uint32_t page,
    uint32_t bytes
) {
    uint32_t h = seed ^ rank * 0x9E3779B1u ^ n * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return {h % pages, (h >> 8) % ((page - bytes) / 4 + 1) * 4};
}
