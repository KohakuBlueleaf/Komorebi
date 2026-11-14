// A tensor read in pages of the command's size, and a range of one DRAM bank, across
// islands: the program's handlers.
#pragma once

#include <stdint.h>

// PAGES (reader): rank r reads a3 pages from r a3 of tensor a0 (pages of a1 bytes),
// checks words, writes them xor kFormsMark at a2 + r a3. Report: page a5 + r of a4.
constexpr uint32_t kHandlerPages = 1;
// BYTES (reader): rank r reads a3 items of a0 (km::read_bytes), checks, puts them xor
// kFormsMark at pages a6 (km::put_bytes, km::put_wait). Report: page a5 + r of a4.
constexpr uint32_t kHandlerBytes = 2;
// RANGE (reader): rank r reads pages 2r, 2r + 1 of bank range a0, checks, writes them
// back xor kFormsMark (km::write_pages, km::write_bytes). Report: page a5 + r of a4.
constexpr uint32_t kHandlerRange = 3;

constexpr uint32_t kFormsMagic = 0x464F524Du;  // "FORM": word 0 of a report
constexpr uint32_t kFormsMark = 0xA5A5A5A5u;
constexpr uint32_t kFormsItem = 128;        // BYTES: an item's bytes
constexpr uint32_t kFormsRangeOffset = 20;  // RANGE: the byte read's offset
constexpr uint32_t kFormsRangeBytes = 40;   // and its bytes
constexpr uint32_t kFormsBanks = 8;         // the p150a's DRAM banks
// a report's words: magic, rank, handler, words checked, words wrong, whether a call
// gave up, the cycles in the calls, this core's island
constexpr uint32_t kFormsRank = 1;
constexpr uint32_t kFormsHandler = 2;
constexpr uint32_t kFormsChecked = 3;
constexpr uint32_t kFormsWrong = 4;
constexpr uint32_t kFormsFailed = 5;
constexpr uint32_t kFormsCycles = 6;
constexpr uint32_t kFormsIsland = 7;
constexpr uint32_t kFormsReport = 64;  // a report page's bytes
// The L1 buffer (a circular buffer index; only its address is used): from its first
// 64-byte line, kFormsData bytes of pages and ranges, then a report.
constexpr uint32_t kCbForms = 0;
constexpr uint32_t kFormsData = 8192;

// The word at bank address `offset` of bank `bank` in the tensor, the host's pattern.
constexpr uint32_t forms_word(uint32_t bank, uint32_t offset) {
    return (bank << 20 | offset >> 2) ^ 0x5A17C0DEu;
}

// The word at byte `offset` of the bank range, the host's pattern.
constexpr uint32_t forms_range_word(uint32_t offset) {
    return (offset >> 2) ^ 0x6A09E667u;
}
