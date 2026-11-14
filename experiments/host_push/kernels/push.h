// What the push kernel and the host script share: limits, status codes, the layout of
// a core's report words and the pattern of the first step.
#pragma once

#include <stdint.h>

constexpr uint32_t kChannelLimit = 64u << 20;  // a push stays below this offset
constexpr uint32_t kCbBuffer = 0;              // circular buffer holding the L1 buffers
constexpr uint32_t kLimit = 1u << 28;          // cycles of every timed wait (0.2 s)
constexpr uint32_t kMagic = 0x50555348;        // "PUSH": ends a sequence word

// The sequence word: 16 bytes written after the data was flushed.
// Words: nonce, bytes pushed, core index, kMagic ^ nonce.

// Report words (16 a core, in the core's L1).
constexpr uint32_t kStatus = 0;  // 0 not finished, 1 done, 2 and up: see below
constexpr uint32_t kPushed = 1;  // bytes pushed
constexpr uint32_t kStartLow = 2;
constexpr uint32_t kStartHigh = 3;
constexpr uint32_t kDataLow = 4;  // wall clock when the data writes were flushed
constexpr uint32_t kDataHigh = 5;
constexpr uint32_t kMidClear = 6;  // 1: the write buffer held no PCIe routing at the end
constexpr uint32_t kChunks = 7;
constexpr uint32_t kEndLow = 8;  // wall clock when the sequence word was flushed
constexpr uint32_t kEndHigh = 9;

constexpr uint32_t kDone = 1;
constexpr uint32_t kBadArguments = 2;  // refused before any write
constexpr uint32_t kReadTimeout = 3;
constexpr uint32_t kWriteTimeout = 4;
constexpr uint32_t kSequenceTimeout = 5;
constexpr uint32_t kRoutingLeft = 6;  // the write buffer could not be cleared

// Arguments: report, mode (0 pattern, 1 DRAM), tensor, first page, bytes, page bytes,
// channel offset, chunk, nonce, sequence word offset, core index, L1 buffer bytes.
constexpr uint32_t kArgumentCount = 12;

constexpr uint32_t kPatternMultiplier = 2654435761u;

// Word i of the channel in the pattern mode.
inline uint32_t pattern_word(uint32_t word, uint32_t nonce) {
    return word * kPatternMultiplier ^ nonce;
}
