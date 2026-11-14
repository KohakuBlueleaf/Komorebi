// What the kernels, the overlays and Python agree on: the handler, the overlays'
// numbers, the buffers, the leader's choice and word, and the overlays' arguments.
#pragma once

#include <stdint.h>

// LEAD: a0 values_address, a1 rounds, a2 results_address, a3 first value, a4 result
// page of rank 0, a5 records_address (0: none), a6 its page of rank 0, a7 stride.
constexpr uint32_t kHandlerLead = 1;

// The overlays: the leader's choice c names overlay c + 1. mix and turn have a reader
// part; scale and shift a reader part and compute parts.
constexpr uint32_t kOverlayMix = 1;
constexpr uint32_t kOverlayTurn = 2;
constexpr uint32_t kOverlayScale = 3;
constexpr uint32_t kOverlayShift = 4;

constexpr uint32_t kCbIn = 0;       // a tile the reader fills from the core's data
constexpr uint32_t kCbChoice = 1;   // the cast word, handed by the reader to compute
constexpr uint32_t kCbValues = 2;   // the leader's page of values
constexpr uint32_t kCbData = 3;     // the core's data: kDataWords words
constexpr uint32_t kCbResult = 4;   // a core's result row, written to DRAM
constexpr uint32_t kCbRecords = 5;  // a core's timing records, written to DRAM
constexpr uint32_t kCbOut = 16;     // the tile compute made, folded back by the reader
constexpr uint32_t kValuesPage = 256;  // values (uint32) per page
constexpr uint32_t kDataWords = 32;
constexpr uint32_t kTileValues = 1024;
constexpr uint32_t kRecordRounds = 128;  // rounds timed in a command
constexpr uint32_t kRecordWords = 4;     // words of a record
// A result row: the data, then the wrong words, the first wrong round, the rounds,
// the loads of the reader part and the loads that met compute.
constexpr uint32_t kResultWords = 48;
constexpr uint32_t kResultWrong = kDataWords;
constexpr uint32_t kNone = 0xFFFFFFFFu;

// Reader arguments: the broadcast's words. Compute: whether the core is included.
constexpr uint32_t kArgWords = 0;

// The leader's choice from a value it read: which quarter of the range it falls in.
inline uint32_t pick(uint32_t value) {
    return value >> 30;
}

// The word cast: the choice in the low 2 bits, the value's index + 1 above it.
inline uint32_t choice_word(uint32_t choice, uint32_t index) {
    return choice | (index + 1) << 2;
}

// The overlay a word names, and whether it has compute parts.
inline uint32_t overlay_of(uint32_t word) {
    return (word & 3u) + 1;
}

inline bool has_compute(uint32_t word) {
    return overlay_of(word) >= kOverlayScale;
}

// The round's index a word names.
inline uint32_t index_of(uint32_t word) {
    return (word >> 2) - 1;
}

// A reader part's argument: the data's L1 address and the round's index.
struct DataArgument {
    uint32_t data;
    uint32_t index;
};

// A compute part's argument: the scalar (float32 bits).
struct TileArgument {
    uint32_t scalar;
};

// The scalar of round `index` for scale and shift: 1 to 4, as float32 bits.
inline uint32_t scalar_of(uint32_t index) {
    const uint32_t n = index & 3u;
    return n == 0 ? 0x3F800000u : 0x40000000u + (n - 1) * 0x00400000u;
}
