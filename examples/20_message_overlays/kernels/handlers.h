// The message overlays example: its three handlers (each an overlay part), the words
// of a message, the per-core results in L1 and the table the cores look keys up in.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerLookup = 1;  // at a key's owner: find the key's value
constexpr uint32_t kHandlerValue = 2;  // at the value's owner: add it, choose the next key
constexpr uint32_t kHandlerReport = 3;  // at the thread's origin: count the result

// Message words (64 bytes); word 0 is the runtime's (mark and handler).
constexpr uint32_t kMessageWords = 16;
constexpr uint32_t kMsgThread = 1;
constexpr uint32_t kMsgOrigin = 2;  // the rank that seeded the thread
constexpr uint32_t kMsgKey = 3;
constexpr uint32_t kMsgValue = 4;
constexpr uint32_t kMsgSteps = 5;   // lookups left
constexpr uint32_t kMsgSum = 6;     // sum of the values found (mod 2**32)
constexpr uint32_t kMsgProbes = 7;  // cores asked for a key, over the thread's life

// RUN command of kHandlerLookup: a0 threads, a1 steps per thread, a2 0 (or a handler
// that is not the program's, which one seed names: the run stops).

// Per-core results, 16 words at the address of the program's compute argument 0: the
// handlers' state between messages, added to by the unpack core.
constexpr uint32_t kResLookups = 0;  // messages handled as kHandlerLookup
constexpr uint32_t kResValues = 1;
constexpr uint32_t kResReports = 2;
constexpr uint32_t kResOwn = 3;    // forwards to this core itself
constexpr uint32_t kResSumLo = 4;  // sum of the reported sums, 64 bits
constexpr uint32_t kResSumHi = 5;
constexpr uint32_t kResProbes = 6;  // sum of the reported probes
constexpr uint32_t kResWrong = 7;  // a report away from its origin, or an unknown handler
constexpr uint32_t kResRejected = 8;  // seeds that found no quota
constexpr uint32_t kResRun = 9;       // the run's sequence, written when seeding
constexpr uint32_t kResWords = 16;

// The table: a 32-bit mixer stands for the data, so nothing is loaded.
inline uint32_t table_mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

inline uint32_t first_key(uint32_t thread) {
    return table_mix(thread + 1);
}

// Whether rank `rank` of `count` holds key `key` (three times in four; always the rank
// before its owner, the last one asked); else the next rank is asked.
inline bool table_holds(uint32_t key, uint32_t rank, uint32_t count) {
    return (
        rank == (key % count + count - 1) % count ||
        (table_mix(key ^ rank * 0x9E3779B1u) & 3) != 0
    );
}

inline uint32_t table_value(uint32_t key) {
    return table_mix(key + 0x632BE5ABu);
}

inline uint32_t next_key(uint32_t value) {
    return table_mix(value ^ 0x85EBCA6Bu);
}
