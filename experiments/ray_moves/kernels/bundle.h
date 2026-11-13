// The transport probe: its handler, message words, command arguments, per-core results
// and the payload every message carries.
#pragma once

#include <stdint.h>

#include "destinations.h"

constexpr uint32_t kHandlerBundle = 1;  // the dynamic handler

// Message words; word 0 is the runtime's mark. The payload is a function of the id, and
// the checksum is the sum of the payload words (mod 2**32).
constexpr uint32_t kWordId = 1;
constexpr uint32_t kWordHops = 2;   // forwards left
constexpr uint32_t kWordState = 3;  // the generator that picks each destination
constexpr uint32_t kWordChecksum = 4;
constexpr uint32_t kWordPayload = 5;

// RUN command: a0 message_count over the group, a1 forwards per message.

// Per-core results, kResultWords words at the address of compute argument 3.
constexpr uint32_t kResultVisits = 0;  // messages taken from the queue
constexpr uint32_t kResultConsumed = 1;
constexpr uint32_t kResultBad = 2;       // messages whose checksum or id was wrong
constexpr uint32_t kResultRejected = 3;  // seeding sends that returned false
constexpr uint32_t kResultRun = 4;  // the run's sequence: the results are this run's
constexpr uint32_t kResultFirst = 5;                // wall clock (low word) of the first message taken
constexpr uint32_t kResultLast = 6;  // wall clock (low word) of the last message taken
constexpr uint32_t kResultCheckCycles = 7;     // pack thread: cycles summing payloads
constexpr uint32_t kResultForwardCycles = 8;   // pack thread: cycles in forward()
constexpr uint32_t kResultWaitCycles = 9;      // pack thread: cycles waiting in next()
constexpr uint32_t kResultReaderBad = 10;      // zero-copy: the reader's bad messages
constexpr uint32_t kResultReaderChecked = 11;  // zero-copy: messages the reader checked
constexpr uint32_t kResultWords = 16;

// The generator (xorshift32), a message's first state and its payload words.
inline uint32_t bundle_next(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint32_t bundle_seed(uint32_t id) {
    const uint32_t state = (id + 1) * 0x9E3779B1u;
    return state == 0 ? 1 : state;
}

inline uint32_t bundle_payload(uint32_t id, uint32_t word) {
    return (id + 1) * 0x85EBCA77u ^ word * 0xC2B2AE3Du;
}

// True when a message's checksum and id are right.
inline bool bundle_valid(
    const volatile uint32_t* message, uint32_t words, uint32_t message_count
) {
    uint32_t checksum = 0;
    for (uint32_t word = kWordPayload; word < words; word++) {
        checksum += message[word];
    }
    return checksum == message[kWordChecksum] && message[kWordId] < message_count;
}

// The rank a message goes to next: drawn from the table of destinations.
inline uint32_t bundle_destination(uint32_t state, uint32_t ranks) {
    return kDestinations[state % kDestinationCount] % ranks;
}
