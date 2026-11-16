// The message body example: its handler, the words of a message, the RUN command's
// arguments, a body's change at each hop and the per-core results.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerChain = 1;  // the dynamic handler
// --send: the RUN's handler, a part every core loads; rank 0 sends the thread to
// itself as chain, so a rank loads chain only when the thread first arrives.
constexpr uint32_t kHandlerStart = 2;

// Message words (64 bytes): word 0 is the runtime's, word 1 too in a program with
// bodies (the body's bytes); the program's fields from word 2.
constexpr uint32_t kMessageWords = 16;
constexpr uint32_t kMsgThread = 2;
constexpr uint32_t kMsgHop = 3;  // arrivals so far
constexpr uint32_t kMsgState = 4;  // a word changed at every hop, with or without a body

// RUN command: a0 threads, a1 hops per thread, a2 body bytes (0: none), a3 1: the body
// changed at each hop (0: carried), a4 1: each hop by direct() (parts/chain.cpp).

// Per-core area at the program's compute argument 0: a body (the seed's, then the
// final body of the thread that ends here), then the result words.
constexpr uint32_t kBodyWordsMax = 1024;
constexpr uint32_t kResArrivals = 1024;  // messages handled here
constexpr uint32_t kResThread = 1025;    // 1 + the thread that ended here
constexpr uint32_t kResHop = 1026;       // its arrivals
constexpr uint32_t kResState = 1027;     // its state word
constexpr uint32_t kResRun = 1028;       // the run's sequence
constexpr uint32_t kResStamps = 1029;           // parts, rank 0: wall clock at arrival hop / count
constexpr uint32_t kStamps = 8;  // laps 0 to 6, then the last lap in slot 7
constexpr uint32_t kResDirect = 1037;  // hops that went by direct() (a4)
constexpr uint32_t kAreaWords = 1040;  // the body, then 16 result words
static_assert(
    kResArrivals == kBodyWordsMax && kResStamps + kStamps <= kResDirect &&
        kResDirect < kAreaWords,
    "the area"
);

inline uint32_t chain_seed(uint32_t thread, uint32_t word) {
    return (thread + 1) * 0x9E3779B1u ^ word * 0x85EBCA6Bu;
}

// The key of arrival `hop` at rank `rank`: every word of the body changes with it.
inline uint32_t chain_key(uint32_t hop, uint32_t rank) {
    return hop * 0x9E3779B1u + rank;
}

inline uint32_t chain_word(uint32_t value, uint32_t key, uint32_t word) {
    return ((value << 1) | (value >> 31)) + (key ^ word);
}

inline uint32_t chain_state(uint32_t state, uint32_t key) {
    state ^= key;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
