// Forward probe: layout of the per-core L1 shard and of a message, shared by the two
// kernels.
#pragma once

#include <stdint.h>

namespace forward {

    // Byte offsets in the shard (the same address on every core).
    constexpr uint32_t kTail = 0x000;   // IN_TAIL: senders add to it (atomic)
    constexpr uint32_t kReady = 0x010;  // core 0: start barrier counter
    constexpr uint32_t kRet = 0x020;    // sender: atomic return line
    constexpr uint32_t kPeek = 0x030;   // sender: read slot of the barrier counter
    constexpr uint32_t kSStat = 0x040;  // sender status, 16 words
    constexpr uint32_t kRStat = 0x080;  // receiver status, 16 words
    constexpr uint32_t kSend = 0x100;   // 32 messages
    constexpr uint32_t kRecv = 0x900;   // 32 messages
    constexpr uint32_t kShardBytes = 0x1100;

    constexpr uint32_t kMessageBytes = 64;
    constexpr uint32_t kMessageWords = kMessageBytes / 4;
    constexpr uint32_t kChunkMessages = 32;  // one DRAM page of a queue
    constexpr uint32_t kMaxPacket = 32;
    constexpr uint32_t kLimit = 20000000;         // cycles: any wait for another core
    constexpr uint32_t kStartLimit = 1350000000;  // cycles: start barrier (launch skew)

    // Sender status words.
    enum : uint32_t {
        kSDone,
        kSFail,
        kSMessages,
        kSStart,
        kSEnd,
        kSAtomMax,
        kSAtomSum
    };

    // Receiver status words.
    enum : uint32_t {
        kRDone,
        kRFail,
        kRGot,
        kRSumLo,
        kRSumHi,
        kRTorn,
        kRShort,
        kRFirst,
        kRLast,
        kRGapPos,
        kRReads,
        kRBadSrc,
    };

    // Fail codes.
    enum : uint32_t {
        kOk,
        kFailAtomic,
        kFailBarrier,
        kFailFlush,
        kFailGap,
        kFailOverflow
    };

    // Message words: the mark in word 0 of every 16-byte chunk, then source, packet,
    // index and a payload the host can recompute.
    constexpr uint32_t kWSrc = 1, kWPacket = 2, kWIndex = 3, kWPayload = 5;

    inline uint32_t payload(uint32_t src, uint32_t k, uint32_t i) {
        return (src * 0x9E3779B1u) ^ (k * 0x85EBCA77u) ^ i;
    }

    inline uint32_t lcg(uint32_t x) {
        return x * 1664525u + 1013904223u;
    }

}  // namespace forward
