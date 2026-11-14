// What the host and the two kernels of the link-rate runs share.
#pragma once

#include <stdint.h>

constexpr uint32_t kMagic = 0x4C524154u;  // "LRAT": a status page was written
constexpr uint32_t kStatusBytes = 64;     // the status words, then a copy of the slots
constexpr uint32_t kCopyBytes = 1024;     // the receiver's copy of its first slots
constexpr uint32_t kHeaderStride = 64;    // bytes between the packet headers in L1

// Status words (32-bit, at the start of a core's page in the status buffer).
constexpr uint32_t kWordMagic = 0;
constexpr uint32_t kWordCode = 1;      // one of the codes below
constexpr uint32_t kWordPackets = 2;   // sender: sent; receiver: counted
constexpr uint32_t kWordWrong = 3;     // receiver: payload words that differ
constexpr uint32_t kWordCyclesLo = 4;  // sender: first send to last flush; receiver:
constexpr uint32_t kWordCyclesHi = 5;  // first count to the last
constexpr uint32_t kWordFirst = 6;     // receiver: start to the first count, kcycles

constexpr uint32_t kCodeOk = 0;
constexpr uint32_t kCodeArgs = 1;          // the arguments do not fit this source
constexpr uint32_t kCodeSlotTimeout = 2;   // sender: no free slot in the router in time
constexpr uint32_t kCodeCloseTimeout = 3;  // sender: no teardown answer in time
constexpr uint32_t kCodeCountTimeout = 4;  // receiver: not every count in time
constexpr uint32_t kCodeFlushTimeout = 5;  // a NoC flush did not end in time

// Where a packet's payload lands on the other card.
constexpr uint32_t kTargetL1 = 0;      // a core's L1 slots, a kernel there counts
constexpr uint32_t kTargetDram = 1;    // a DRAM bank, no kernel there
constexpr uint32_t kTargetBareL1 = 2;  // a core's L1 slots, no kernel there

// The payload word w of slot s from sender `sender`: both sides compute it.
inline uint32_t pattern(uint32_t seed, uint32_t sender, uint32_t slot, uint32_t w) {
    return seed ^ (sender * 0x9E3779B1u) ^ (slot * 0x85EBCA77u) ^ (w * 0x27D4EB2Fu);
}
