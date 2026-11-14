// Islands experiment: the handlers, the L1 row every core holds and the pattern of a
// 64-byte write over the fabric link, shared by the kernels and the host.
#pragma once

#include <stdint.h>

// OPEN: the link core opens its connection to the fabric router (once a run).
constexpr uint32_t kHandlerOpen = 1;
// SEND: the link core writes the 64 bytes of seed a3 into L1 a2 of the core at NoC
// (a0, a1) of the other island, one hop away, and waits until they left.
constexpr uint32_t kHandlerSend = 2;
// COPY: waits (bounded) until its slot holds seed a3, then copies the slot's 64 bytes
// into page a1 of the DRAM tensor at a0.
constexpr uint32_t kHandlerCopy = 3;
// CLOSE: the link core closes its connection and waits for the router's answer.
constexpr uint32_t kHandlerClose = 4;

// The L1 row of every core, from its address: offsets in bytes.
constexpr uint32_t kRowTeardown = 0;   // the router's teardown answer
constexpr uint32_t kRowCursor = 16;    // the producer cursor read at the open (16 B)
constexpr uint32_t kRowHeader = 64;    // the packet header
constexpr uint32_t kRowPayload = 128;  // what SEND sends
constexpr uint32_t kRowSlot = 256;     // where the other island's SEND lands
constexpr uint32_t kRowBytes = 512;
constexpr uint32_t kSlotBytes = 64;
constexpr uint32_t kNoLink = 0xFFFFFFFFu;  // writer argument 0 of a core without a link

// Word w of the 64 bytes of seed s: word 0 is the seed itself.
inline uint32_t slot_word(uint32_t seed, uint32_t w) {
    return w == 0 ? seed : seed * 0x9E3779B1u + w;
}
