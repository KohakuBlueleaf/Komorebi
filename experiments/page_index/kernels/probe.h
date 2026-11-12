// Page-by-index probe: what the kernels on the two cores share.
#pragma once

#include <stdint.h>

constexpr uint32_t kCbArrive = 0;  // S pages on both cores; the sender's page 0 is
                                   // its source
constexpr uint32_t kCbOut = 16;    // receiver: the pages compute read, in order
constexpr uint32_t kPages = 4;     // S
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kLimit = 1u << 26;  // cycles of every bounded wait (about 50 ms)

// Words of the flag shard (16 per core, the same address on both).
constexpr uint32_t kFlag = 0;    // receiver: bumped by the sender after its put
constexpr uint32_t kSend = 4;    // sender: 1 done, 2 read timed out, 3 write, 4 bump
constexpr uint32_t kRecv = 8;    // receiver reader: 1 flag seen, 2 timed out
constexpr uint32_t kWrite = 12;  // receiver writer: 1 written, 2 timed out

// Args. Sender: in, flag shard, receiver NoC x | y << 16, slot k. Reader: in, flags, k.
// Writer: out, flag shard. Compute: slot k, then the other slot it reads.

inline uint64_t probe_wall() {
    const uint32_t lo = (
        *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
    );
    const uint32_t hi = (
        *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
    );
    return static_cast<uint64_t>(hi) << 32 | lo;
}

// Polls ready() for at most kLimit cycles; false when the bound passed.
template <typename Ready>
inline bool probe_poll(Ready ready) {
    const uint64_t t0 = probe_wall();
    for (;;) {
        if (ready()) {
            return true;
        }
        if (probe_wall() - t0 > kLimit) {
            return false;
        }
    }
}
