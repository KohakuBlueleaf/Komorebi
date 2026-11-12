// Crossing costs: what the kernels of the hop and bump probes share.
#pragma once

#include <stdint.h>

constexpr uint32_t kCbBlock = 0;  // every core: the put's source and destination
constexpr uint32_t kCbTimes = 1;  // timing core: cycles per repetition
constexpr uint32_t kCbSlot = 2;   // every core: 16-byte words (release, port, sync)
constexpr uint32_t kFailed = 0xFFFFFFFFu;  // a repetition whose wait passed its bound
constexpr uint32_t kLate = 0x10000u;  // a bump from a sender that saw its release late

// Words of kCbSlot (16 bytes apart).
constexpr uint32_t kSlotSync = 0;     // hop: the NoC0 kernel is done (the launch's tag)
constexpr uint32_t kSlotRelease = 4;  // bump sender: release time, tag | repetition + 1
constexpr uint32_t kSlotPort = 8;     // bump receiver: the port senders increment
constexpr uint32_t kSlotOut = 12;     // source of 4- and 8-byte writes
constexpr uint32_t kSlotCheckin = 16;  // bump receiver: sender k's tag at word 16 + 4 k
constexpr uint32_t kSlotBytes = 64 + 16 * 128;

inline uint32_t wall_lo() {
    return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

// Polls ready() for at most `limit` cycles; false when the bound passed.
template <typename Ready>
inline bool poll_bounded(Ready ready, uint32_t limit) {
    const uint32_t t0 = wall_lo();
    for (;;) {
        invalidate_l1_cache();
        if (ready()) {
            return true;
        }
        if (wall_lo() - t0 > limit) {
            return false;
        }
    }
}
