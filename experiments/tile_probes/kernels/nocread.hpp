// Probe 7: one 64-byte NoC read by a compute core (TRISC2) from another tile's L1, on
// NoC 0, command buffer 1; registers saved and restored, a timed poll for the response.
#pragma once

#include <stdint.h>

#include "probe.hpp"

namespace nocread {

    constexpr uint32_t kNocRegisters = 0xFFB20000;  // NoC 0, command buffer 0
    constexpr uint32_t kCmdBufStride = 0x800;
    constexpr uint32_t kTargLo = 0x0, kTargMid = 0x4, kTargHi = 0x8;
    constexpr uint32_t kRetLo = 0xC, kRetMid = 0x10, kRetHi = 0x14;
    constexpr uint32_t kCtrl = 0x1C, kLenBe = 0x20, kCmdCtrl = 0x40;
    constexpr uint32_t kCfgIdLogical = 0x100 + 4 * 0x12;  // NOC_CFG(NOC_ID_LOGICAL)
    constexpr uint32_t kStatusRdResp = 0x200 + 4 * 0x2;   // NIU_MST_RD_RESP_RECEIVED
    constexpr uint32_t kReadCtrl = (1u << 4) | (1u << 7) | (1u << 13);  // marked, VC 1
    constexpr uint32_t kCmdBuf = 1;

    // Output data words.
    constexpr uint32_t kWhere = 0;
    constexpr uint32_t kCode = 1;
    constexpr uint32_t kPolls = 2;
    constexpr uint32_t kCycles = 3;
    constexpr uint32_t kMyId = 4;
    constexpr uint32_t kCounterBefore = 5;
    constexpr uint32_t kCounterAfter = 6;
    constexpr uint32_t kGot = 16;

    // Where codes.
    constexpr uint32_t kStarted = 1;
    constexpr uint32_t kWaitedPeer = 2;
    constexpr uint32_t kBufReady = 3;
    constexpr uint32_t kSubmitting = 4;
    constexpr uint32_t kResponded = 5;
    constexpr uint32_t kRestored = 6;
    // Result codes.
    constexpr uint32_t kOk = 0, kBufBusy = 1, kNoResponse = 2;

    inline volatile uint32_t& reg(uint32_t buf, uint32_t off) {
        return (
            *reinterpret_cast<volatile uint32_t*>(
                kNocRegisters + buf * kCmdBufStride + off
            )
        );
    }

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    }

    inline void mark(volatile uint32_t* d, uint32_t where) {
        d[kWhere] = where;
        asm volatile("fence" ::: "memory");
        (void)d[kWhere];
    }

    // Reads 64 bytes at `addr` of the tile at NoC-0 `coord` (x | y << 6) into dst after
    // `wait` cycles (the peer's reader fills that address first).
    inline uint32_t run(
        volatile uint32_t* out,
        volatile uint32_t* dst,
        uint32_t coord,
        uint32_t addr,
        uint32_t wait
    ) {
        volatile uint32_t* d = out + probe::kData;
        mark(d, kStarted);
        const uint32_t w0 = wall();
        while (wall() - w0 < wait) {
            ;
        }
        mark(d, kWaitedPeer);
        for (uint32_t i = 0; i < 16; ++i) {
            dst[i] = 0;
        }
        asm volatile("fence" ::: "memory");
        const uint32_t my_id = reg(0, kCfgIdLogical);
        const uint32_t my_coord = my_id & 0xFFF;  // x | y << 6
        d[kMyId] = my_id;
        uint32_t n = 0;
        while (reg(kCmdBuf, kCmdCtrl) != 0) {
            if (++n >= probe::kSpinLimit) {
                d[kCode] = kBufBusy;
                return kBufBusy;
            }
        }
        mark(d, kBufReady);
        const uint32_t offs[] = {
            kTargLo, kTargMid, kTargHi, kRetLo, kRetMid, kRetHi, kCtrl, kLenBe
        };
        uint32_t saved[8];
        for (uint32_t i = 0; i < 8; ++i) {
            saved[i] = reg(kCmdBuf, offs[i]);
        }
        const uint32_t before = reg(0, kStatusRdResp);
        d[kCounterBefore] = before;
        reg(kCmdBuf, kCtrl) = kReadCtrl;
        reg(kCmdBuf, kRetLo) = reinterpret_cast<uint32_t>(dst);
        reg(kCmdBuf, kRetMid) = 0;
        reg(kCmdBuf, kRetHi) = my_coord;
        reg(kCmdBuf, kTargLo) = addr;
        reg(kCmdBuf, kTargMid) = 0;
        reg(kCmdBuf, kTargHi) = coord;
        reg(kCmdBuf, kLenBe) = 64;
        mark(d, kSubmitting);
        const uint32_t t0 = wall();
        reg(kCmdBuf, kCmdCtrl) = 1;
        uint32_t polls = 0, code = kOk;
        while (reg(0, kStatusRdResp) == before) {
            if (++polls >= probe::kSpinLimit) {
                code = kNoResponse;
                break;
            }
        }
        const uint32_t t1 = wall();
        d[kCounterAfter] = reg(0, kStatusRdResp);
        d[kPolls] = polls;
        d[kCycles] = t1 - t0;
        d[kCode] = code;
        mark(d, kResponded);
        n = 0;
        while (reg(kCmdBuf, kCmdCtrl) != 0 && ++n < probe::kSpinLimit) {
            ;
        }
        for (uint32_t i = 0; i < 8; ++i) {
            reg(kCmdBuf, offs[i]) = saved[i];
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t i = 0; i < 16; ++i) {
            d[kGot + i] = dst[i];
        }
        mark(d, kRestored);
        return code;
    }

}  // namespace nocread
