// Reservation probe: each adder core issues K non-posted atomic adds of n to one word
// on the target and records the value each returned. Every wait is bounded.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Word offsets in the shard (same address on every core).
    constexpr uint32_t kTarget = 0;  // target core: the reserved word (16-byte line)
    constexpr uint32_t kReady = 4;  // target core: start barrier counter (16-byte line)
    constexpr uint32_t kRet = 8;    // adder: return line of the atomic (16 bytes)
    constexpr uint32_t kPeek = 12;  // adder: read slot of the barrier counter
    constexpr uint32_t kStat = 16;  // adder: status words, see below
    constexpr uint32_t kPos = 32;   // adder: returned values, one per reservation
    constexpr uint32_t kSentinel = 0xDEADBEEFu;
    constexpr uint32_t kLimit = 10000000;  // cycles of every bounded wait
    constexpr uint32_t kStartLimit = 1350000000;  // start barrier: about 1 s (launch skew)

    // Status: done, fail, late, ret_mask, cyc_min, cyc_max, cyc_sum lo, cyc_sum hi.
    enum : uint32_t { kDone, kFail, kLate, kMask, kMin, kMax, kSumLo, kSumHi };

    uint64_t wall64() {
        const uint32_t lo = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
        );
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

    // One non-posted atomic add with the response written to this core's line at ret.
    void atomic_add(uint64_t dst, uint32_t incr, uint32_t ret) {
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            dst,
            NOC_UNICAST_WRITE_VC,
            incr,
            31,
            false,
            false,
            ret
        );
    }

    bool atomics_flushed_bounded() {
        const uint64_t t0 = wall64();
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index)) {
            if (wall64() - t0 > kLimit) {
                return false;
            }
        }
        return true;
    }

}  // namespace

// Args: shard address, target NoC x | y << 16, adders, K, core index.
// Reservation k of core c adds n = 1 + (7 c + k) mod 32.
void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0), xy = get_arg_val<uint32_t>(1);
    const uint32_t adders = get_arg_val<uint32_t>(2), K = get_arg_val<uint32_t>(3);
    const uint32_t c = get_arg_val<uint32_t>(4);
    volatile uint32_t* o = reinterpret_cast<volatile uint32_t*>(out);
    const uint64_t target = get_noc_addr(xy & 0xFFFF, xy >> 16, out + 4 * kTarget);
    const uint64_t ready = get_noc_addr(xy & 0xFFFF, xy >> 16, out + 4 * kReady);
    for (uint32_t i = 0; i < 8; i++) {
        o[kStat + i] = 0;
    }
    o[kStat + 8] = c;  // lets the host check which shard is whose
    o[kStat + kMin] = 0xFFFFFFFFu;

    // Start barrier: count in on the target, then wait until every adder has.
    atomic_add(ready, 1, out + 4 * kRet);
    if (!atomics_flushed_bounded()) {
        o[kStat + kFail] = 1;
        o[kStat + kDone] = 1;
        return;
    }
    const uint64_t t0 = wall64();
    for (;;) {
        noc_async_read(ready, out + 4 * kPeek, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (o[kPeek] >= adders) {
            break;
        }
        if (wall64() - t0 > kStartLimit) {
            o[kStat + kFail] = 2;
            o[kStat + kDone] = 1;
            return;
        }
    }

    uint64_t sum = 0;
    uint32_t lo = 0xFFFFFFFFu, hi = 0, late = 0;
    for (uint32_t k = 0; k < K; k++) {
        for (uint32_t w = 0; w < 4; w++) {
            o[kRet + w] = kSentinel;
        }
        const uint32_t n = 1 + (7 * c + k) % 32;
        const uint64_t a = wall64();
        atomic_add(target, n, out + 4 * kRet);
        if (!atomics_flushed_bounded()) {
            o[kStat + kFail] = 3;
            break;
        }
        const uint32_t cyc = static_cast<uint32_t>(wall64() - a);
        invalidate_l1_cache();
        if (o[kRet] == kSentinel) {  // acknowledged before the value landed
            late++;
            const uint64_t t1 = wall64();
            while (o[kRet] == kSentinel && wall64() - t1 < kLimit) {
                invalidate_l1_cache();
            }
        }
        if (k == 0) {
            uint32_t mask = 0;
            for (uint32_t w = 0; w < 4; w++) {
                mask |= (o[kRet + w] != kSentinel) << w;
            }
            o[kStat + kMask] = mask;
        }
        o[kPos + k] = o[kRet];
        sum += cyc;
        lo = cyc < lo ? cyc : lo;
        hi = cyc > hi ? cyc : hi;
    }
    o[kStat + kLate] = late;
    o[kStat + kMin] = lo;
    o[kStat + kMax] = hi;
    o[kStat + kSumLo] = static_cast<uint32_t>(sum);
    o[kStat + kSumHi] = static_cast<uint32_t>(sum >> 32);
    o[kStat + kDone] = 1;
}
