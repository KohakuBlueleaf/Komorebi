// One core writes 4-byte NoC writes into another core's L1 with chosen source and
// destination words; the other core runs nothing. Every wait is bounded.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kSrcWord = 32;          // 16 source words S[i] = 0xA0A00000 | i
    constexpr uint32_t kResultWord = 64;       // per test: 1 posted, 2 acked, 3 no ack
    constexpr uint32_t kSettle = 20000;        // cycles after a posted write
    constexpr uint32_t kFlushLimit = 1000000;  // cycles of the bounded flush poll

    uint64_t wall64() {
        const uint32_t lo = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
        );
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

}  // namespace

// Args: out (the shard address on both cores), the other core's NoC x | y << 16, the
// number of tests, then per test: slot | sw << 8 | dw << 16 | posted << 24.
void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0), xy = get_arg_val<uint32_t>(1);
    const uint32_t tests = get_arg_val<uint32_t>(2);
    volatile uint32_t* o = reinterpret_cast<volatile uint32_t*>(out);
    for (uint32_t i = 0; i < 16; i++) {
        o[kSrcWord + i] = 0xA0A00000u | i;
    }
    for (uint32_t t = 0; t < tests; t++) {
        const uint32_t w = get_arg_val<uint32_t>(3 + t);
        const uint32_t slot = w & 0xFF, sw = (w >> 8) & 0xFF, dw = (w >> 16) & 0xFF;
        const uint32_t src = out + 4 * (kSrcWord + sw);
        const uint64_t dst = (
            get_noc_addr(xy & 0xFFFF, xy >> 16, out + 64 * slot + 4 * dw)
        );
        uint32_t result = 1;
        if (w >> 24) {  // posted: no acknowledgement to wait for
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(src, dst, 4);
            const uint64_t t0 = wall64();
            while (wall64() - t0 < kSettle) {
                ;
            }
        } else {  // non-posted: poll the acknowledgement count, bounded
            noc_async_write(src, dst, 4);
            const uint64_t t0 = wall64();
            bool flushed = false;
            while (!flushed && wall64() - t0 < kFlushLimit) {
                flushed = ncrisc_noc_nonposted_writes_flushed(noc_index);
            }
            result = flushed ? 2 : 3;  // 3: no acknowledgement within the bound
        }
        o[kResultWord + t] = result;
    }
}
