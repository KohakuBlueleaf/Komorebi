// The probe's persistent kernel on data-movement core 1: parked in a wait loop outside
// every slot; on each go it drains or checks a buffer, runs its part or sets up.
#include <stdint.h>

#include "api.h"
#include "costs.h"
#include "internal/circular_buffer_init.h"

namespace {

    volatile uint32_t part_calls = 0;

    __attribute__((noinline)) uint32_t mix(uint32_t x) {
        return cs::mix(x);
    }

    inline uint64_t wall64() {
        volatile tt_reg_ptr uint32_t*
            lo = (
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_L
                )
            );
        volatile tt_reg_ptr uint32_t*
            hi = (
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_1
                )
            );
        for (;;) {
            const uint32_t h = *hi, l = *lo;
            if (*hi == h) {
                return static_cast<uint64_t>(h) << 32 | l;
            }
        }
    }

    // Pops `count` pages of kCbRead, summing each page's first word.
    uint32_t drain(uint32_t count, volatile uint32_t* stop, uint32_t& sum) {
        for (uint32_t i = 0; i < count; i++) {
            if (!cs::wait_front(cs::kCbRead, 1, stop)) {
                return cs::kWaitTimeout;
            }
            sum += (
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_read_ptr(cs::kCbRead)
                )
            );
            cb_pop_front(cs::kCbRead, 1);
        }
        return cs::kOk;
    }

    // Pops `count` tiles of kCbOut, compares `values` values of each with the 16
    // expected (repeating); counts wrong tiles and read pointers outside [low, high).
    uint32_t check(
        uint32_t count,
        uint32_t values,
        volatile tt_l1_ptr uint32_t* c,
        uint32_t low,
        uint32_t high,
        uint32_t& wrong,
        uint32_t& bad
    ) {
        volatile tt_l1_ptr uint16_t*
            expect = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c + cs::kExpect));
        uint16_t want[16];
        for (uint32_t j = 0; j < 16; j++) {
            want[j] = expect[j];
        }
        for (uint32_t i = 0; i < count; i++) {
            if (!cs::wait_front(cs::kCbOut, 1, c + cs::kStop)) {
                return cs::kWaitTimeout;
            }
            const uint32_t at = get_read_ptr(cs::kCbOut);
            bad += at < low || at + cs::kTileBytes > high;
            volatile tt_l1_ptr uint16_t*
                out = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(at));
            uint32_t differ = 0;
            for (uint32_t j = 0; j < values; j++) {
                differ |= out[j] ^ want[j % 16];
            }
            wrong += differ != 0;
            cb_pop_front(cs::kCbOut, 1);
        }
        return cs::kOk;
    }

}  // namespace

void kernel_main() {
    const uint32_t control = get_arg_val<uint32_t>(0);
    const uint32_t slots = get_arg_val<uint32_t>(1);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(2);
    const uint64_t idle = static_cast<uint64_t>(get_arg_val<uint32_t>(3)) << 20;
    volatile tt_l1_ptr uint32_t* c = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control)
    );
    volatile tt_l1_ptr uint32_t* out = c + cs::kOut;
    const cs::Entry entry = reinterpret_cast<cs::Entry>(slots + slot_bytes);
    cs::Call call{};
    call.stop = c + cs::kStop;
    call.mix = mix;
    call.calls = &part_calls;
    uint32_t done = 0;
    uint64_t seen_at = wall64();
    for (;;) {
        invalidate_l1_cache();
        const uint32_t go = c[cs::kGo];
        if (go == cs::kLeave) {
            return;
        }
        if (go == done) {
            if (wall64() - seen_at > idle) {
                return;
            }
            continue;
        }
        const uint64_t t0 = wall64();
        seen_at = t0;
        out[cs::kOutStart] = static_cast<uint32_t>(t0);
        const uint32_t op = c[cs::kParkedOp];
        uint32_t status = cs::kOk, result = 0, bad = 0;
        if (op == cs::kParkedDrain) {
            status = drain(c[cs::kCount], call.stop, result);
        } else if (op == cs::kParkedCompute || op == cs::kParkedComputeSlot) {
            status = check(c[cs::kCount], 16, c, 0, 0xFFFFFFFFu, result, bad);
        } else if (op == cs::kParkedCheck) {
            const uint32_t* layout = (
                reinterpret_cast<const uint32_t*>(c[cs::kParkedLayout])
            );
            const uint32_t low = layout[4 * cs::kCbOut];
            const uint32_t high = low + layout[4 * cs::kCbOut + 1];
            status = check(c[cs::kTiles], cs::kTileValues, c, low, high, result, bad);
        } else if (op == cs::kParkedPart) {
            call.arg = c[cs::kArg];
            result = entry(&call);
        } else if (op == cs::kParkedSetup) {
            setup_local_cb_read_write_interfaces<true, true, false>(
                reinterpret_cast<uint32_t tt_l1_ptr*>(c[cs::kParkedLayout]),
                0,
                c[cs::kParkedMask]
            );
        } else if (op != cs::kParkedNone) {
            status = cs::kBadOp;
        }
        out[cs::kOutStatus] = status;
        out[cs::kOutResult] = result;
        out[cs::kOutBad] = bad;
        out[cs::kOutCycles] = static_cast<uint32_t>(wall64() - t0);
        done = go;
        c[cs::kDone] = done;
    }
}
