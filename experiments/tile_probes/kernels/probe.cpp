// Tile probes compute kernel: one probe per build (define PROBE, see probe.hpp).
// Probes 0 to 4 run on the pack thread (TRISC2), which issues the vector-unit
// instructions and reads Dst through the window. Probe 5 runs on the TRISCs in its run
// mask. The pack thread writes the nonce last, then pushes the output page.
// Runtime args: nonce, then probe parameters p0..p6.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "probe.hpp"
#include "rv.hpp"
#ifdef TRISC_PACK
#include "vec.hpp"
#include "nocread.hpp"
#include "walk.hpp"
#endif

#ifndef PROBE
#define PROBE 0
#endif

namespace {

#if defined(TRISC_UNPACK)
    constexpr uint32_t kThread = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t kThread = 1;
#else
    constexpr uint32_t kThread = 2;
#endif

    inline void l1_fence() {
        asm volatile("fence" ::: "memory");
    }

#ifdef TRISC_PACK
    // Waits (bounded) for the done words of the other TRISCs in `mask`.
    uint32_t wait_done(volatile uint32_t* out, uint32_t mask, uint32_t nonce) {
        for (uint32_t t = 0; t < 2; ++t) {
            if (!((mask >> t) & 1)) {
                continue;
            }
            uint32_t n = 0;
            while (true) {
                l1_fence();
                if (out[probe::kDone0 + t] == nonce) {
                    break;
                }
                if (++n >= probe::kSpinLimit) {
                    return probe::kErrSpin;
                }
            }
        }
        return 0;
    }
#endif

}  // namespace

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    uint32_t p[7];
    for (uint32_t i = 0; i < 7; ++i) {
        p[i] = get_arg_val<uint32_t>(1 + i);
    }
    compute_kernel_hw_startup(probe::kCbIn, probe::kCbOut);
    cb_wait_front(probe::kCbIn, 1);
    const uint32_t in_addr = get_tile_address(probe::kCbIn, 0);
    const uint32_t out_addr = get_tile_address(probe::kCbOut, 0);
    const uint32_t scratch_addr = get_tile_address(probe::kCbScratch, 0);
    const volatile uint32_t* in = reinterpret_cast<const volatile uint32_t*>(in_addr);
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(out_addr);
    volatile uint32_t* scratch = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    (void)in;
    (void)scratch;

#if PROBE == 5
    // p0 run mask, p1 loops, p2 n, p3 n_float, p4 a bits, p5 b bits.
    if ((p[0] >> kThread) & 1) {
        rv::run(kThread, out, scratch, p[1], p[2], p[3], p[4], p[5]);
    }
#ifndef TRISC_PACK
    l1_fence();
    out[probe::kDone0 + kThread] = nonce;
    l1_fence();
#endif
#endif

#ifdef TRISC_PACK
    uint32_t err = 0;
    vec::Times t{out};
    out[probe::kNumTimes] = 0;
#if PROBE == 0
    vec::setup();
    vec::smoke(in, out, t);
    vec::finish();
#elif PROBE == 1
    vec::setup();
    vec::pred(out, p[0], p[1], t);  // reps, K
    vec::finish();
#elif PROBE == 2
    vec::setup();
    vec::anycc(out, p[0], t, err);  // K
    vec::finish();
#elif PROBE == 3
    vec::setup();
    if (p[0] == 1) {  // W, reps, sets, variant (0 plain, 1 scheduled, 2 RVV)
        vec::gather<1>(in, out, p[1], p[2], p[3], t);
    } else if (p[0] == 2) {
        vec::gather<2>(in, out, p[1], p[2], p[3], t);
    } else {
        vec::gather<4>(in, out, p[1], p[2], p[3], t);
    }
    vec::finish();
#elif PROBE == 4
    vec::setup();
    vec::slab(in, out, p[0], p[1], static_cast<vec::SlabMode>(p[2]), p[3], t);
    vec::finish();
#elif PROBE == 5
    err |= wait_done(out, p[0], nonce);
#elif PROBE == 6
    vec::setup();
    walk::run(in, out, const_cast<uint32_t*>(scratch), p, err);
    vec::finish();
#elif PROBE == 7
    // p0 peer NoC-0 coordinate, p1 wait cycles; the peer's input CB is at in_addr.
    nocread::run(out, scratch, p[0], in_addr, p[1]);
#endif
    out[probe::kError] = err;
    l1_fence();
    out[probe::kNonce] = nonce;
    l1_fence();
    (void)out[probe::kNonce];
    cb_reserve_back(probe::kCbOut, 1);
    cb_push_back(probe::kCbOut, 1);
#endif
}
