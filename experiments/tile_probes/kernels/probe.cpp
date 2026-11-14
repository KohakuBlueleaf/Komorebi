// Tile probes compute kernel: one probe per build (define PROBE, see probe.hpp); the
// pack thread (TRISC2) writes the nonce last, then pushes the output page.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "fpu.hpp"
#include "icache.hpp"
#include "probe.hpp"
#include "rv.hpp"
#ifdef TRISC_PACK
#include "vec.hpp"
#include "costs.hpp"
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
    // Waits (with a timeout) for the done words of the other TRISCs in `mask`.
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
                    return probe::kErrorSpin;
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
#if PROBE == 14
    compute_kernel_hw_startup(probe::kCbIn, probe::kCbRne);
    const uint32_t rne_addr = get_tile_address(probe::kCbRne, 0);
    const uint32_t sr_addr = get_tile_address(probe::kCbSr, 0);
    (void)rne_addr;
    (void)sr_addr;
#else
    compute_kernel_hw_startup(probe::kCbIn, probe::kCbOut);
#endif
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

#if PROBE == 11 || PROBE == 14
    // TRISC 0 fills SrcA and SrcB; TRISC 1 runs the matrix unit once they are full.
#if defined(TRISC_UNPACK)
    fpu::unpack(out, nonce);
#elif defined(TRISC_MATH)
    if (fpu::wait_word(out, probe::kDone0, nonce)) {
#if PROBE == 11
        fpu::issue(out);
#else
        fpu::ties_math();
#endif
    } else {
        out[probe::kAuxError] = probe::kErrorSpin;
    }
    fpu::post_word(out, probe::kDone0 + 1, nonce);
#endif
#endif

#if PROBE == 12
    // p0 run mask, p1 loop passes.
    if ((p[0] >> kThread) & 1) {
        icache::run(kThread, out, p[1]);
    }
#ifndef TRISC_PACK
    fpu::post_word(out, probe::kDone0 + kThread, nonce);
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
#elif PROBE == 8
    vec::setup();
    costs::push(out, scratch, p[0], p[1], t);  // reps, K
    vec::finish();
#elif PROBE == 9
    vec::setup();
    costs::macro(in, out, p[0], p[1], t);  // passes, K
    vec::finish();
#elif PROBE == 10
    vec::setup();
    costs::sfpcfg(out);
    vec::finish();
#elif PROBE == 11
    err |= wait_done(out, 3, nonce);
    err |= out[probe::kAuxError];
#elif PROBE == 12
    err |= wait_done(out, p[0] & 3, nonce);
#elif PROBE == 13
    vec::setup();
    costs::sfpi_run(in, out);
    vec::finish();
#elif PROBE == 14
    err |= wait_done(out, 3, nonce);
    err |= out[probe::kAuxError];
    if (err == 0) {
        fpu::ties_pack(
            in, out, probe::kCbRne, probe::kCbSr,
            reinterpret_cast<const volatile uint32_t*>(rne_addr),
            reinterpret_cast<const volatile uint32_t*>(sr_addr)
        );
    }
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
