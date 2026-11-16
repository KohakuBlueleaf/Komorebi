// The matrix product's compute with Y's blocks split over the movers: output block b of
// the core goes to the half of the mover that Y_PATTERN names (LAYER left out).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

namespace {

#if defined(TRISC_UNPACK)
    constexpr uint32_t kBody = kPrUnBody;
#elif defined(TRISC_MATH)
    constexpr uint32_t kBody = kPrMathBody;
#else
    constexpr uint32_t kBody = kPrPkBody;
#endif

}  // namespace

void kernel_main() {
    km::Compute cp;
    // Every operand is bf16; A's buffer stands for W's, which the streamed form lacks.
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbA, kCbA, kCbOut);
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_tile_address(kCbProf, 0));
    uint32_t loaded[kCbW2 + 1] = {};  // W pages held in each W buffer
    cp.run([&](const km_command_t& r) {
        const uint32_t t0 = km::wall();
        const uint32_t kt = r.a3, nt = r.a4;
        if (km::handler(r) == kHandlerLoad) {
            const uint32_t cb = r.a1;
            if (cb != kCbW && cb != kCbW3 && cb != kCbW2) {
                return;
            }
            if (loaded[cb] != 0) {
                cb_pop_front(cb, loaded[cb]);
                loaded[cb] = 0;
            }
            if (cp.wait_front(cb, kt * nt)) {
                loaded[cb] = kt * nt;
            }
            return;
        }
        if (km::handler(r) != kHandlerProduct) {
            return;
        }
        const km::Partition s = (
            km::partition(r.a2, cp.me() - km::first(r), km::count(r))
        );
        const uint32_t kc = r.a5, ct = r.a6, mode = r.a7;
        const bool stream = r.a8 == kFormStream;
        const bool row_item = stream || kc == kt;
        const bool reads = mode != kModeComputeOnly;
        const bool math = mode == kModeFull || mode == kModeComputeOnly;
        const uint32_t cbw = stream ? kCbWs : kCbW;
        if (math) {
            matmul_block_init(kCbA, cbw, 0, ct, 1, kc);
        }
        uint32_t wait_a = 0, wait_w = 0, mm = 0, wait_math = 0, wait_out = 0;
        uint32_t pack = 0, tiles = 0, block = 0;
        const km::Pattern pattern = km::Pattern::of(Y_PATTERN);
        bool ok = true;
        for (uint32_t m = s.start; ok && m < s.start + s.len; ++m) {
            if (row_item && reads) {
                const uint32_t t1 = km::wall();
                ok = cp.wait_front(kCbA, kt);
                wait_a += km::wall() - t1;
            }
            for (uint32_t j0 = 0; ok && j0 < nt; j0 += ct) {
                const uint32_t t1 = km::wall();
                if (math) {
                    tile_regs_acquire();
                }
                for (uint32_t k0 = 0; ok && k0 < kt; k0 += kc) {
                    if (!row_item && reads) {
                        const uint32_t t2 = km::wall();
                        ok = cp.wait_front(kCbA, kc);
                        wait_a += km::wall() - t2;
                    }
                    if (ok && stream && reads) {
                        const uint32_t t2 = km::wall();
                        ok = cp.wait_front(kCbWs, kc * ct);
                        wait_w += km::wall() - t2;
                    }
                    if (ok && math) {
                        // A: the chunk's tiles (or the row's from k0); W: k rows of nt
                        // (resident) or of ct (a streamed chunk).
                        const uint32_t a0 = row_item ? k0 : 0;
                        for (uint32_t k = 0; k < kc; ++k) {
                            const uint32_t w = stream ? k * ct : (k0 + k) * nt + j0;
                            matmul_block(kCbA, cbw, a0 + k, w, 0, 0, ct, 1, kc);
                        }
                    }
                    if (ok && !row_item && reads) {
                        cb_pop_front(kCbA, kc);
                    }
                    if (ok && stream && reads) {
                        cb_pop_front(kCbWs, kc * ct);
                    }
                }
                if (!math) {
                    continue;
                }
                tile_regs_commit();
                const uint32_t t2 = km::wall();
                mm += t2 - t1;
                tile_regs_wait();
                const uint32_t t3 = km::wall();
                const uint32_t out = km::block_half(
                    {kOutputMoverA, kCbOut}, pattern, block
                );
                block++;
                const bool room = ok && cp.reserve_back(out, ct);
                const uint32_t t4 = km::wall();
                if (room) {
                    for (uint32_t j = 0; j < ct; ++j) {
                        pack_tile(j, out);
                    }
                }
                tile_regs_release();
                if (room) {
                    cb_push_back(out, ct);
                    tiles += ct;
                }
                wait_math += t3 - t2;
                wait_out += t4 - t3;
                pack += km::wall() - t4;
            }
            if (ok && row_item && reads) {
                cb_pop_front(kCbA, kt);
            }
        }
        UNPACK((km::add_sat(prof + kPrUnWaitA, wait_a)));
        UNPACK((km::add_sat(prof + kPrUnWaitW, wait_w)));
        MATH((km::add_sat(prof + kPrMathMm, mm)));
        PACK((km::add_sat(prof + kPrPkWaitMath, wait_math)));
        PACK((km::add_sat(prof + kPrPkWaitOut, wait_out)));
        PACK((km::add_sat(prof + kPrPkPack, pack)));
        PACK((km::add_sat(prof + kPrPkTiles, tiles)));
        km::add_sat(prof + kBody, km::wall() - t0);
        (void)pack;
    });
}
