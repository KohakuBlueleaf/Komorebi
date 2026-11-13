// Matrix product compute: per owned row tile m and column block j0, the K chunks'
// products accumulate in DST (fp32) and the block is packed to the output buffer.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
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

    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr auto kMul = EltwiseBinaryType::ELWMUL;

    // eltwise_binary.h's mul_reuse_dest_init and _tiles at HiFi4 whatever the program's
    // fidelity: at HiFi2 an element-wise product loses its operands' low bits.
    inline void mul_hifi4_init(uint32_t icb) {
        state_configure(icb, __LINE__);
        UNPACK(
            (llk_unpack_A_init<BroadcastType::NONE, true, kReuse>(false, false, icb))
        );
        MATH((llk_math_eltwise_binary_init<
              kMul,
              BroadcastType::NONE,
              MathFidelity::HiFi4,
              kReuse>(icb, icb, false)));
    }

    inline void mul_hifi4(uint32_t icb, uint32_t tile, uint32_t dst) {
        UNPACK((llk_unpack_A<BroadcastType::NONE, true, kReuse>(icb, tile)));
        MATH((llk_math_eltwise_binary<
              kMul,
              BroadcastType::NONE,
              DST_ACCUM_MODE,
              MathFidelity::HiFi4,
              kReuse>(icb, icb, dst, true)));
    }

    // LAYER: per owned row and column block, A's row times W's block in DST (the gate
    // pass: A's block copied), the epilogue on DST, the pack to Y's buffer.
    void layer(
        const km::Compute& cp, const km_record_t& r, volatile tt_l1_ptr uint32_t* prof
    ) {
        const km::Share s = km::share(r.a2, cp.me() - km::first(r), km::count(r));
        const uint32_t kt = r.a3, nt = r.a4, ct = r.a5, wcb = r.a6, ep = r.a7;
        const uint32_t a_at = r.a8 & 0xFF, bcb = (r.a8 >> 8) & 0xFF;
        const uint32_t y_at = (r.a8 >> 16) & 0xFF;
        const uint32_t acb = a_at == kDram ? kCbA : a_at;
        const uint32_t ycb = y_at == kDram ? kCbOut : y_at;
        const bool gate = wcb == 0;
        const bool silu = ep == kEpSilu || ep == kEpGate, mul = ep == kEpMul || gate;
        uint32_t wait_a = 0, mm = 0, epi = 0, wait_math = 0, wait_out = 0;
        uint32_t pack = 0, tiles = 0;
        bool ok = true, init = true;
        for (uint32_t m = s.start; ok && m < s.start + s.len; ++m) {
            if (!gate) {
                const uint32_t t1 = km::wall();
                ok = cp.wait_front(acb, kt);
                wait_a += km::wall() - t1;
            }
            for (uint32_t j0 = 0; ok && j0 < nt; j0 += ct) {
                const uint32_t t1 = km::wall();
                tile_regs_acquire();
                if (gate) {
                    ok = cp.wait_front(acb, ct);
                    if (ok) {
                        copy_init(acb);
                        for (uint32_t j = 0; j < ct; ++j) {
                            copy_tile(acb, j, j);
                        }
                    }
                } else {
                    if (init) {  // first block, or an epilogue changed the units' state
                        matmul_block_init(acb, wcb, 0, ct, 1, kt);
                        init = ep != kEpNone;
                    }
                    for (uint32_t k = 0; k < kt; ++k) {
                        matmul_block(acb, wcb, k, k * nt + j0, 0, 0, ct, 1, kt);
                    }
                }
                const uint32_t t2 = km::wall();
                if (ok && silu) {
                    silu_tile_init();
                    for (uint32_t j = 0; j < ct; ++j) {
                        silu_tile(j);
                    }
                }
                if (ok && mul) {
                    ok = cp.wait_front(bcb, ct);
                    if (ok) {
                        mul_hifi4_init(bcb);
                        for (uint32_t j = 0; j < ct; ++j) {
                            mul_hifi4(bcb, j, j);
                        }
                        cb_pop_front(bcb, ct);
                    }
                }
                if (ok && gate) {
                    cb_pop_front(acb, ct);
                }
                tile_regs_commit();
                const uint32_t t3 = km::wall();
                tile_regs_wait();
                const uint32_t t4 = km::wall();
                const bool room = ok && cp.reserve_back(ycb, ct);
                const uint32_t t5 = km::wall();
                if (room) {
                    for (uint32_t j = 0; j < ct; ++j) {
                        pack_tile(j, ycb);
                    }
                }
                tile_regs_release();
                if (room) {
                    cb_push_back(ycb, ct);
                    tiles += ct;
                }
                mm += t2 - t1;
                epi += t3 - t2;
                wait_math += t4 - t3;
                wait_out += t5 - t4;
                pack += km::wall() - t5;
            }
            if (ok && !gate) {
                cb_pop_front(acb, kt);
            }
        }
        UNPACK((km::add_sat(prof + kPrUnWaitA, wait_a)));
        MATH((km::add_sat(prof + kPrMathMm, mm)));
        MATH((km::add_sat(prof + kPrMathEp, epi)));
        PACK((km::add_sat(prof + kPrPkWaitMath, wait_math)));
        PACK((km::add_sat(prof + kPrPkWaitOut, wait_out)));
        PACK((km::add_sat(prof + kPrPkPack, pack)));
        PACK((km::add_sat(prof + kPrPkTiles, tiles)));
        (void)wait_a, (void)mm, (void)epi, (void)wait_math, (void)wait_out, (void)pack;
        (void)tiles;
    }

}  // namespace

void kernel_main() {
    km::Compute cp;
    // Every operand is bf16; A's buffer stands for W's, which the streamed form lacks.
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbA, kCbA, kCbOut);
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_tile_address(kCbProf, 0));
    uint32_t loaded[kCbW2 + 1] = {};  // W pages held in each W buffer
    cp.run([&](const km_record_t& r) {
        const uint32_t t0 = km::wall();
        const uint32_t kt = r.a3, nt = r.a4;
        if (km::seg(r) == kSegLoad) {
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
        if (km::seg(r) == kSegLayer) {
            layer(cp, r, prof);
            km::add_sat(prof + kBody, km::wall() - t0);
            return;
        }
        if (km::seg(r) != kSegProduct) {
            return;
        }
        const km::Share s = km::share(r.a2, cp.me() - km::first(r), km::count(r));
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
        uint32_t pack = 0, tiles = 0;
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
                const bool room = ok && cp.reserve_back(kCbOut, ct);
                const uint32_t t4 = km::wall();
                if (room) {
                    for (uint32_t j = 0; j < ct; ++j) {
                        pack_tile(j, kCbOut);
                    }
                }
                tile_regs_release();
                if (room) {
                    cb_push_back(kCbOut, ct);
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
