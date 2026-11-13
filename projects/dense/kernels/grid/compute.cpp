// Grid compute: A blocks from the row broadcast, W blocks from the column broadcast,
// summed into fp32 partials (kCbPart) by packer accumulation; the last K block packs
// bf16 output.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "grid.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

namespace {

#if defined(TRISC_UNPACK)
    constexpr uint32_t kBody = kGpUnBody;
#elif defined(TRISC_MATH)
    constexpr uint32_t kBody = kGpMathBody;
#else
    constexpr uint32_t kBody = kGpPkBody;
#endif

}  // namespace

void kernel_main() {
    km::Compute cp;
    const auto plan = [&](uint32_t i) {
        return cp.arg(kCorePlan + i);
    };
    const bool included = cp.arg(kCoreRank) != 0;
    const uint32_t bm = plan(kPlBm), bn = plan(kPlBn), kb = plan(kPlKb);
    const uint32_t nk = plan(kPlNk), kb_last = plan(kPlKbLast);
    const uint32_t nb = plan(kPlNbh) * plan(kPlNbw);
    const uint32_t sh = plan(kPlSh), sw = plan(kPlSw), sub = sh * sw;
    const uint32_t pa = bm * kb, pw = kb * bn, block_tiles = bm * bn;
    const uint32_t wa = cp.arg(kCoreWords), ww = cp.arg(kCoreWords + 1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        kCbRow, kCbCol, nk > 1 ? kCbPart : kCbGridOut
    );
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_tile_address(kCbGridProf, 0));
    cp.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerGrid || !included) {
            return;
        }
        const uint32_t t0 = km::wall();
        const bool broadcasts = r.a3 != kGridProduct;
        const bool math = r.a3 != kGridBroadcasts;
        uint32_t wait_a = 0, wait_w = 0, wait_part = 0, mm = 0, wait_out = 0, pack = 0;
        if (math) {
            matmul_block_init(kCbRow, kCbCol, 0, sw, sh, kb);
        }
        bool ok = true;  // the same on all three threads (wait_front tells them)
        uint32_t taken = 0;
        for (uint32_t blk = 0; ok && blk < nb; ++blk) {
            if (math && nk > 1) {
                pack_reconfig_data_format(kCbPart);
            }
            for (uint32_t kblk = 0; ok && kblk < nk; ++kblk) {
                const bool last = kblk == nk - 1;
                const uint32_t inner = last ? kb_last : kb;
                if (broadcasts) {
                    const uint32_t t1 = km::wall();
                    ok = km::broadcast_receive(cp, kCbRow, pa);
                    const uint32_t t2 = km::wall();
                    ok = ok && km::broadcast_receive(cp, kCbCol, pw);
                    wait_a += t2 - t1;
                    wait_w += km::wall() - t2;
                    taken += ok;
                }
                for (uint32_t i = 0; ok && math && i < bm; i += sh) {
                    for (uint32_t j = 0; ok && j < bn; j += sw) {
                        const bool reload = last && nk > 1;
                        if (reload) {
                            const uint32_t t1 = km::wall();
                            ok = cp.wait_front(kCbPart, sub);
                            wait_part += km::wall() - t1;
                            if (!ok) {
                                break;
                            }
                        }
                        const uint32_t t1 = km::wall();
                        tile_regs_acquire();
                        if (reload) {
                            reconfig_data_format_srca(kCbCol, kCbPart);
                            copy_init(kCbPart);
                            copy_block(kCbPart, 0, 0, sub);
                            cb_pop_front(kCbPart, sub);
                            reconfig_data_format_srca(kCbPart, kCbCol);
                            matmul_block_init(kCbRow, kCbCol, 0, sw, sh, kb);
                        }
                        uint32_t a_index = i * kb, w_index = j;
                        for (uint32_t k = 0; k < inner; ++k) {
                            matmul_block(
                                kCbRow, kCbCol, a_index, w_index, 0, 0, sw, sh, kb
                            );
                            ++a_index;
                            w_index += bn;
                        }
                        tile_regs_commit();
                        const uint32_t t2 = km::wall();
                        const uint32_t cb = last ? kCbGridOut : kCbPart;
                        const bool room = cp.reserve_back(cb, sub);
                        const uint32_t t3 = km::wall();
                        tile_regs_wait();
                        if (room) {
                            if (last && nk > 1) {
                                pack_reconfig_data_format(kCbGridOut);
                                pack_reconfig_l1_acc(0);
                            } else if (!last) {
                                pack_reconfig_l1_acc(kblk > 0 ? 1 : 0);
                            }
                            for (uint32_t t = 0; t < sub; ++t) {
                                pack_tile(t, cb);
                            }
                        }
                        tile_regs_release();
                        if (room) {
                            cb_push_back(cb, sub);
                        }
                        mm += t2 - t1;
                        wait_out += t3 - t2;
                        pack += km::wall() - t3;
                    }
                }
                if (ok && math && !last && kblk + 2 < nk) {
                    const uint32_t t1 = km::wall();
                    ok = cp.wait_front(kCbPart, block_tiles);
                    wait_part += km::wall() - t1;
                    if (ok) {
                        cb_pop_front(kCbPart, block_tiles);
                    }
                }
                if (ok && broadcasts) {
                    km::broadcast_done(kCbRow, pa);
                    km::broadcast_done(kCbCol, pw);
                }
            }
            if (math && nk > 1) {
                pack_reconfig_l1_acc(0);
            }
        }
        if (ok && broadcasts) {
            ok = (
                km::broadcast_end(cp, kCbRow, pa, wa, taken) &&
                km::broadcast_end(cp, kCbCol, pw, ww, taken)
            );
        }
        UNPACK((km::add_sat(prof + kGpUnWaitA, wait_a)));
        UNPACK((km::add_sat(prof + kGpUnWaitW, wait_w)));
        UNPACK((km::add_sat(prof + kGpUnWaitPart, wait_part)));
        MATH((km::add_sat(prof + kGpMathMm, mm)));
        PACK((km::add_sat(prof + kGpPkWaitOut, wait_out)));
        PACK((km::add_sat(prof + kGpPkPack, pack)));
        km::add_sat(prof + kBody, km::wall() - t0);
    });
}
