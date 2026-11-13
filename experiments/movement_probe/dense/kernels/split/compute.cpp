// K-split compute: this core's part summed over its K chunks (chunk q from the halves
// of its mover) into kCbSum (fp32), then the parts received; the root packs it in bf16.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "../../../kernels/pattern.hpp"
#include "komorebi/device/km_port.hpp"
#include "split.h"

namespace {

    // Tile `page` of fp32 buffer `from` into tile `to` of buffer `cb` (accumulated when
    // the packer's L1 accumulation is on), through DST 0.
    void move_tile(uint32_t from, uint32_t page, uint32_t cb, uint32_t to) {
        tile_regs_acquire();
        copy_tile(from, page, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb, to);
        tile_regs_release();
    }

}  // namespace

void kernel_main() {
    km::Compute cp;
    const auto plan = [&](uint32_t i) {
        return cp.arg(i);
    };
    const uint32_t kt = plan(kSpKt), oh = plan(kSpOh), ow = plan(kSpOw);
    const uint32_t kc = plan(kSpKc), sh = plan(kSpSh), sw = plan(kSpSw), s = plan(kSpS);
    const uint32_t p = oh * ow;
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbSa, kCbSw, kCbSum);
    cp.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerSplit) {
            return;
        }
        const km::Group g = km::digit_group(r, cp.me(), 1, s);
        const km::Partition part = km::partition(kt, g.rank, g.size);
        const km::Arrival arrival = {kCbArrive, plan(kSpArrive), p};
        const bool full = r.a4 == kSplitFull;
        const bool room = cp.reserve_back(kCbSum, p);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        matmul_block_init(kCbSa, kCbSw, 0, sw, sh, kc);
        pack_reconfig_data_format(kCbSum);
        const movement::Pattern pattern = movement::pattern_of(CHUNK_PATTERN);
        for (uint32_t k0 = 0, chunk = 0; ok && k0 < part.len; k0 += kc, chunk++) {
            const uint32_t n = part.len - k0 < kc ? part.len - k0 : kc;
            const bool mover_a = movement::owner(pattern, chunk) == movement::kMoverA;
            const uint32_t chunk_a = mover_a ? kCbSa : kChunkAMoverB;
            const uint32_t chunk_w = mover_a ? kCbSw : kChunkWMoverB;
            ok = cp.wait_front(chunk_a, oh * kc) && cp.wait_front(chunk_w, kc * ow);
            if (!ok) {
                break;
            }
            pack_reconfig_l1_acc(k0 > 0 ? 1 : 0);
            for (uint32_t i = 0; i < oh; i += sh) {
                for (uint32_t j = 0; j < ow; j += sw) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < n; ++k) {
                        matmul_block(
                            chunk_a, chunk_w, i * kc + k, k * ow + j, 0, 0, sw, sh, kc
                        );
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t a = 0; room && a < sh; ++a) {
                        for (uint32_t b = 0; b < sw; ++b) {
                            pack_tile<true>(a * sw + b, kCbSum, (i + a) * ow + j + b);
                        }
                    }
                    tile_regs_release();
                }
            }
            cb_pop_front(chunk_a, oh * kc);
            cb_pop_front(chunk_w, kc * ow);
        }
        reconfig_data_format_srca(kCbSw, kCbArrive);
        copy_init(kCbArrive);
        if (ok && full) {
            pack_reconfig_l1_acc(1);
            ok = km::reduce(cp, arrival, g, r.a3, [&](uint32_t k) {
                for (uint32_t d = 0; d < p; ++d) {
                    move_tile(kCbArrive, km::slot_page(arrival, k) + d, kCbSum, d);
                }
            });
        }
        pack_reconfig_l1_acc(0);
        if (ok && full && room) {
            cb_push_back(kCbSum, p);
        }
        if (ok && full && g.rank == 0) {  // the group's sum, in bf16, for the writer
            ok = cp.wait_front(kCbSum, p);
            const bool out = cp.reserve_back(kCbSplitOut, p);
            pack_reconfig_data_format(kCbSplitOut);
            copy_init(kCbSum);
            for (uint32_t d = 0; ok && d < p; ++d) {
                tile_regs_acquire();
                copy_tile(kCbSum, d, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (out) {
                    pack_tile(0, kCbSplitOut);
                }
                tile_regs_release();
            }
            if (ok && out) {
                cb_push_back(kCbSplitOut, p);
            }
            if (ok) {
                cb_pop_front(kCbSum, p);
            }
        }
        reconfig_data_format_srca(kCbArrive, kCbSw);
    });
}
