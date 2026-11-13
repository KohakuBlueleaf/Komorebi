// Grid compute: A blocks from the row broadcast, W blocks from the column broadcast,
// summed into fp32 partials (kCbPart) by the packer; the last K block packs bf16.
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
    km::Compute compute;
    const auto plan = [&](uint32_t word) {
        return compute.arg(kCorePlan + word);
    };
    const bool included = compute.arg(kCoreRank) != 0;
    const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
    const uint32_t k_block_tiles = plan(kPlKb);
    const uint32_t k_blocks = plan(kPlNk), last_k_block_tiles = plan(kPlKbLast);
    const uint32_t block_count = plan(kPlNbh) * plan(kPlNbw);
    const uint32_t subblock_height = plan(kPlSh), subblock_width = plan(kPlSw);
    const uint32_t subblock_tiles = subblock_height * subblock_width;
    const uint32_t a_block_tiles = block_height * k_block_tiles;
    const uint32_t w_block_tiles = k_block_tiles * block_width;
    const uint32_t block_tiles = block_height * block_width;
    const uint32_t row_window = compute.arg(kCoreWords);
    const uint32_t column_window = compute.arg(kCoreWords + 1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        kCbRow, kCbCol, k_blocks > 1 ? kCbPart : kCbGridOut
    );
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_tile_address(kCbGridProf, 0));
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerGrid || !included) {
            return;
        }
        const uint32_t started = km::wall();
        const uint32_t mode = command.a3;
        const bool broadcasts = mode != kGridProduct;
        const bool math = mode != kGridBroadcasts;
        uint32_t wait_a = 0, wait_w = 0, wait_part = 0, matmul_cycles = 0;
        uint32_t wait_out = 0, pack = 0;
        if (math) {
            matmul_block_init(
                kCbRow, kCbCol, 0, subblock_width, subblock_height, k_block_tiles
            );
        }
        bool ok = true;  // the same on all three threads (wait_front tells them)
        uint32_t taken = 0;
        for (uint32_t block = 0; ok && block < block_count; ++block) {
            if (math && k_blocks > 1) {
                pack_reconfig_data_format(kCbPart);
            }
            for (uint32_t k_block = 0; ok && k_block < k_blocks; ++k_block) {
                const bool last = k_block == k_blocks - 1;
                const uint32_t inner = last ? last_k_block_tiles : k_block_tiles;
                if (broadcasts) {
                    const uint32_t asked = km::wall();
                    ok = km::broadcast_receive(compute, kCbRow, a_block_tiles);
                    const uint32_t a_arrived = km::wall();
                    ok = ok && km::broadcast_receive(compute, kCbCol, w_block_tiles);
                    wait_a += a_arrived - asked;
                    wait_w += km::wall() - a_arrived;
                    taken += ok;
                }
                for (
                    uint32_t subblock_row = 0;
                    ok && math && subblock_row < block_height;
                    subblock_row += subblock_height
                ) {
                    for (
                        uint32_t subblock_column = 0;
                        ok && subblock_column < block_width;
                        subblock_column += subblock_width
                    ) {
                        const bool reload = last && k_blocks > 1;
                        if (reload) {
                            const uint32_t asked = km::wall();
                            ok = compute.wait_front(kCbPart, subblock_tiles);
                            wait_part += km::wall() - asked;
                            if (!ok) {
                                break;
                            }
                        }
                        const uint32_t subblock_started = km::wall();
                        tile_regs_acquire();
                        if (reload) {
                            reconfig_data_format_srca(kCbCol, kCbPart);
                            copy_init(kCbPart);
                            copy_block(kCbPart, 0, 0, subblock_tiles);
                            cb_pop_front(kCbPart, subblock_tiles);
                            reconfig_data_format_srca(kCbPart, kCbCol);
                            matmul_block_init(
                                kCbRow,
                                kCbCol,
                                0,
                                subblock_width,
                                subblock_height,
                                k_block_tiles
                            );
                        }
                        uint32_t a_index = subblock_row * k_block_tiles;
                        uint32_t w_index = subblock_column;
                        for (uint32_t k_tile = 0; k_tile < inner; ++k_tile) {
                            matmul_block(
                                kCbRow,
                                kCbCol,
                                a_index,
                                w_index,
                                0,
                                0,
                                subblock_width,
                                subblock_height,
                                k_block_tiles
                            );
                            ++a_index;
                            w_index += block_width;
                        }
                        tile_regs_commit();
                        const uint32_t product_issued = km::wall();
                        const uint32_t cb = last ? kCbGridOut : kCbPart;
                        const bool room = compute.reserve_back(cb, subblock_tiles);
                        const uint32_t output_reserved = km::wall();
                        tile_regs_wait();
                        if (room) {
                            if (last && k_blocks > 1) {
                                pack_reconfig_data_format(kCbGridOut);
                                pack_reconfig_l1_acc(0);
                            } else if (!last) {
                                pack_reconfig_l1_acc(k_block > 0 ? 1 : 0);
                            }
                            for (uint32_t tile = 0; tile < subblock_tiles; ++tile) {
                                pack_tile(tile, cb);
                            }
                        }
                        tile_regs_release();
                        if (room) {
                            cb_push_back(cb, subblock_tiles);
                        }
                        matmul_cycles += product_issued - subblock_started;
                        wait_out += output_reserved - product_issued;
                        pack += km::wall() - output_reserved;
                    }
                }
                if (ok && math && !last && k_block + 2 < k_blocks) {
                    const uint32_t asked = km::wall();
                    ok = compute.wait_front(kCbPart, block_tiles);
                    wait_part += km::wall() - asked;
                    if (ok) {
                        cb_pop_front(kCbPart, block_tiles);
                    }
                }
                if (ok && broadcasts) {
                    km::broadcast_done(kCbRow, a_block_tiles);
                    km::broadcast_done(kCbCol, w_block_tiles);
                }
            }
            if (math && k_blocks > 1) {
                pack_reconfig_l1_acc(0);
            }
        }
        if (ok && broadcasts) {
            ok = (
                km::broadcast_end(compute, kCbRow, a_block_tiles, row_window, taken) &&
                km::broadcast_end(compute, kCbCol, w_block_tiles, column_window, taken)
            );
        }
        UNPACK((km::add_sat(profile + kGpUnWaitA, wait_a)));
        UNPACK((km::add_sat(profile + kGpUnWaitW, wait_w)));
        UNPACK((km::add_sat(profile + kGpUnWaitPart, wait_part)));
        MATH((km::add_sat(profile + kGpMathMm, matmul_cycles)));
        PACK((km::add_sat(profile + kGpPkWaitOut, wait_out)));
        PACK((km::add_sat(profile + kGpPkPack, pack)));
        km::add_sat(profile + kBody, km::wall() - started);
    });
}
