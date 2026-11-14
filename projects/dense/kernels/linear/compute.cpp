// Linear compute: the grid product over this island's K a window of columns, bf16; in
// its own window the tiles other islands put here start DST of the K blocks they join.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_port.hpp"
#include "komorebi/device/km_worker.hpp"
#include "linear.h"

void kernel_main() {
    km::Compute compute;
    const auto plan = [&](uint32_t word) {
        return compute.arg(kCorePlan + word);
    };
    const bool included = compute.arg(kCoreRank) != 0;
    const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
    const uint32_t k_block_tiles = plan(kPlKb);
    const uint32_t block_count = plan(kPlNbh) * plan(kPlNbw);
    const uint32_t subblock_height = plan(kPlSh), subblock_width = plan(kPlSw);
    const uint32_t subblock_tiles = subblock_height * subblock_width;
    const uint32_t a_block_tiles = block_height * k_block_tiles;
    const uint32_t w_block_tiles = k_block_tiles * block_width;
    const uint32_t block_tiles = block_height * block_width;
    const uint32_t row_window = compute.arg(kCoreWords);
    const uint32_t column_window = compute.arg(kCoreWords + 1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbRow, kCbCol, kCbPart);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return;
        }
        const uint32_t w = command.a1, y = command.a2;
        const uint32_t k_whole = command.a4, n_whole = command.a5;
        const uint32_t here = km::island(), islands = km::islands();
        const uint32_t m_tiles = plan(kPlMt);
        const Span mine = k_part(w, here, k_whole, n_whole);
        const uint32_t k_blocks = blocks_of(mine.count, k_block_tiles);
        const uint32_t last_k_block_tiles = mine.count - (k_blocks - 1) * k_block_tiles;
        // the other islands that put here: sender j's tiles join at K block j
        const uint32_t x = command.a0;
        const uint32_t senders = senders_to(x, w, y, here, m_tiles, k_whole, n_whole);
        if (!included || mine.count == 0) {
            return;
        }
        bool ok = true;  // the same on all three threads (wait_front tells them)
        matmul_block_init(
            kCbRow, kCbCol, 0, subblock_width, subblock_height, k_block_tiles
        );
        uint32_t taken = 0;
        for (uint32_t turn = 1; ok && turn <= islands; ++turn) {
            const uint32_t to = island_at(here, turn);
            if (window(x, y, here, to, m_tiles, k_whole, n_whole).count == 0) {
                continue;
            }
            const bool own = to == here;
            const uint32_t out = own ? kCbYOut : kCbSend;
            for (uint32_t block = 0; ok && block < block_count; ++block) {
                // K blocks over one: every K block's products packed into the fp32
                // partials, reserved once a block, added in L1 from the second on;
                // after the last, the partials to bf16 a subblock at a time
                const bool partial = k_blocks > 1;
                pack_reconfig_data_format(partial ? kCbPart : out);
                pack_reconfig_l1_acc(0);
                const bool held = (
                    !partial || compute.reserve_back(kCbPart, block_tiles)
                );
                for (uint32_t k_block = 0; ok && k_block < k_blocks; ++k_block) {
                    const bool last = k_block == k_blocks - 1;
                    const bool joined = own && k_block < senders;
                    const uint32_t inner = last ? last_k_block_tiles : k_block_tiles;
                    ok = (
                        km::broadcast_receive(compute, kCbRow, a_block_tiles) &&
                        km::broadcast_receive(compute, kCbCol, w_block_tiles)
                    );
                    taken += ok;
                    if (k_block == 1) {
                        pack_reconfig_l1_acc(1);
                    }
                    uint32_t at = 0;  // the subblock's first tile in the partials
                    for (
                        uint32_t subblock_row = 0; ok && subblock_row < block_height;
                        subblock_row += subblock_height
                    ) {
                        for (
                            uint32_t subblock_column = 0;
                            ok && subblock_column < block_width;
                            subblock_column += subblock_width
                        ) {
                            if (joined) {
                                ok = (
                                    compute.wait_front(
                                        kCbSent, subblock_tiles, km::kBehindPort
                                    )
                                );
                            }
                            if (!ok) {
                                break;
                            }
                            tile_regs_acquire();
                            if (joined) {
                                // DST starts from another island's tiles of this
                                // subblock
                                reconfig_data_format_srca(kCbCol, kCbSent);
                                copy_init(kCbSent);
                                copy_block(kCbSent, 0, 0, subblock_tiles);
                                cb_pop_front(kCbSent, subblock_tiles);
                                reconfig_data_format_srca(kCbSent, kCbCol);
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
                            if (partial) {
                                tile_regs_wait();
                                for (
                                    uint32_t tile = 0; held && tile < subblock_tiles;
                                    ++tile
                                ) {
                                    pack_tile<true>(tile, kCbPart, at + tile);
                                }
                                tile_regs_release();
                            } else {
                                const bool room = (
                                    compute.reserve_back(out, subblock_tiles)
                                );
                                tile_regs_wait();
                                for (
                                    uint32_t tile = 0; room && tile < subblock_tiles;
                                    ++tile
                                ) {
                                    pack_tile(tile, out);
                                }
                                tile_regs_release();
                                if (room) {
                                    cb_push_back(out, subblock_tiles);
                                }
                            }
                            at += subblock_tiles;
                        }
                    }
                    if (ok) {
                        km::broadcast_done(kCbRow, a_block_tiles);
                        km::broadcast_done(kCbCol, w_block_tiles);
                    }
                }
                if (!ok || !partial) {
                    continue;
                }
                pack_reconfig_l1_acc(0);
                if (held) {
                    cb_push_back(kCbPart, block_tiles);
                }
                ok = compute.wait_front(kCbPart, block_tiles);
                if (!ok) {
                    break;
                }
                reconfig_data_format_srca(kCbCol, kCbPart);
                copy_init(kCbPart);
                pack_reconfig_data_format(out);
                for (uint32_t tile = 0; tile < block_tiles; tile += subblock_tiles) {
                    tile_regs_acquire();
                    copy_block(kCbPart, tile, 0, subblock_tiles);
                    tile_regs_commit();
                    const bool room = compute.reserve_back(out, subblock_tiles);
                    tile_regs_wait();
                    for (uint32_t index = 0; room && index < subblock_tiles; ++index) {
                        pack_tile(index, out);
                    }
                    tile_regs_release();
                    if (room) {
                        cb_push_back(out, subblock_tiles);
                    }
                }
                cb_pop_front(kCbPart, block_tiles);
                reconfig_data_format_srca(kCbPart, kCbCol);
                matmul_block_init(
                    kCbRow, kCbCol, 0, subblock_width, subblock_height, k_block_tiles
                );
            }
        }
        ok = (
            ok &&
            (
                km::broadcast_end(compute, kCbRow, a_block_tiles, row_window, taken) &&
                km::broadcast_end(compute, kCbCol, w_block_tiles, column_window, taken)
            )
        );
    });
}
