// Linear compute: the grid product over this island's K, bf16 when that K is all of
// it, else fp32 partial sums; then this core's share of Y summed over the islands.
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
    constexpr uint32_t kYPlace = 0;
    constexpr uint32_t kWPlace = (
        km::Placement<kYPlace>::next_compile_time_args_offset()
    );
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
        const uint32_t k_whole = command.a4;
        const Cover mine = cover<kWPlace>(km::island(), plan(kPlNt));
        const Parts held_by = parts<kWPlace>(k_whole, plan(kPlNt));
        const bool whole = held_by.whole;
        const uint32_t out = whole ? kCbYOut : kCbPartOut;
        const uint32_t k_blocks = blocks_of(mine.k_tiles, k_block_tiles);
        const uint32_t last_k_block_tiles = (
            mine.k_tiles - (k_blocks - 1) * k_block_tiles
        );
        bool ok = true;  // the same on all three threads (wait_front tells them)
        if (included && mine.k_tiles != 0) {
            matmul_block_init(
                kCbRow, kCbCol, 0, subblock_width, subblock_height, k_block_tiles
            );
            uint32_t taken = 0;
            for (uint32_t block = 0; ok && block < block_count; ++block) {
                pack_reconfig_data_format(k_blocks > 1 ? kCbPart : out);
                for (uint32_t k_block = 0; ok && k_block < k_blocks; ++k_block) {
                    const bool last = k_block == k_blocks - 1;
                    const uint32_t inner = last ? last_k_block_tiles : k_block_tiles;
                    ok = (
                        km::broadcast_receive(compute, kCbRow, a_block_tiles) &&
                        km::broadcast_receive(compute, kCbCol, w_block_tiles)
                    );
                    taken += ok;
                    for (
                        uint32_t subblock_row = 0; ok && subblock_row < block_height;
                        subblock_row += subblock_height
                    ) {
                        for (
                            uint32_t subblock_column = 0;
                            ok && subblock_column < block_width;
                            subblock_column += subblock_width
                        ) {
                            const bool reload = last && k_blocks > 1;
                            if (reload) {
                                ok = compute.wait_front(kCbPart, subblock_tiles);
                                if (!ok) {
                                    break;
                                }
                            }
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
                            const uint32_t cb = last ? out : kCbPart;
                            const bool room = compute.reserve_back(cb, subblock_tiles);
                            tile_regs_wait();
                            if (room) {
                                if (last && k_blocks > 1) {
                                    pack_reconfig_data_format(out);
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
                        }
                    }
                    if (ok && !last && k_block + 2 < k_blocks) {
                        ok = compute.wait_front(kCbPart, block_tiles);
                        if (ok) {
                            cb_pop_front(kCbPart, block_tiles);
                        }
                    }
                    if (ok) {
                        km::broadcast_done(kCbRow, a_block_tiles);
                        km::broadcast_done(kCbCol, w_block_tiles);
                    }
                }
                if (k_blocks > 1) {
                    pack_reconfig_l1_acc(0);
                }
            }
            ok = (
                ok &&
                (
                    km::broadcast_end(
                        compute, kCbRow, a_block_tiles, row_window, taken
                    ) &&
                    km::broadcast_end(
                        compute, kCbCol, w_block_tiles, column_window, taken
                    )
                )
            );
        }
        if (!ok || whole) {
            return;
        }
        // This core's share of Y: each holder's partial sum of a tile in fp32 (the
        // first in DST 0, the next in DST 1, added), packed bf16.
        const km::Group local = km::group_of(command, compute.me()).local();
        const Span y_here = held<kYPlace>(km::island());
        const km::Partition share = km::partition(y_here.pages, local.rank, local.size);
        const uint32_t sums = __builtin_popcount(held_by.holders);
        reconfig_data_format_srca(kCbCol, kCbSums);
        pack_reconfig_data_format(kCbYOut);
        for (uint32_t done = 0; ok && done < share.len; done += kChunk) {
            const uint32_t count = (
                share.len - done < kChunk ? share.len - done : kChunk
            );
            ok = compute.wait_front(kCbSums, kChunk * sums, km::kBehindPort);
            for (uint32_t tile = 0; ok && tile < count; ++tile) {
                tile_regs_acquire();
                copy_init(kCbSums);
                copy_tile(kCbSums, tile, 0);
                for (uint32_t sum = 1; sum < sums; ++sum) {
                    copy_init(kCbSums);
                    copy_tile(kCbSums, sum * kChunk + tile, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
                tile_regs_commit();
                ok = compute.reserve_back(kCbYOut, 1);
                tile_regs_wait();
                if (ok) {
                    pack_tile(0, kCbYOut);
                }
                tile_regs_release();
                if (ok) {
                    cb_push_back(kCbYOut, 1);
                }
            }
            if (ok) {
                cb_pop_front(kCbSums, kChunk * sums);
            }
        }
        reconfig_data_format_srca(kCbSums, kCbCol);
    });
}
