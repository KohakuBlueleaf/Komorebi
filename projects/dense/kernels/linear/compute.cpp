// Linear compute: the grid product over this island's K for its own window of
// columns, bf16 (LINEAR_ADD: then each block's two partials summed in fp32).
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
#include "komorebi/device/movement/km_broadcast.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "komorebi/device/km_worker.hpp"
#include "linear.h"

#if defined(LINEAR_SPLIT)  // the K-split plan (linear.h): its own body
#include "ksplit_compute.hpp"
#else

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
        const uint32_t x = command.a0;
        if (!included || mine.count == 0) {
            return;
        }
        bool ok = true;  // the same on all three threads (wait_front tells them)
        // the account: each thread its own counts, its words stored at the end
        LINEAR_ACCOUNT_DO(uint32_t account[kAcComputeStart - kAcInputWait + 1] = {});
        LINEAR_ACCOUNT_DO(account[kAcComputeStart - kAcInputWait] = km::wall());
        LINEAR_ACCOUNT_DO(uint32_t since = 0);
        matmul_block_init(
            kCbRow, kCbCol, 0, subblock_width, subblock_height, k_block_tiles
        );
        uint32_t taken = 0;
        // this island's own columns of the rows it makes (none: nothing to make)
        const bool own = (
            window(x, w, y, here, here, m_tiles, k_whole, n_whole).count != 0
        );
        const uint32_t out = kCbYOut;
        for (uint32_t block = 0; ok && own && block < block_count; ++block) {
            // over one K block: the partials reserved once, each K block added in
            // L1 (from the second), then copied to bf16 a subblock at a time
            const bool partial = k_blocks > 1;
            pack_reconfig_data_format(partial ? kCbPart : out);
            pack_reconfig_l1_acc(0);
            const bool held = (!partial || compute.reserve_back(kCbPart, block_tiles));
            for (uint32_t k_block = 0; ok && k_block < k_blocks; ++k_block) {
                const bool last = k_block == k_blocks - 1;
                const uint32_t inner = last ? last_k_block_tiles : k_block_tiles;
                LINEAR_ACCOUNT_DO(since = km::wall());
                ok = (
                    km::broadcast_receive(compute, kCbRow, a_block_tiles) &&
                    km::broadcast_receive(compute, kCbCol, w_block_tiles)
                );
                LINEAR_ACCOUNT_DO(lap(account[0], since));
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
                        tile_regs_acquire();
                        LINEAR_ACCOUNT_DO(since = km::wall());
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
                        LINEAR_ACCOUNT_DO(
                            account
                                [kAcProducts - kAcInputWait] += (inner * subblock_tiles)
                        );
                        if (partial) {
                            tile_regs_wait();
                            LINEAR_ACCOUNT_DO(
                                lap(account[kAcMath - kAcInputWait], since)
                            );
                            for (
                                uint32_t tile = 0; held && tile < subblock_tiles; ++tile
                            ) {
                                pack_tile<true>(tile, kCbPart, at + tile);
                            }
                            tile_regs_release();
                            LINEAR_ACCOUNT_DO(
                                lap(account[kAcPack - kAcInputWait], since)
                            );
                        } else {
                            const bool room = compute.reserve_back(out, subblock_tiles);
                            tile_regs_wait();
                            LINEAR_ACCOUNT_DO(
                                lap(account[kAcMath - kAcInputWait], since)
                            );
                            for (
                                uint32_t tile = 0; room && tile < subblock_tiles; ++tile
                            ) {
                                pack_tile(tile, out);
                            }
                            tile_regs_release();
                            LINEAR_ACCOUNT_DO(
                                lap(account[kAcPack - kAcInputWait], since)
                            );
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
                LINEAR_ACCOUNT_DO(since = km::wall());
                tile_regs_acquire();
                copy_block(kCbPart, tile, 0, subblock_tiles);
                tile_regs_commit();
                const bool room = compute.reserve_back(out, subblock_tiles);
                tile_regs_wait();
                for (uint32_t index = 0; room && index < subblock_tiles; ++index) {
                    pack_tile(index, out);
                }
                tile_regs_release();
                LINEAR_ACCOUNT_DO(lap(account[kAcPack - kAcInputWait], since));
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
        ok = (
            ok &&
            (
                km::broadcast_end(compute, kCbRow, a_block_tiles, row_window, taken) &&
                km::broadcast_end(compute, kCbCol, w_block_tiles, column_window, taken)
            )
        );
        // LINEAR_ADD (two islands, Y a copy): a block's two partials, a subblock of
        // each in turn, summed in fp32 (the second packed on the first), bf16 to Y
        if (ok && LINEAR_ADDS && islands == 2 && copied(y)) {
            reconfig_data_format_srca(kCbCol, kCbSent);
            for (uint32_t block = 0; ok && block < block_count; ++block) {
                LINEAR_ACCOUNT_DO(since = km::wall());
                copy_init(kCbSent);
                pack_reconfig_data_format(kCbPart);
                const bool held = compute.reserve_back(kCbPart, block_tiles);
                for (uint32_t at = 0; ok && at < block_tiles; at += subblock_tiles) {
                    for (uint32_t half = 0; ok && half < 2; ++half) {
                        ok = (
                            compute.wait_front(kCbSent, subblock_tiles, km::kBehindPort)
                        );
                        if (!ok) {
                            break;
                        }
                        pack_reconfig_l1_acc(half);
                        tile_regs_acquire();
                        copy_block(kCbSent, 0, 0, subblock_tiles);
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t tile = 0; held && tile < subblock_tiles; ++tile) {
                            pack_tile<true>(tile, kCbPart, at + tile);
                        }
                        tile_regs_release();
                        cb_pop_front(kCbSent, subblock_tiles);
                    }
                }
                pack_reconfig_l1_acc(0);
                if (!ok) {
                    break;
                }
                if (held) {
                    cb_push_back(kCbPart, block_tiles);
                }
                ok = compute.wait_front(kCbPart, block_tiles);
                if (!ok) {
                    break;
                }
                reconfig_data_format_srca(kCbSent, kCbPart);
                copy_init(kCbPart);
                pack_reconfig_data_format(kCbYOut);
                for (uint32_t tile = 0; tile < block_tiles; tile += subblock_tiles) {
                    tile_regs_acquire();
                    copy_block(kCbPart, tile, 0, subblock_tiles);
                    tile_regs_commit();
                    const bool room = compute.reserve_back(kCbYOut, subblock_tiles);
                    tile_regs_wait();
                    for (uint32_t index = 0; room && index < subblock_tiles; ++index) {
                        pack_tile(index, kCbYOut);
                    }
                    tile_regs_release();
                    if (room) {
                        cb_push_back(kCbYOut, subblock_tiles);
                    }
                }
                cb_pop_front(kCbPart, block_tiles);
                reconfig_data_format_srca(kCbPart, kCbSent);
                LINEAR_ACCOUNT_DO(lap(account[kAcPack - kAcInputWait], since));
            }
            reconfig_data_format_srca(kCbSent, kCbCol);
        }
#if defined(LINEAR_ACCOUNT) && !defined(TRISC_MATH)  // the math thread: no buffers
        account[kAcComputeEnd - kAcInputWait] = km::wall();
        volatile uint32_t* stamps = stamp_row();
        UNPACK((stamps[kAcInputWait] = account[0]));
        PACK((stamps[kAcMath] = account[kAcMath - kAcInputWait]));
        PACK((stamps[kAcProducts] = account[kAcProducts - kAcInputWait]));
        PACK((stamps[kAcPack] = account[kAcPack - kAcInputWait]));
        PACK((stamps[kAcComputeEnd] = account[kAcComputeEnd - kAcInputWait]));
        PACK((stamps[kAcComputeStart] = account[kAcComputeStart - kAcInputWait]));
#endif
    });
}
#endif  // LINEAR_SPLIT
