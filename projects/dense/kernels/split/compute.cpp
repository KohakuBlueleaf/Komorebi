// K-split compute: this core's part of the oh x ow block summed over its K chunks into
// kCbSum (fp32), then the parts received in rank order; the root packs it in bf16.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_port.hpp"
#include "split.h"

namespace {

    // Tile `page` of fp32 buffer `source_buffer` into tile `target_tile` of buffer `cb`
    // (accumulated when the packer's L1 accumulation is on), through DST 0.
    void move_tile(
        uint32_t source_buffer, uint32_t page, uint32_t cb, uint32_t target_tile
    ) {
        tile_regs_acquire();
        copy_tile(source_buffer, page, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb, target_tile);
        tile_regs_release();
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const auto plan = [&](uint32_t word) {
        return compute.arg(word);
    };
    const uint32_t k_tiles = plan(kSpKt), block_height = plan(kSpOh);
    const uint32_t block_width = plan(kSpOw);
    const uint32_t chunk_tiles = plan(kSpKc), subblock_height = plan(kSpSh);
    const uint32_t subblock_width = plan(kSpSw), group_size = plan(kSpS);
    const uint32_t block_tiles = block_height * block_width;
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbSa, kCbSw, kCbSum);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerSplit) {
            return;
        }
        const uint32_t reduce_form = command.a3, mode = command.a4;
        const km::Group group = km::digit_group(command, compute.me(), 1, group_size);
        const km::Partition k_part = km::partition(k_tiles, group.rank, group.size);
        const km::Arrival arrival = {kCbArrive, plan(kSpArrive), block_tiles};
        const bool full = mode == kSplitFull;
        const bool room = compute.reserve_back(kCbSum, block_tiles);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        matmul_block_init(
            kCbSa, kCbSw, 0, subblock_width, subblock_height, chunk_tiles
        );
        pack_reconfig_data_format(kCbSum);
        for (uint32_t offset = 0; ok && offset < k_part.len; offset += chunk_tiles) {
            const uint32_t chunk_length = (
                k_part.len - offset < chunk_tiles ? k_part.len - offset : chunk_tiles
            );
            ok = (
                compute.wait_front(kCbSa, block_height * chunk_tiles) &&
                compute.wait_front(kCbSw, chunk_tiles * block_width)
            );
            if (!ok) {
                break;
            }
            pack_reconfig_l1_acc(offset > 0 ? 1 : 0);
            for (
                uint32_t subblock_row = 0; subblock_row < block_height;
                subblock_row += subblock_height
            ) {
                for (
                    uint32_t subblock_column = 0; subblock_column < block_width;
                    subblock_column += subblock_width
                ) {
                    tile_regs_acquire();
                    for (uint32_t k_tile = 0; k_tile < chunk_length; ++k_tile) {
                        matmul_block(
                            kCbSa,
                            kCbSw,
                            subblock_row * chunk_tiles + k_tile,
                            k_tile * block_width + subblock_column,
                            0,
                            0,
                            subblock_width,
                            subblock_height,
                            chunk_tiles
                        );
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (
                        uint32_t tile_row = 0; room && tile_row < subblock_height;
                        ++tile_row
                    ) {
                        for (
                            uint32_t tile_column = 0; tile_column < subblock_width;
                            ++tile_column
                        ) {
                            pack_tile<true>(
                                tile_row * subblock_width + tile_column,
                                kCbSum,
                                (subblock_row + tile_row) * block_width +
                                    subblock_column + tile_column
                            );
                        }
                    }
                    tile_regs_release();
                }
            }
            cb_pop_front(kCbSa, block_height * chunk_tiles);
            cb_pop_front(kCbSw, chunk_tiles * block_width);
        }
        reconfig_data_format_srca(kCbSw, kCbArrive);
        copy_init(kCbArrive);
        if (ok && full) {
            pack_reconfig_l1_acc(1);
            ok = km::reduce(compute, arrival, group, reduce_form, [&](uint32_t slot) {
                for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                    move_tile(
                        kCbArrive, km::slot_page(arrival, slot) + tile, kCbSum, tile
                    );
                }
            });
        }
        pack_reconfig_l1_acc(0);
        if (ok && full && room) {
            cb_push_back(kCbSum, block_tiles);
        }
        if (
            ok && full && group.rank == 0
        ) {  // the group's sum, in bf16, for the writer
            ok = compute.wait_front(kCbSum, block_tiles);
            const bool out = compute.reserve_back(kCbSplitOut, block_tiles);
            pack_reconfig_data_format(kCbSplitOut);
            copy_init(kCbSum);
            for (uint32_t tile = 0; ok && tile < block_tiles; ++tile) {
                tile_regs_acquire();
                copy_tile(kCbSum, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (out) {
                    pack_tile(0, kCbSplitOut);
                }
                tile_regs_release();
            }
            if (ok && out) {
                cb_push_back(kCbSplitOut, block_tiles);
            }
            if (ok) {
                cb_pop_front(kCbSum, block_tiles);
            }
        }
        reconfig_data_format_srca(kCbArrive, kCbSw);
    });
}
