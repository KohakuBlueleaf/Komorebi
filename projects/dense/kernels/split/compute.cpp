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
#include "chunks.hpp"
#include "komorebi/device/km_port.hpp"
#include "sets.hpp"
#include "split.h"
#include "split_op.hpp"

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
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbSa, kCbSw, kCbSum);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerSplit || command.a5 >= kSplitPlans) {
            return;
        }
        const auto plan = [&](uint32_t word) {
            return compute.arg(plan_words(command) + word);
        };
        const uint32_t k_tiles = plan(kSpKt), block_height = plan(kSpOh);
        const uint32_t block_width = plan(kSpOw);
        const uint32_t chunk_tiles = plan(kSpKc), subblock_height = plan(kSpSh);
        const uint32_t subblock_width = plan(kSpSw), group_size = plan(kSpS);
        const uint32_t block_tiles = block_height * block_width;
        const uint32_t reduce_form = split_form(command), mode = command.a4;
        if (mode == kSplitOp) {
            if (split_op_part(command, kOpCompute)) {
                split_op(compute, command);
            }
            return;
        }
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
        for (
            uint32_t offset = 0, chunk = 0; ok && offset < k_part.len;
            offset += chunk_tiles, chunk++
        ) {
            const uint32_t chunk_length = (
                k_part.len - offset < chunk_tiles ? k_part.len - offset : chunk_tiles
            );
            const uint32_t a_half = km::block_half(kChunksA, kChunkPattern, chunk);
            const uint32_t w_half = km::block_half(kChunksW, kChunkPattern, chunk);
            ok = (
                compute.wait_front(a_half, block_height * chunk_tiles) &&
                compute.wait_front(w_half, chunk_tiles * block_width)
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
                            a_half,
                            w_half,
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
            cb_pop_front(a_half, block_height * chunk_tiles);
            cb_pop_front(w_half, chunk_tiles * block_width);
        }
        reconfig_data_format_srca(kCbSw, kCbArrive);
        copy_init(kCbArrive);
        if (ok && full) {
            pack_reconfig_l1_acc(1);
            ok = reduce_set(
                compute, command, arrival, group, reduce_form, [&](uint32_t slot) {
                    for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                        move_tile(
                            kCbArrive, km::slot_page(arrival, slot) + tile, kCbSum, tile
                        );
                    }
                }
            );
        }
        pack_reconfig_l1_acc(0);
        if (ok && full && room) {
            cb_push_back(kCbSum, block_tiles);
        }
        // A root's unpack pops the sum, a sender's writer pops it: roles change with
        // the plan, so unpack counts the writer's pop too (one item: the whole buffer).
        if (ok && full && group.rank != 0) {
            UNPACK((get_local_cb_interface(kCbSum).tiles_acked += block_tiles));
        }
        if (ok && full && group.rank == 0 && split_op_part(command, kOpCompute)) {
            ok = split_op(compute, command);         // the operation takes the sum
        } else if (ok && full && group.rank == 0) {  // the group's sum, in bf16, for
                                                     // the writer
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
