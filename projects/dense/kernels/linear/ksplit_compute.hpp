// Linear compute, the K-split plan: this rank's part over its K chunks (chunk q from
// the buffers of its mover) into kCbKSum (fp32), then the parts it receives; the
// group's first rank packs the sum in bf16.
#include "ksplit.hpp"

namespace {

    // Tile `page` of fp32 buffer `source` into tile `target` of buffer `cb`
    // (accumulated when the packer's L1 accumulation is on), through Dst 0.
    void move_tile(uint32_t source, uint32_t page, uint32_t cb, uint32_t target) {
        tile_regs_acquire();
        copy_tile(source, page, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb, target);
        tile_regs_release();
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const auto plan = [&](uint32_t word) {
        return compute.arg(word);
    };
    const uint32_t block_height = plan(kKsOh), block_width = plan(kKsOw);
    const uint32_t chunk_tiles = plan(kKsKc), subblock_height = plan(kKsSh);
    const uint32_t subblock_width = plan(kKsSw);
    const uint32_t block_tiles = block_height * block_width;
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbKa, kCbKw, kCbKSum);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return;
        }
        const KPlace at = k_place(command, compute.me(), plan, km::island());
        if (!at.any) {
            return;
        }
        uint32_t index = 0;
        const km::Group group = k_group(command, compute.me(), plan(kKsS), index);
        const km::Arrival arrival = {kCbKArrive, plan(kKsArrive), block_tiles};
        const uint32_t form = plan(kKsForm);
        const bool room = compute.reserve_back(kCbKSum, block_tiles);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        matmul_block_init(
            kCbKa, kCbKw, 0, subblock_width, subblock_height, chunk_tiles
        );
        pack_reconfig_data_format(kCbKSum);
        for (
            uint32_t offset = 0, chunk = 0; ok && offset < at.k_count;
            offset += chunk_tiles, chunk++
        ) {
            const uint32_t length = (
                at.k_count - offset < chunk_tiles ? at.k_count - offset : chunk_tiles
            );
            const uint32_t x_half = km::block_half(kKChunksX, kKChunkPattern, chunk);
            const uint32_t w_half = km::block_half(kKChunksW, kKChunkPattern, chunk);
            ok = (
                compute.wait_front(x_half, block_height * chunk_tiles) &&
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
                    for (uint32_t k_tile = 0; k_tile < length; ++k_tile) {
                        matmul_block(
                            x_half,
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
                                kCbKSum,
                                (subblock_row + tile_row) * block_width +
                                    subblock_column + tile_column
                            );
                        }
                    }
                    tile_regs_release();
                }
            }
            cb_pop_front(x_half, block_height * chunk_tiles);
            cb_pop_front(w_half, chunk_tiles * block_width);
        }
        reconfig_data_format_srca(kCbKw, kCbKArrive);
        copy_init(kCbKArrive);
        if (ok) {
            pack_reconfig_l1_acc(1);
            ok = km::reduce(compute, arrival, group, form, [&](uint32_t slot) {
                for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                    move_tile(
                        kCbKArrive, km::slot_page(arrival, slot) + tile, kCbKSum, tile
                    );
                }
            });
        }
        pack_reconfig_l1_acc(0);
        if (ok && room) {
            cb_push_back(kCbKSum, block_tiles);
        }
        if (ok && group.rank == 0) {  // the group's sum, in bf16, for the writer
            ok = compute.wait_front(kCbKSum, block_tiles);
            const bool out = compute.reserve_back(kCbKOut, block_tiles);
            pack_reconfig_data_format(kCbKOut);
            copy_init(kCbKSum);
            for (uint32_t tile = 0; ok && tile < block_tiles; ++tile) {
                tile_regs_acquire();
                copy_tile(kCbKSum, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                if (out) {
                    pack_tile(0, kCbKOut);
                }
                tile_regs_release();
            }
            if (ok && out) {
                cb_push_back(kCbKOut, block_tiles);
            }
            if (ok) {
                cb_pop_front(kCbKSum, block_tiles);
            }
        }
        reconfig_data_format_srca(kCbKArrive, kCbKw);
    });
}
