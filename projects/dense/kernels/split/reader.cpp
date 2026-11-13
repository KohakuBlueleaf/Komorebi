// K-split reader, NoC0: this core's 1/S of K for its group's output block, A and W in
// chunks of kc K tiles; then the parts this core sums (by owner or by tree).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "split.h"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto w_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    km::Reader reader;
    const auto plan = [&](uint32_t word) {
        return reader.arg(word);
    };
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerSplit) {
            return;
        }
        const uint32_t a_address = command.a0, w_address = command.a1;
        const uint32_t reduce_form = command.a3, mode = command.a4;
        const uint32_t m_tiles = plan(kSpMt);
        const uint32_t k_tiles = plan(kSpKt);
        const uint32_t n_tiles = plan(kSpNt);
        const uint32_t block_height = plan(kSpOh), block_width = plan(kSpOw);
        const uint32_t chunk_tiles = plan(kSpKc);
        const uint32_t blocks_per_row = plan(kSpNbw), group_size = plan(kSpS);
        const km::Group group = km::digit_group(command, reader.me(), 1, group_size);
        const uint32_t group_index = (reader.me() - km::first(command)) / group_size;
        const uint32_t first_row = group_index / blocks_per_row * block_height;
        const uint32_t first_column = group_index % blocks_per_row * block_width;
        const km::Partition k_part = km::partition(k_tiles, group.rank, group.size);
        const auto a_matrix = TensorAccessor(a_args, a_address);
        const auto w_matrix = TensorAccessor(w_args, w_address);
        for (uint32_t offset = 0; offset < k_part.len; offset += chunk_tiles) {
            const uint32_t chunk_length = (
                k_part.len - offset < chunk_tiles ? k_part.len - offset : chunk_tiles
            );
            const uint32_t k_start = k_part.start + offset;
            if (
                !reader.reserve(kCbSa, block_height * chunk_tiles) ||
                !reader.reserve(kCbSw, chunk_tiles * block_width)
            ) {
                return;
            }
            const uint32_t a_destination = get_write_ptr(kCbSa);
            const uint32_t w_destination = get_write_ptr(kCbSw);
            for (
                uint32_t tile_row = 0;
                tile_row < block_height && first_row + tile_row < m_tiles;
                ++tile_row
            ) {
                for (uint32_t k_tile = 0; k_tile < chunk_length; ++k_tile) {
                    noc_async_read_page(
                        (first_row + tile_row) * k_tiles + k_start + k_tile,
                        a_matrix,
                        a_destination + (tile_row * chunk_tiles + k_tile) * kTileBytes
                    );
                }
            }
            for (uint32_t k_tile = 0; k_tile < chunk_length; ++k_tile) {
                for (
                    uint32_t tile_column = 0;
                    tile_column < block_width && first_column + tile_column < n_tiles;
                    ++tile_column
                ) {
                    noc_async_read_page(
                        (k_start + k_tile) * n_tiles + first_column + tile_column,
                        w_matrix,
                        w_destination +
                            (k_tile * block_width + tile_column) * kTileBytes
                    );
                }
            }
            if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                return;
            }
            cb_push_back(kCbSa, block_height * chunk_tiles);
            cb_push_back(kCbSw, chunk_tiles * block_width);
        }
        if (mode == kSplitFull) {
            km::reduce_arrive(
                reader,
                command,
                {kCbArrive, plan(kSpArrive), block_height * block_width},
                group,
                reduce_form
            );
        }
    });
}
