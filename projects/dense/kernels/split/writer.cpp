// K-split writer, NoC1: a core other than its group's root sends its part (fp32) to
// its receiver; the root writes the group's sum (bf16) to Y, tiles outside Y dropped.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "split.h"

void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    km::Writer writer;
    const auto plan = [&](uint32_t word) {
        return writer.arg(word);
    };
    writer.run([&](const km_command_t& command) {
        const uint32_t y_address = command.a2, reduce_form = command.a3;
        const uint32_t mode = command.a4;
        if (km::handler(command) != kHandlerSplit || mode != kSplitFull) {
            return true;
        }
        const uint32_t m_tiles = plan(kSpMt), n_tiles = plan(kSpNt);
        const uint32_t block_height = plan(kSpOh), block_width = plan(kSpOw);
        const uint32_t block_tiles = block_height * block_width;
        const uint32_t blocks_per_row = plan(kSpNbw), group_size = plan(kSpS);
        const km::Group group = km::digit_group(command, writer.me(), 1, group_size);
        const km::Arrival arrival = {kCbArrive, plan(kSpArrive), block_tiles};
        if (group.rank != 0) {
            return (
                km::reduce_emit(
                    writer, command, arrival, group, reduce_form, kCbSum, block_tiles
                )
            );
        }
        if (!km::reduce_result(writer, kCbSplitOut, block_tiles)) {
            return false;
        }
        const uint32_t group_index = (writer.me() - km::first(command)) / group_size;
        const uint32_t first_row = group_index / blocks_per_row * block_height;
        const uint32_t first_column = group_index % blocks_per_row * block_width;
        const auto y_matrix = TensorAccessor(y_args, y_address);
        const uint32_t source = get_read_ptr(kCbSplitOut);
        for (
            uint32_t tile_row = 0;
            tile_row < block_height && first_row + tile_row < m_tiles;
            ++tile_row
        ) {
            for (
                uint32_t tile_column = 0;
                tile_column < block_width && first_column + tile_column < n_tiles;
                ++tile_column
            ) {
                noc_async_write_page(
                    (first_row + tile_row) * n_tiles + first_column + tile_column,
                    y_matrix,
                    source + (tile_row * block_width + tile_column) * kTileBytes
                );
            }
        }
        if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
            return false;
        }
        cb_pop_front(kCbSplitOut, block_tiles);
        return true;
    });
}
