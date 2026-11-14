// K-split writer, NoC1 (mover B): its K chunks (chunks.hpp); then a non-root core sends
// its part (fp32) on, the group's root writes the sum (bf16) to Y.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "chunks.hpp"
#include "komorebi/device/km_port.hpp"
#include "split.h"

void kernel_main() {
    using Y = km::Tensor<0>;
    using A = km::Tensor<Y::next_compile_time_args_offset()>;
    using W = km::Tensor<A::next_compile_time_args_offset()>;
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        const uint32_t y_address = command.a2, reduce_form = command.a3;
        const uint32_t mode = command.a4;
        if (km::handler(command) != kHandlerSplit) {
            return true;
        }
        if (command.a5 >= kSplitPlans) {  // no such plan: the command is not completed
            return false;
        }
        const auto plan = [&](uint32_t word) {
            return writer.arg(plan_words(command) + word);
        };
        if (!move_chunks(writer, command, A(command.a0), W(command.a1))) {
            return false;
        }
        if (mode != kSplitFull) {
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
        const Y y_matrix(y_address);
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
                if (!km::write_pages(
                    writer,
                    command,
                    y_matrix,
                    (first_row + tile_row) * n_tiles + first_column + tile_column,
                    1,
                    source + (tile_row * block_width + tile_column) * kTileBytes
                )) {
                    return false;
                }
            }
        }
        if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
            return false;
        }
        cb_pop_front(kCbSplitOut, block_tiles);
        return true;
    });
}
