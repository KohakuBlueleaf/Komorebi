// Add, writer: the root's block of Y (a2), as compute packs it, in place of Y's write;
// then its rows' sums of squares into tile `group` of a9.
#include "api/dataflow/dataflow_api.h"
#include "../ops.hpp"

uint32_t km_overlay(km::Writer& writer, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const km::Tensor<0> output(command.a2);
    const km::Tensor<0> squares(command.a9);
    const uint32_t block_height = writer.arg(block.plan + kSpOh);
    const uint32_t block_width = writer.arg(block.plan + kSpOw);
    const uint32_t tiles = block_height * block_width;
    const uint32_t n_tiles = writer.arg(block.plan + kSpNt);
    const uint32_t group = (
        block.first_row / block_height * writer.arg(block.plan + kSpNbw) +
        block.first_column / block_width
    );
    if (!km::reduce_result(writer, kCbSplitOut, tiles)) {
        return 0;
    }
    const uint32_t source = get_read_ptr(kCbSplitOut);
    for (uint32_t row = 0; row < block.rows; row++) {
        if (!km::write_pages(
            writer,
            command,
            output,
            (block.first_row + row) * n_tiles + block.first_column,
            block.columns,
            source + row * block_width * kTileBytes
        )) {
            return 0;
        }
    }
    if (
        !writer.wait_front(DC_CB_PART, 1) ||
        !km::write_pages(
            writer, command, squares, group, 1, get_read_ptr(DC_CB_PART)
        ) ||
        !flushed(writer, km::kStWriter, km::kFlushSent)
    ) {
        return 0;
    }
    cb_pop_front(kCbSplitOut, tiles);
    cb_pop_front(DC_CB_PART, 1);
    return 1;
}
