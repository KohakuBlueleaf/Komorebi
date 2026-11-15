// Gate, writer: the root's block of the output (a2, half as wide as the sum: a block
// of gate_up gives GATE_TILES tiles of it), as compute packs it, in place of Y's write.
#include "api/dataflow/dataflow_api.h"
#include "../ops.hpp"

uint32_t km_overlay(km::Writer& writer, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const km::Tensor<0> output(command.a2);
    const uint32_t block_width = writer.arg(block.plan + kSpOw);
    const uint32_t tiles = writer.arg(block.plan + kSpOh) * block_width;
    const uint32_t n_tiles = writer.arg(block.plan + kSpNt) / 2;
    if (!km::reduce_result(writer, kCbSplitOut, tiles)) {
        return 0;
    }
    const uint32_t source = get_read_ptr(kCbSplitOut);
    for (uint32_t row = 0; row < block.rows; row++) {
        const uint32_t first = (
            (block.first_row + row) * n_tiles + block.first_column / 2
        );
        if (!km::write_pages(
            writer,
            command,
            output,
            first,
            block.columns / 2,
            source + row * block_width * kTileBytes
        )) {
            return 0;
        }
    }
    if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
        return 0;
    }
    cb_pop_front(kCbSplitOut, tiles);
    return 1;
}
