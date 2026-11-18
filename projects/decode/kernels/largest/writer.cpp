// Largest, writer: in place of the block's write, its row 0's largest bf16 bits and
// first column below a9 >> 16 (words 0, 1) as page part (a9 & 0xFFFF) x blocks + block.
#include "api/dataflow/dataflow_api.h"
#include "../ops.hpp"

// bf16 bits as a key that orders as the values do.
static inline uint32_t ordered(uint32_t bits) {
    return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}

uint32_t km_overlay(km::Writer& writer, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const km::Tensor<0> output(command.a2);
    const uint32_t block_width = writer.arg(block.plan + kSpOw);
    const uint32_t tiles = writer.arg(block.plan + kSpOh) * block_width;
    const uint32_t blocks = writer.arg(block.plan + kSpNt) / block_width;
    const uint32_t part = command.a9 & 0xFFFFu;
    const uint32_t limit = command.a9 >> 16;
    if (!km::reduce_result(writer, kCbSplitOut, tiles)) {
        return 0;
    }
    const uint32_t source = get_read_ptr(kCbSplitOut);
    volatile uint16_t* values = reinterpret_cast<volatile uint16_t*>(source);
    uint32_t best_key = 0;
    uint32_t best_bits = 0;
    uint32_t best = 0xFFFFFFFFu;
    for (uint32_t tile = 0; tile < block.columns; tile++) {
        for (uint32_t column = 0; column < 32; column++) {
            const uint32_t at = (block.first_column + tile) * 32 + column;
            if (at >= limit) {
                break;
            }
            // row 0 of a tile: face 0's first 16, then face 1's (256 values on)
            const uint32_t bits = (
                values
                    [
                        tile * (kTileBytes / 2) +
                        (column < 16 ? column : 256 + column - 16)
                    ]
            );
            const uint32_t key = ordered(bits);
            if (best == 0xFFFFFFFFu || key > best_key) {
                best_key = key;
                best_bits = bits;
                best = at;
            }
        }
    }
    volatile uint32_t* page = reinterpret_cast<volatile uint32_t*>(source);
    page[0] = best_bits;
    page[1] = best == 0xFFFFFFFFu ? 0xFFFFFFFFu : best;
    if (!km::write_pages(
        writer,
        command,
        output,
        part * blocks + block.first_column / block_width,
        1,
        source
    ) ||
        !flushed(writer, km::kStWriter, km::kFlushSent)) {
        return 0;
    }
    cb_pop_front(kCbSplitOut, tiles);
    return 1;
}
