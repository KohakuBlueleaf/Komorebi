// The operation a SPLIT command names after its sum (a6): an overlay that the root of
// each group (in mode kSplitOp every core of the command) loads and calls, once.
#pragma once

#include <stdint.h>

#include "komorebi/device/core/km_group.hpp"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "split.h"

// a6: the overlay's number in bits 0 to 7 (0: no operation), its parts above them.
constexpr uint32_t kOpReader = 0x100;
constexpr uint32_t kOpCompute = 0x200;
constexpr uint32_t kOpWriter = 0x400;

// What a part is called with: the command, the first argument of its plan, and the
// block of Y this core's group holds (its first tile row and column, its tiles).
struct SplitBlock {
    const km_command_t* command;
    uint32_t plan;
    uint32_t first_row;
    uint32_t first_column;
    uint32_t rows;
    uint32_t columns;
};

inline uint32_t split_op_overlay(const km_command_t& command) {
    return command.a6 & 0xFFu;
}

// Whether the command names an operation with a part on this kernel.
inline bool split_op_part(const km_command_t& command, uint32_t part) {
    return split_op_overlay(command) != 0 && (command.a6 & part) != 0;
}

template <typename Kernel>
SplitBlock split_block(const Kernel& kernel, const km_command_t& command) {
    const uint32_t plan = plan_words(command);
    const uint32_t m_tiles = kernel.arg(plan + kSpMt);
    const uint32_t n_tiles = kernel.arg(plan + kSpNt);
    const uint32_t block_height = kernel.arg(plan + kSpOh);
    const uint32_t block_width = kernel.arg(plan + kSpOw);
    const uint32_t group_index = (
        (kernel.me() - km::first(command)) / kernel.arg(plan + kSpS)
    );
    const uint32_t blocks_per_row = kernel.arg(plan + kSpNbw);
    const uint32_t first_row = group_index / blocks_per_row * block_height;
    const uint32_t first_column = group_index % blocks_per_row * block_width;
    return {
        &command,
        plan,
        first_row,
        first_column,
        m_tiles - first_row < block_height ? m_tiles - first_row : block_height,
        n_tiles - first_column < block_width ? n_tiles - first_column : block_width
    };
}

// Loads the operation's parts (the reader reads them all, the writer and compute meet
// it) and, with `call`, calls this kernel's part. False: gave up (recorded).
template <typename Kernel>
bool split_op_run(Kernel& kernel, const km_command_t& command, bool call) {
    km::Overlay operation(kernel, split_op_overlay(command));
    if (!operation.load()) {
        return false;
    }
#if defined(SPLIT_CHAIN_STAMPS) && defined(SPLIT_STAMP)
    SPLIT_STAMP(kChainStampOpLoaded);
#endif
    const bool called = !call || operation.call(split_block(kernel, command)) != 0;
#if defined(SPLIT_CHAIN_STAMPS) && defined(SPLIT_STAMP) && !defined(COMPILE_FOR_TRISC)
    SPLIT_STAMP(kChainStampOpCalled);  // compute's end is its kChainStampOut
#endif
    return called;
}

#if !defined(COMPILE_FOR_TRISC)
// The reader, the tile's loader: loads every part, calls its own if it has one.
inline bool split_op(km::Reader& reader, const km_command_t& command) {
    return (
        split_op_overlay(command) == 0 ||
        split_op_run(reader, command, split_op_part(command, kOpReader))
    );
}

inline bool split_op(km::Writer& writer, const km_command_t& command) {
    return split_op_run(writer, command, true);
}
#else
inline bool split_op(km::Compute& compute, const km_command_t& command) {
    return split_op_run(compute, command, true);
}
#endif
