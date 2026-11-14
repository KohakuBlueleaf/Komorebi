// The K chunks of a core as two streams, A's and W's, dealt to both movers by AB: chunk
// q of both goes to the halves of the mover that the pattern names.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_movement.hpp"
#include "split.h"

constexpr km::BufferPair kChunksA = {kCbSa, kCbSaMoverB};
constexpr km::BufferPair kChunksW = {kCbSw, kCbSwMoverB};
constexpr km::Pattern kChunkPattern = km::kPatternAB;

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"

// This mover's K chunks of its core's partition: A's oh x kc and W's kc x ow tiles, the
// last chunk ragged. False: gave up (recorded).
template <typename Runtime, typename AccessorA, typename AccessorW>
bool move_chunks(
    const Runtime& core,
    const km_command_t& command,
    const AccessorA& a_matrix,
    const AccessorW& w_matrix
) {
    const auto plan = [&](uint32_t word) {
        return core.arg(word);
    };
    const uint32_t m_tiles = plan(kSpMt), k_tiles = plan(kSpKt);
    const uint32_t n_tiles = plan(kSpNt), block_height = plan(kSpOh);
    const uint32_t block_width = plan(kSpOw), chunk_tiles = plan(kSpKc);
    const uint32_t blocks_per_row = plan(kSpNbw), group_size = plan(kSpS);
    const km::Group group = km::digit_group(command, core.me(), 1, group_size);
    const uint32_t group_index = (core.me() - km::first(command)) / group_size;
    const uint32_t first_row = group_index / blocks_per_row * block_height;
    const uint32_t first_column = group_index % blocks_per_row * block_width;
    const km::Partition k_part = km::partition(k_tiles, group.rank, group.size);
    if (k_part.len == 0) {
        return true;
    }
    const uint32_t chunks = (k_part.len + chunk_tiles - 1) / chunk_tiles;
    const uint32_t last_length = k_part.len - (chunks - 1) * chunk_tiles;
    const uint32_t rows = (
        m_tiles - first_row < block_height ? m_tiles - first_row : block_height
    );
    const uint32_t columns = (
        n_tiles - first_column < block_width ? n_tiles - first_column : block_width
    );
    const km::Walk a_walk = {
        (first_row * k_tiles) + k_part.start,
        chunks,
        chunk_tiles,
        rows,
        k_tiles,
        chunk_tiles,
        1,
        chunk_tiles,
        block_height * chunk_tiles,
        rows,
        last_length
    };
    const km::Walk w_walk = {
        (k_part.start * n_tiles) + first_column,
        chunks,
        chunk_tiles * n_tiles,
        chunk_tiles,
        n_tiles,
        columns,
        1,
        block_width,
        chunk_tiles * block_width,
        last_length,
        columns
    };
    const km::Stream a_chunks(
        km::kMoveRead, a_matrix, a_walk, 1, kChunksA, kChunkPattern, kAskPort
    );
    const km::Stream w_chunks(
        km::kMoveRead, w_matrix, w_walk, 1, kChunksW, kChunkPattern, kAskPort
    );
    return km::move(core, a_chunks, w_chunks);
}
#endif
