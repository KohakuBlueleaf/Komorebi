// The K chunks of a core as two streams, A's and W's, dealt to the movers by the plan's
// pattern (AB, AAAB): chunk q of both goes to the halves of the mover it names.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_movement.hpp"
#include "split.h"

constexpr km::BufferPair kChunksA = {kCbSa, kCbSaMoverB};
constexpr km::BufferPair kChunksW = {kCbSw, kCbSwMoverB};

// A kernel's product: the chain's records and a plain command call one copy of it.
#if defined(SPLIT_CHAIN)
#define SPLIT_PRODUCT __attribute__((noinline))
#else
#define SPLIT_PRODUCT
#endif

// A record's stamp in a product (chain.h kChainStamp*; compute's on pack), or nothing.
#if defined(SPLIT_CHAIN) && defined(SPLIT_CHAIN_STAMPS) && defined(COMPILE_FOR_TRISC)
#define SPLIT_STAMP(word) PACK((chain_stamp_once(word)))
#elif defined(SPLIT_CHAIN) && defined(SPLIT_CHAIN_STAMPS)
#define SPLIT_STAMP(word) chain_stamp_once(word)
#else
#define SPLIT_STAMP(word) ((void)0)
#endif

// The pattern a plan's kSpPattern word names.
inline km::Pattern chunk_pattern(uint32_t word) {
    return word == kSplitAAAB ? km::kPatternAAAB : km::kPatternAB;
}

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"

#if defined(SPLIT_CHAIN)
#include "chain.hpp"

// `pattern` from chunk `first` on: chunk first + b is its block b.
inline km::Pattern pattern_from(const km::Pattern& pattern, uint32_t first) {
    const uint32_t shift = first % pattern.length;
    if (shift == 0) {
        return pattern;
    }
    const uint32_t mask = (1u << pattern.length) - 1u;
    return {
        pattern.length,
        ((pattern.owners >> shift) | (pattern.owners << (pattern.length - shift))) &
            mask
    };
}

// The W stream of `command` on this core from chunk `skip` on, kc x ow tiles a block
// (no blocks: `on` false, or no K here).
template <typename Runtime, typename AccessorW>
auto chunk_w_stream(
    const Runtime& core,
    const km_command_t& command,
    const AccessorW& w_matrix,
    uint32_t skip,
    bool on
) {
    const auto plan = [&](uint32_t word) {
        return core.arg(plan_words(command) + word);
    };
    const uint32_t k_tiles = plan(kSpKt), n_tiles = plan(kSpNt);
    const uint32_t block_width = plan(kSpOw), chunk_tiles = plan(kSpKc);
    const uint32_t blocks_per_row = plan(kSpNbw), group_size = plan(kSpS);
    const km::Group group = km::digit_group(command, core.me(), 1, group_size);
    const uint32_t group_index = (core.me() - km::first(command)) / group_size;
    const uint32_t first_column = group_index % blocks_per_row * block_width;
    const km::Partition k_part = km::partition(k_tiles, group.rank, group.size);
    const uint32_t chunks = (k_part.len + chunk_tiles - 1) / chunk_tiles;
    const uint32_t last_length = (
        chunks == 0 ? chunk_tiles : k_part.len - (chunks - 1) * chunk_tiles
    );
    const uint32_t columns = (
        n_tiles - first_column < block_width ? n_tiles - first_column : block_width
    );
    const km::Walk walk = {
        ((k_part.start + skip * chunk_tiles) * n_tiles) + first_column,
        on && skip < chunks ? chunks - skip : 0,
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
    const km::Pattern pattern = chunk_pattern(plan(kSpPattern));
    return (
        km::Stream(
            km::kMoveRead, w_matrix, walk, 1, kChunksW, pattern_from(pattern, skip)
        )
    );
}
#endif

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
        return core.arg(plan_words(command) + word);
    };
    const uint32_t m_tiles = plan(kSpMt), k_tiles = plan(kSpKt);
    const uint32_t n_tiles = plan(kSpNt), block_height = plan(kSpOh);
    const uint32_t block_width = plan(kSpOw), chunk_tiles = plan(kSpKc);
    const uint32_t blocks_per_row = plan(kSpNbw), group_size = plan(kSpS);
    const km::Pattern pattern = chunk_pattern(plan(kSpPattern));
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
#if defined(SPLIT_CHAIN)
    // a chain's record: W's blocks while its input is not yet marked (then the next
    // record's), then A's chunks with the rest of W (two streams; chain.hpp)
    const uint32_t ahead = chain_ahead_taken();
    uint32_t words[kChainWords];
    const bool next = chain_next(core, words);
    const km_command_t after = chain_command(command, words);
    uint32_t skip = 0, after_skip = 0;
    if (!chain_weight_first(
        core,
        chunk_w_stream(core, command, w_matrix, ahead, true),
        chunk_w_stream(core, after, AccessorW(after.a1), 0, next),
        ahead != 0,
        skip,
        after_skip
    )) {
        return false;
    }
    chain_ahead_keep(next ? after_skip : 0);
    SPLIT_STAMP(kChainStampWeight);  // no block before the mark: W's first is now
    const km::Stream a_chunks(km::kMoveRead, a_matrix, a_walk, 1, kChunksA, pattern);
    const bool moved = (
        km::move(
            core, a_chunks, chunk_w_stream(core, command, w_matrix, ahead + skip, true)
        )
    );
    SPLIT_STAMP(kChainStampMoved);
    return moved;
#else
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
    const km::Stream a_chunks(km::kMoveRead, a_matrix, a_walk, 1, kChunksA, pattern);
    const km::Stream w_chunks(km::kMoveRead, w_matrix, w_walk, 1, kChunksW, pattern);
    return km::move(core, a_chunks, w_chunks);
#endif
}
#endif
