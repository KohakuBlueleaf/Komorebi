// Attend's parts share: which key tiles a core of a KV head's group holds, and the
// pages of the cache, the query and the token's key and value.
#pragma once

#include <stdint.h>

#include "../ops.hpp"
#include "komorebi/device/movement/km_port.hpp"

// DC_KEYS_MOST (the overlay's define): key tiles a core holds at most, its buffers'
// size in key tiles (keys and values 4 pages each, scores 2, P 1).
#if !defined(DC_KEYS_MOST)
#error "DC_KEYS_MOST: the key tiles a core holds at most, the overlay's define"
#endif

constexpr uint32_t kTileRows = 32;
constexpr uint32_t kFaceBytes = 512;  // a bf16 face: 16 rows of 16 values

// A core's part of the head's attention: the head, its rank, the full key tiles it
// holds (first, count); the root also the tokens' one or two tiles (from last).
struct Attend {
    uint32_t head;
    uint32_t rank;
    uint32_t size;
    uint32_t first;
    uint32_t count;
    uint32_t last;     // the first token's key tile
    uint32_t row;      // the first token's row in it
    uint32_t rows;     // the tokens: the query's live rows, the rows appended
    uint32_t span;     // 1 when the tokens reach into the next tile
    uint32_t cache;    // key tiles a head in the cache
    uint32_t keys;     // key tiles this core holds: count, and the tokens' at the root
    uint32_t q_tiles;  // the query heads' width in tiles: q's columns in qkv
    uint32_t flags;    // a3's DC_ATTEND_*
    km::Group group;
};

// The full key tiles 0..last-1 split over the group, the larger shares on the higher
// ranks; the root (rank 0) also holds the tokens' tiles, which it appends to.
template <typename Kernel>
Attend attend_of(const Kernel& kernel, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const uint32_t size = kernel.arg(block.plan + kSpS);
    const km::Group group = km::digit_group(command, kernel.me(), 1, size);
    const uint32_t position = command.a9 & 0xFFFFu;
    const uint32_t last = position / kTileRows;
    const uint32_t rows = decode::live_rows(command);
    const uint32_t span = (position % kTileRows + rows - 1u) / kTileRows;
    const km::Partition share = (
        km::partition(last, group.size - 1 - group.rank, group.size)
    );
    return {
        (kernel.me() - km::first(command)) / size,
        group.rank,
        group.size,
        share.start,
        share.len,
        last,
        position % kTileRows,
        rows,
        span,
        (command.a9 >> 16) & 0xFFu,
        share.len + (group.rank == 0 ? 1u + span : 0u),
        kernel.arg(block.plan + kSpKt),
        command.a3 & (DC_ATTEND_READ_ONLY | DC_ATTEND_LOCAL | DC_ATTEND_EXP_TWICE),
        group
    };
}

// The first page of key tile `tile` of the head in a cache tensor: [heads x rows, 128].
inline uint32_t cache_page(const Attend& at, uint32_t tile) {
    return (at.head * at.cache + tile) * DC_HEAD_TILES;
}

// qkv's columns: the group's query tiles (two heads), the head's key, its value.
inline uint32_t query_page(const Attend& at) {
    return at.head * DC_HEAD_GROUP * DC_HEAD_TILES;
}

inline uint32_t token_page(const Attend& at, bool value) {
    const uint32_t keys = at.q_tiles / DC_HEAD_GROUP;
    return at.q_tiles + (value ? keys : 0u) + at.head * DC_HEAD_TILES;
}

// Whether the merge runs: the tree's rounds over the group's ports.
inline bool merges(const Attend& at) {
    return (at.flags & (DC_ATTEND_READ_ONLY | DC_ATTEND_LOCAL)) == 0;
}

// The merge's messages: a state of DC_STATE_ITEM pages in each arrival slot.
inline km::Arrival attend_arrival(uint32_t pages) {
    return {kCbArrive, pages, DC_STATE_ITEM};
}

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace attend {

    // Writer or reader: a timed reserve of `pages` contiguous pages of `cb`.
    template <typename Kernel>
    inline bool reserve(
        const Kernel& core, uint32_t cb, uint32_t pages, uint32_t word
    ) {
        if (!km::item_fits(cb, get_local_cb_interface(cb).fifo_wr_ptr, pages)) {
            core.gave_up(word, km::kGaveBadItem, km::kWhereData, 0, 0);
            return false;
        }
        core.produce(cb);
        const uint32_t code = core.poll_at(
            km::kWhereData,
            [&] {
                return cb_pages_reservable_at_back(cb, pages);
            },
            core.args().wait_limit
        );
        if (code != km::kGaveNone) {
            core.gave_up(
                word, code, km::kWhereData, core.running_slot(), core.running_sequence()
            );
            return false;
        }
        cb_reserve_back(cb, pages);
        return true;
    }

    // This core's cached key (value) tiles into `cb`, DC_KEY_BATCH at a time, each
    // pushed when landed; the buffer reserved whole (each command starts at its base).
    template <typename Kernel>
    inline bool read_cache(
        const Kernel& core,
        const km_command_t& command,
        const Attend& at,
        uint32_t cb,
        uint32_t tensor,
        uint32_t word
    ) {
        const km::Tensor<0> cache(tensor);
        if (!reserve(core, cb, DC_KEYS_MOST * DC_HEAD_TILES, word)) {
            return false;
        }
        for (uint32_t done = 0; done < at.count;) {
            const uint32_t batch = (
                at.count - done < DC_KEY_BATCH ? at.count - done : DC_KEY_BATCH
            );
            const uint32_t pages = batch * DC_HEAD_TILES;
            if (!km::read_pages(
                core,
                command,
                cache,
                cache_page(at, at.first + done),
                pages,
                get_write_ptr(cb)
            ) ||
                !core.flushed(km::kFlushReads, word)) {
                return false;
            }
            cb_push_back(cb, pages);
            done += batch;
        }
        return true;
    }

    // The root: the cache's tiles of the tokens with their key (value) rows put in at
    // their positions, written back to the cache, then pushed as this core's last.
    template <typename Kernel>
    inline bool append(
        const Kernel& core,
        const km_command_t& command,
        const Attend& at,
        uint32_t cb,
        uint32_t tensor,
        bool value,
        uint32_t word
    ) {
        const km::Tensor<0> cache(tensor);
        const km::Tensor<0> qkv(command.a0);
        const uint32_t slot = get_write_ptr(cb);
        if (!reserve(core, DC_CB_TOKEN, 2 * DC_HEAD_TILES, word)) {
            return false;
        }
        // the reader's key tiles in the first half, the writer's value tiles after them
        const uint32_t token = (
            get_write_ptr(DC_CB_TOKEN) + (value ? DC_HEAD_TILES * kTileBytes : 0u)
        );
        const uint32_t page = cache_page(at, at.last);
        const uint32_t pages = (1u + at.span) * DC_HEAD_TILES;
        if (
            !km::read_pages(core, command, cache, page, pages, slot) ||
            !km::read_pages(
                core, command, qkv, token_page(at, value), DC_HEAD_TILES, token
            ) ||
            !core.flushed(km::kFlushReads, word)
        ) {
            return false;
        }
        invalidate_l1_cache();  // the core reads what the NoC wrote
        // token row i into row `row` + i of the tiles: two faces' rows of 16 values a
        // tile
        for (uint32_t i = 0; i < at.rows; i++) {
            const uint32_t place = at.row + i;
            const uint32_t to_row = place % kTileRows;
            const uint32_t to_offset = (
                (place / kTileRows) * DC_HEAD_TILES * kTileBytes +
                (to_row >> 4) * 2 * kFaceBytes + (to_row & 15) * 32
            );
            const uint32_t from_offset = (i >> 4) * 2 * kFaceBytes + (i & 15) * 32;
            for (uint32_t tile = 0; tile < DC_HEAD_TILES; tile++) {
                for (uint32_t face = 0; face < 2; face++) {
                    const uint32_t at_face = tile * kTileBytes + face * kFaceBytes;
                    volatile tt_l1_ptr uint32_t* to = (
                        km::l1(slot + at_face + to_offset)
                    );
                    const volatile tt_l1_ptr uint32_t* from = (
                        km::l1(token + at_face + from_offset)
                    );
                    for (uint32_t word_index = 0; word_index < 8; word_index++) {
                        to[word_index] = from[word_index];
                    }
                }
            }
        }
        if (
            !km::write_pages(core, command, cache, page, pages, slot) ||
            !core.flushed(km::kFlushWrites, word)
        ) {
            return false;
        }
        cb_push_back(cb, pages);
        return true;
    }

    // The buffer's pages past this core's keys, pushed empty: every command ends with
    // the buffer pushed to its end.
    inline void pad(const Attend& at, uint32_t cb) {
        const uint32_t left = (DC_KEYS_MOST - at.keys) * DC_HEAD_TILES;
        if (left != 0) {
            cb_push_back(cb, left);
        }
    }

}  // namespace attend
#endif
