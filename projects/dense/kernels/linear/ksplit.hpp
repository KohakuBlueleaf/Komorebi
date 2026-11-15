// The K-split plan (LINEAR_SPLIT): the island's part as its placements give it, in
// groups of S ranks; this rank's K chunks as X's and W's streams on both movers.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_movement.hpp"
#include "linear.h"

constexpr km::BufferPair kKChunksX = {kCbKa, kCbKaB};
constexpr km::BufferPair kKChunksW = {kCbKw, kCbKwB};

// This rank's group of the K-split plan: S consecutive ranks of its island's part of
// the command's group (a group never crosses islands); `index`, the group's number.
inline km::Group k_group(
    const km_command_t& command, uint32_t me, uint32_t size, uint32_t& index
) {
    const km::Group local = km::group_of(command, me).local();
    index = local.rank / size;
    const uint32_t low = index * size;
    const uint32_t left = local.size - low;
    return {local.base + low, 1, left < size ? left : size, local.rank - low};
}

// A rank's place in the K-split plan: its group (a block of the window), its K.
struct KPlace {
    bool any;            // a group of the plan, with K to take
    uint32_t first_row;  // the block's first row and column tile of Y
    uint32_t rows;
    uint32_t first_column;
    uint32_t columns;
    uint32_t k_first;  // its K tiles: first .. first + count - 1
    uint32_t k_count;
};

template <typename Plan>
inline KPlace k_place(
    const km_command_t& command, uint32_t me, const Plan& plan, uint32_t here
) {
    const uint32_t x = command.a0, w = command.a1, y = command.a2;
    const uint32_t k_whole = command.a4, n_whole = command.a5;
    const uint32_t m_tiles = plan(kKsMt), group_size = plan(kKsS);
    const uint32_t block_height = plan(kKsOh), block_width = plan(kKsOw);
    invalidate_l1_cache();  // the tensor table
    const Span mine = k_part(w, here, k_whole, n_whole);
    const Span made = made_rows(x, here, m_tiles, k_whole);
    const Span kept = window(x, w, y, here, here, m_tiles, k_whole, n_whole);
    uint32_t index = 0;
    const km::Group group = k_group(command, me, group_size, index);
    const uint32_t first_row = made.first + index / plan(kKsNbw) * block_height;
    const uint32_t column = index % plan(kKsNbw) * block_width;
    const uint32_t made_end = made.first + made.count;
    if (mine.count == 0 || first_row >= made_end || column >= kept.count) {
        return {false, 0, 0, 0, 0, 0, 0};
    }
    const km::Partition part = km::partition(mine.count, group.rank, group.size);
    return {
        true,
        first_row,
        made_end - first_row < block_height ? made_end - first_row : block_height,
        kept.first + column,
        kept.count - column < block_width ? kept.count - column : block_width,
        mine.first + part.start,
        part.len
    };
}

// The whole-K plan's X pages: its K chunks of oh x kc tiles.
template <typename Plan>
inline uint32_t x_pages(const Plan& plan, const KPlace& at) {
    const uint32_t chunk_tiles = plan(kKsKc);
    return (at.k_count + chunk_tiles - 1) / chunk_tiles * plan(kKsOh) * chunk_tiles;
}

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

// This mover's K chunks of its rank's part: X's oh x kc and W's kc x ow tiles, the
// last chunk ragged. False: gave up (recorded).
template <typename Runtime, typename Plan, typename X, typename W>
bool move_k_chunks(
    const Runtime& core,
    const km_command_t& command,
    const Plan& plan,
    const KPlace& at,
    const X& x_matrix,
    const W& w_matrix
) {
    if (!at.any || at.k_count == 0) {
        return true;
    }
    const uint32_t k_whole = command.a4, n_whole = command.a5;
    const uint32_t chunk_tiles = plan(kKsKc), block_height = plan(kKsOh);
    const uint32_t block_width = plan(kKsOw);
    const uint32_t chunks = (at.k_count + chunk_tiles - 1) / chunk_tiles;
    const uint32_t last_length = at.k_count - (chunks - 1) * chunk_tiles;
    const km::Walk x_walk = {
        at.first_row * k_whole + at.k_first,
        chunks,
        chunk_tiles,
        at.rows,
        k_whole,
        chunk_tiles,
        1,
        chunk_tiles,
        block_height * chunk_tiles,
        at.rows,
        last_length
    };
    const km::Walk w_walk = {
        at.k_first * n_whole + at.first_column,
        chunks,
        chunk_tiles * n_whole,
        chunk_tiles,
        n_whole,
        at.columns,
        1,
        block_width,
        chunk_tiles * block_width,
        last_length,
        at.columns
    };
    const km::Pattern pattern = km::Pattern::of(plan(kKsPattern));
    const km::Stream x_chunks(km::kMoveRead, x_matrix, x_walk, 1, kKChunksX, pattern);
    const km::Stream w_chunks(km::kMoveRead, w_matrix, w_walk, 1, kKChunksW, pattern);
    if (plan(kKsXCast) != 0) {  // the whole-K plan: X comes by the panel's multicast
        return km::move(core, w_chunks);
    }
    return km::move(core, x_chunks, w_chunks);
}

// The whole-K plan's X (writer): the sender, all buffers free, reads chunks while the
// last is cast and marked; a receiver's compute waits for the marks. False: gave up.
template <typename Plan, typename X>
bool hand_x(
    const km::Writer& writer,
    const km_command_t& command,
    const Plan& plan,
    const KPlace& at,
    const X& x_matrix
) {
    const uint32_t k_whole = command.a4, chunk_tiles = plan(kKsKc);
    const uint32_t chunk_pages = plan(kKsOh) * chunk_tiles, per = plan(kKsXChunks);
    const uint32_t pages = x_pages(plan, at), dests = plan(kKsXDests);
    const bool sender = plan(kKsXRole) == kKXSender;
    const uint32_t named = km::slot(command) | kPortKXGot << 8;
    writer.produce(kCbKX);
    uint32_t code = writer.poll_at(
        km::kWhereData,
        [&] {
            return cb_pages_reservable_at_back(kCbKX, pages);
        },
        writer.args().wait_limit
    );
    const LocalCBInterface& buffer = get_local_cb_interface(kCbKX);
    const uint32_t base = buffer.fifo_limit - buffer.fifo_size;  // the same everywhere
    if (code == km::kGaveNone) {
        cb_reserve_back(kCbKX, pages);
        if (buffer.fifo_wr_ptr != base || pages != buffer.fifo_num_pages || per == 0) {
            code = km::kGaveBadItem;
        }
    }
    if (code != km::kGaveNone) {
        writer.gave_up(km::kStWriter, code, km::kWhereData, named, command.sequence);
        return false;
    }
    const uint32_t ports = writer.args().control + KM_CONTROL_PORT + 4 * KM_PORT_COUNT;
    const uint32_t got = ports + kPortKXGot * KM_PORT_STRIDE;
    if (!sender) {
        noc_semaphore_inc(
            writer.core_noc(plan(kKsXSender), ports + kPortKXFree * KM_PORT_STRIDE), 1
        );
        cb_push_back(kCbKX, pages);  // compute takes each chunk once it is marked
    } else if (!counted(writer, km::kStWriter, command, kPortKXFree, plan(kKsXWaits))) {
        return false;
    }
    const uint32_t start = plan(kKsXStart), end = plan(kKsXEnd);
    const auto panel = [&](uint32_t address) {
        return (
            get_noc_multicast_addr(
                start & 0xFFFFu, start >> 16, end & 0xFFFFu, end >> 16, address
            )
        );
    };
    volatile tt_l1_ptr uint32_t* marker = km::port_words(writer, kPortKXGot);
    const auto mark =
        [&](uint32_t cast) {  // all cast so far landed (and the last mark)
            if (!landed(writer, km::kStWriter, command)) {
                return false;
            }
            marker[KM_PORT_COUNT] = km::broadcast_marker(command.sequence, cast);
            noc_semaphore_set_multicast(got, panel(got), dests);
            return true;
        };
    const uint32_t chunks = pages / chunk_pages;
    for (uint32_t first = 0; sender && first < chunks; first += per) {
        const uint32_t count = chunks - first < per ? chunks - first : per;
        const uint32_t offset = first * chunk_pages * kBf16Bytes;
        for (uint32_t chunk = first; chunk < first + count; ++chunk) {
            const uint32_t k = chunk * chunk_tiles;
            const uint32_t length = (
                at.k_count - k < chunk_tiles ? at.k_count - k : chunk_tiles
            );
            for (uint32_t row = 0; row < at.rows; ++row) {
                if (!km::read_pages(
                    writer,
                    command,
                    x_matrix,
                    (at.first_row + row) * k_whole + at.k_first + k,
                    length,
                    base + (chunk * chunk_pages + row * chunk_tiles) * kBf16Bytes
                )) {
                    return false;
                }
            }
        }
        if (!flushed(writer, km::kStWriter, km::kFlushReads)) {
            return false;
        }
        cb_push_back(kCbKX, count * chunk_pages);
        if (dests == 0) {
            continue;
        }
        // the last multicast, cast while these reads came in, marked once landed
        if (first != 0 && !mark(first)) {
            return false;
        }
        const uint32_t bytes = count * chunk_pages * kBf16Bytes;
        noc_async_write_multicast(base + offset, panel(base + offset), bytes, dests);
    }
    if (sender && dests != 0 && !mark(chunks)) {
        return false;
    }
    code = writer.poll_at(  // the receiver's increment, the sender's last marker
        km::kWhereData,
        [] {
            return (
                ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                ncrisc_noc_nonposted_writes_flushed(noc_index)
            );
        },
        writer.args().wait_limit
    );
    if (code != km::kGaveNone) {
        writer.gave_up(km::kStWriter, code, km::kWhereData, named, command.sequence);
        return false;
    }
    return true;
}
#endif
