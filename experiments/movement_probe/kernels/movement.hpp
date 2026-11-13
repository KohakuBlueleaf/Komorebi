// Streams declared once and run by both data-movement CPUs of a core (the movers): each
// moves the blocks its pattern gives it through its own half of a paired buffer.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_core.hpp"
#include "pattern.hpp"

namespace movement {

#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t kMover = kMoverA;
#else
    constexpr uint32_t kMover = kMoverB;
#endif
    static_assert(noc_mode != DM_DYNAMIC_NOC, "the movers use one NoC each");

    // A read moves tensor pages into this mover's half; a write moves pages of its
    // half to the tensor.
    constexpr uint32_t kRead = 0;
    constexpr uint32_t kWrite = 1;

    // A logical buffer: two circular buffers, mover A's and mover B's. Each has one
    // CPU filling it and one draining it.
    struct Pair {
        uint32_t buffer_a;
        uint32_t buffer_b;
    };

    inline uint32_t half(const Pair& pair, uint32_t mover) {
        return mover == kMoverA ? pair.buffer_a : pair.buffer_b;
    }

    // Block k of a stream holds pages first_page + (k mod blocks per repetition) *
    // tiles_per_block onwards; tiles is a whole number of blocks.
    template <typename Accessor>
    struct Stream {
        uint32_t direction;
        Accessor tensor;
        uint32_t first_page;
        uint32_t tiles;
        uint32_t tiles_per_block;
        uint32_t repetitions;
        Pair pair;
        Pattern pattern;
    };

    // What one mover spent on its streams: tiles moved, cycles inside NoC calls and
    // barriers.
    struct Busy {
        uint32_t tiles;
        uint32_t cycles;
    };

    // Timed wait until this mover's reads have landed and its writes have left L1
    // (`acknowledged`: until they are acknowledged). False: gave up (recorded).
    inline bool settle(
        const km::Core& core,
        uint32_t gave_up_word,
        bool reads,
        bool writes,
        bool acknowledged = false
    ) {
        const uint32_t code = core.poll(
            [&] {
                const bool read_done = !reads || ncrisc_noc_reads_flushed(noc_index);
                if (!writes) {
                    return read_done;
                }
                return (
                    read_done &&
                    (
                        acknowledged
                            ? ncrisc_noc_nonposted_writes_flushed(noc_index)
                            : ncrisc_noc_nonposted_writes_sent(noc_index)
                    )
                );
            },
            core.args().wait_limit,
            core.args().wait_limit
        );
        if (code != km::kGaveNone) {
            core.gave_up(gave_up_word, code, km::kWhereData, 0, 0);
        }
        return code == km::kGaveNone;
    }

    // Where a mover is in a stream: the next block, its position in the pattern and
    // its first page's offset in the repetition, advanced without a division.
    struct Cursor {
        uint32_t block;
        uint32_t position;
        uint32_t offset;
    };

    template <typename Accessor>
    inline void step(Cursor& cursor, const Stream<Accessor>& stream) {
        cursor.block++;
        cursor.position = (
            cursor.position + 1 == stream.pattern.length ? 0 : cursor.position + 1
        );
        cursor.offset += stream.tiles_per_block;
        cursor.offset = cursor.offset == stream.tiles ? 0 : cursor.offset;
    }

    // Steps to the first block from the cursor on that this mover owns.
    template <typename Accessor>
    inline void seek(Cursor& cursor, const Stream<Accessor>& stream, uint32_t blocks) {
        while (
            cursor.block < blocks &&
            ((stream.pattern.owners >> cursor.position) & 1u) != kMover
        ) {
            step(cursor, stream);
        }
    }

    // Issues the page reads or writes of the cursor's block at this mover's half.
    template <typename Accessor>
    inline void issue(
        const Stream<Accessor>& stream, const Cursor& cursor, uint32_t page_bytes
    ) {
        const uint32_t buffer = half(stream.pair, kMover);
        const uint32_t page = stream.first_page + cursor.offset;
        if (stream.direction == kRead) {
            uint32_t address = get_write_ptr(buffer);
            for (uint32_t tile = 0; tile < stream.tiles_per_block; tile++) {
                noc_async_read_page(page + tile, stream.tensor, address);
                address += page_bytes;
            }
        } else {
            uint32_t address = get_read_ptr(buffer);
            for (uint32_t tile = 0; tile < stream.tiles_per_block; tile++) {
                noc_async_write_page(page + tile, stream.tensor, address);
                address += page_bytes;
            }
        }
    }

    // The engine: per pass, issue the next block of every stream whose half has room or
    // data, one barrier, push or pop; writes acknowledged at the end. False: gave up.
    template <typename Accessor, uint32_t Count>
    bool run(
        const km::Core& core,
        uint32_t gave_up_word,
        const Stream<Accessor> (&streams)[Count],
        Busy& busy
    ) {
        Cursor cursors[Count];
        uint32_t blocks[Count], page_bytes[Count];
        bool issued[Count];
        bool any_write = false;
        for (uint32_t index = 0; index < Count; index++) {
            const Stream<Accessor>& stream = streams[index];
            const LocalCBInterface& ring = (
                get_local_cb_interface(half(stream.pair, kMover))
            );
            if (
                !valid(stream.pattern) || stream.tiles_per_block == 0 ||
                stream.tiles % stream.tiles_per_block != 0 ||
                ring.fifo_num_pages % stream.tiles_per_block != 0
            ) {
                core.gave_up(gave_up_word, km::kGaveBadItem, km::kWhereData, 0, 0);
                return false;
            }
            blocks[index] = stream.tiles / stream.tiles_per_block * stream.repetitions;
            page_bytes[index] = ring.fifo_page_size;
            cursors[index] = {0, 0, 0};
            seek(cursors[index], stream, blocks[index]);
            issued[index] = false;
            any_write = any_write || stream.direction == kWrite;
        }
        const uint64_t limit = static_cast<uint64_t>(core.args().wait_limit) << 10;
        uint64_t deadline = km::wall64() + limit;
        for (uint32_t pass = 0;; pass++) {
            invalidate_l1_cache();
            bool open = false, reads = false, writes = false;
            const uint32_t start = km::wall();
            for (uint32_t index = 0; index < Count; index++) {
                const Stream<Accessor>& stream = streams[index];
                if (cursors[index].block >= blocks[index]) {
                    continue;
                }
                open = true;
                const uint32_t buffer = half(stream.pair, kMover);
                const uint32_t pages = stream.tiles_per_block;
                const LocalCBInterface& ring = get_local_cb_interface(buffer);
                const uint32_t pointer = (
                    stream.direction == kRead ? ring.fifo_wr_ptr : ring.fifo_rd_ptr
                );
                if (!km::item_fits(buffer, pointer, pages)) {
                    core.gave_up(gave_up_word, km::kGaveBadItem, km::kWhereData, 0, 0);
                    return false;
                }
                if (stream.direction == kRead) {
                    if (!cb_pages_reservable_at_back(buffer, pages)) {
                        continue;
                    }
                    cb_reserve_back(buffer, pages);
                    reads = true;
                } else {
                    if (!cb_pages_available_at_front(buffer, pages)) {
                        continue;
                    }
                    cb_wait_front(buffer, pages);
                    writes = true;
                }
                issue(stream, cursors[index], page_bytes[index]);
                issued[index] = true;
            }
            if (!open) {
                break;
            }
            if (!reads && !writes) {
                if ((pass & 15) == 15) {
                    if (core.stop()) {
                        core.gave_up(gave_up_word, km::kGaveStop, km::kWhereData, 0, 0);
                        return false;
                    }
                    if (km::wall64() > deadline) {
                        core.gave_up(
                            gave_up_word, km::kGaveLimit, km::kWhereData, 0, 0
                        );
                        return false;
                    }
                }
                continue;
            }
            if (!settle(core, gave_up_word, reads, writes)) {
                return false;
            }
            for (uint32_t index = 0; index < Count; index++) {
                if (!issued[index]) {
                    continue;
                }
                const Stream<Accessor>& stream = streams[index];
                const uint32_t buffer = half(stream.pair, kMover);
                if (stream.direction == kRead) {
                    cb_push_back(buffer, stream.tiles_per_block);
                } else {
                    cb_pop_front(buffer, stream.tiles_per_block);
                }
                busy.tiles += stream.tiles_per_block;
                step(cursors[index], stream);
                seek(cursors[index], stream, blocks[index]);
                issued[index] = false;
            }
            busy.cycles += km::wall() - start;
            deadline = km::wall64() + limit;
        }
        if (any_write) {
            const uint32_t start = km::wall();
            if (!settle(core, gave_up_word, false, true, true)) {
                return false;
            }
            busy.cycles += km::wall() - start;
        }
        return true;
    }

}  // namespace movement
