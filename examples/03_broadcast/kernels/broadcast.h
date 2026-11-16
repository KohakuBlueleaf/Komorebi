// Broadcast example: handlers, buffers and arguments its kernels share, and the two
// ways a core gets its blocks (broadcast, or its own DRAM reads).
#pragma once

#include <stdint.h>

#include "komorebi/device/movement/km_broadcast.hpp"

// CAST: a0 row_address (A), a1 column_address (W), a2 output_address, a3 blocks, a4
// stalled_rank at block a5 stall_block (kNone: none), a6 first_page, a7 data_blocks.
constexpr uint32_t kHandlerBroadcast = 1;
// READ: as CAST, but every core of the group reads both of its blocks from DRAM itself.
constexpr uint32_t kHandlerRead = 2;

constexpr uint32_t kCbRow = 0;   // row broadcast buffer: B blocks of P tiles of A
constexpr uint32_t kCbCol = 1;   // column broadcast buffer: the same for W
constexpr uint32_t kCbOut = 16;  // one fp32 tile
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kNone = 0xFFFFFFFFu;

// Program arguments of reader and writer: the broadcast's words, then the core's row
// (or column) in the rectangle. Compute: included (1 inside the rectangle), B, P.
constexpr uint32_t kArgBroadcast = 0;
constexpr uint32_t kArgIndex = km::kBroadcastArgs;

#if !defined(COMPILE_FOR_TRISC)

#include "komorebi/device/tensors/km_ask.hpp"

// Timed wait for this kernel's reads; gives up into state word `word`.
inline bool reads_landed(
    const km::Core& core, uint32_t word, const km_command_t& command
) {
    const uint32_t code = core.poll(
        [] {
            return ncrisc_noc_reads_flushed(noc_index);
        },
        core.args().wait_limit,
        core.args().wait_limit
    );
    if (code != km::kGaveNone) {
        core.gave_up(word, code, km::kWhereData, km::slot(command), command.sequence);
    }
    return code == km::kGaveNone;
}

// Each block: the sender reads it (tiles first_tile + block % data_blocks * block_pages
// on) into its slot and broadcasts it; a receiver waits for it. Stops at block `stall`.
template <typename Source, typename Stall>
bool broadcast_blocks(
    const km::Core& core,
    const km_command_t& command,
    km::Broadcast& broadcast,
    const Source& source,
    uint32_t first_tile,
    uint32_t blocks,
    uint32_t data_blocks,
    Stall stall
) {
    const uint32_t block_pages = broadcast.pages();
    for (uint32_t block = 0; block < blocks; block++) {
        if (stall(block)) {
            return false;
        }
        if (!broadcast.sender()) {
            if (!broadcast.arrive(block)) {
                return false;
            }
            continue;
        }
        const uint32_t address = broadcast.reserve(block);
        if (address == 0) {
            return false;
        }
        if (!km::read_pages(
            core,
            command,
            source,
            first_tile + block % data_blocks * block_pages,
            block_pages,
            address
        )) {
            return false;
        }
        if (!broadcast.cast(block)) {
            return false;
        }
    }
    return broadcast.finish();
}

// READ: the core reads its blocks itself into the same buffer, then pads it to
// whole windows as Broadcast::finish does (compute runs the same code for both).
template <typename Source>
bool read_blocks(
    const km::Core& core,
    uint32_t word,
    const km_command_t& command,
    const km::Broadcast& broadcast,
    uint32_t cb,
    const Source& source,
    uint32_t first_tile,
    uint32_t blocks,
    uint32_t data_blocks
) {
    const uint32_t block_pages = broadcast.pages(), window = broadcast.window();
    const uint32_t padding = (window - blocks % window) % window * block_pages;
    for (uint32_t block = 0; block <= blocks; block++) {
        const uint32_t pages = block < blocks ? block_pages : padding;
        if (pages == 0) {
            break;
        }
        const uint32_t code = core.poll(
            [&] {
                return cb_pages_reservable_at_back(cb, pages);
            },
            core.args().wait_limit
        );
        if (code != km::kGaveNone || !km::item_fits(cb, get_write_ptr(cb), pages)) {
            const uint32_t reason = code != km::kGaveNone ? code : km::kGaveBadItem;
            core.gave_up(
                word, reason, km::kWhereData, km::slot(command), command.sequence
            );
            return false;
        }
        cb_reserve_back(cb, pages);
        const uint32_t destination = get_write_ptr(cb);
        if (
            block < blocks &&
            !km::read_pages(
                core,
                command,
                source,
                first_tile + block % data_blocks * block_pages,
                block_pages,
                destination
            )
        ) {
            return false;
        }
        if (!reads_landed(core, word, command)) {
            return false;
        }
        cb_push_back(cb, pages);
    }
    return true;
}

#endif
