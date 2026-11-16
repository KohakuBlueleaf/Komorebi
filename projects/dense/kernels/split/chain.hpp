// SPLIT's chain (SPLIT_CHAIN): the framework's (komorebi/device/movement/km_chain.hpp),
// the split's plans as the records' shape, the next W read while a mark is awaited.
#pragma once

#include <stdint.h>

#if defined(SPLIT_CHAIN_STAMPS) && !defined(KM_CHAIN_STAMPS)
#define KM_CHAIN_STAMPS 1
#endif
#include "chain.h"
#include "komorebi/device/movement/km_chain.hpp"
#include "split.h"

#define CHAIN_STAMP(statement) KM_CHAIN_STAMP(statement)
using km::chain_command;
#if defined(KM_CHAIN_STAMPS) && !defined(KM_OVERLAY_PART)
using km::chain_stamp;
using km::chain_stamp_add;
using km::chain_stamp_once;
#endif

#if !defined(COMPILE_FOR_TRISC)
#include "komorebi/device/movement/km_movement.hpp"

using km::chain_credit;

#if !defined(KM_OVERLAY_PART)
using km::chain_now;
using km::chain_start;

// A record's shape from its plan (a5): its group size, the output tiles a group
// writes (the info word's, else the plan's block width), its K range, its tree's form.
struct SplitShape {
    template <typename Kernel>
    static uint32_t plan(const Kernel& core, const uint32_t* record, uint32_t word) {
        return core.arg(record[kChainArguments + 5] * kSplitWords + word);
    }

    template <typename Kernel>
    uint32_t size(const Kernel& core, const uint32_t* record) const {
        return plan(core, record, kSpS);
    }

    template <typename Kernel>
    uint32_t width(const Kernel& core, const uint32_t* record) const {
        const uint32_t width = (
            (record[kChainInfo] >> kChainWidthShift) & kChainWidthMask
        );
        return width != 0 ? width : plan(core, record, kSpOw);
    }

    template <typename Kernel>
    km::ChainRange k_range(const Kernel& core, const uint32_t* record) const {
        return {0, plan(core, record, kSpKt)};
    }

    uint32_t form(const km_command_t& command) const {
        return split_form(command);
    }
};

// The split's own: the next W early allowed (a1 kChainAheadOff clear), its tag and
// block; a sent input's DRAM copy acked too (a1 kChainDataDram).
struct SplitChain {
    bool ahead_on;
    bool data_dram;
    uint32_t ahead_tag;
    uint32_t ahead_block;
};

inline SplitChain split_chain = {};

// The record's input: waits until it is marked (a record without marks: at once).
template <typename Kernel>
bool chain_input(const Kernel& core) {
    return km::chain_input(SplitShape{}, core);
}

// Until the input is marked: W's blocks once A freed the half (`held`: some are in),
// then the next record's (`after`); `skip`, `after_skip` the blocks. False: gave up.
template <typename Kernel, typename Accessor>
bool chain_weight_first(
    const Kernel& core,
    const km::Stream<Accessor>& weights,
    const km::Stream<Accessor>& after,
    bool held,
    uint32_t& skip,
    uint32_t& after_skip
) {
    skip = 0;
    after_skip = 0;
    uint32_t target = 0;
    if (!km::chain_target(SplitShape{}, core, target)) {
        CHAIN_STAMP(chain_stamp(kChainStampMarked));
        return true;
    }
    CHAIN_STAMP(uint32_t full_at = 0);
    km::MoveCursor cursor, later;
    uint32_t filled = 0, drained = 0;
    if (
        !km::move_begin(weights, cursor, filled, drained) ||
        !km::move_begin(after, later, filled, drained)
    ) {
        core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
        return false;
    }
    const uint32_t half = weights.pair.half(km::kMover);
    bool freed = held, any_after = false;
    const bool marked = km::chain_wait(core, target, [&]() -> uint32_t {
        freed = (
            freed ||
            cb_pages_reservable_at_back(
                half, get_local_cb_interface(half).fifo_num_pages
            )
        );
        bool issued = false, open = false, reads = false, writes = false;
        bool later_issued = false;
        if (
            freed && !km::move_offer(core, weights, cursor, issued, open, reads, writes)
        ) {
            core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
            return km::kChainStopped;
        }
        if (
            freed && !open &&
            !km::move_offer(core, after, later, later_issued, open, reads, writes)
        ) {
            core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
            return km::kChainStopped;
        }
        if (issued || later_issued) {
            CHAIN_STAMP(chain_stamp_once(kChainStampWeight));
            if (!km::move_settle(core, true, false)) {
                return km::kChainStopped;
            }
            km::Moved moved = {0, 0};
            km::move_hand_on(weights, cursor, issued, moved);
            any_after = any_after || later_issued;
            km::move_hand_on(after, later, later_issued, moved);
            return km::kChainWorked;
        }
        CHAIN_STAMP(full_at = freed && full_at == 0 ? km::wall() | 1u : full_at);
        return km::kChainIdle;
    });
    if (!marked) {
        return false;
    }
    CHAIN_STAMP(
        chain_stamp_add(kChainStampFull, full_at != 0 ? km::wall() - full_at : 0)
    );
    CHAIN_STAMP(chain_stamp(kChainStampMarked));
    skip = cursor.block < cursor.blocks ? cursor.block : cursor.blocks;
    after_skip = (
        any_after ? (later.block < later.blocks ? later.block : later.blocks) : 0
    );
    return true;
}

// The record after this one when this mover may read its W early: a product this core
// is in, already in the ring (`words`). False: none (`words` this record's).
template <typename Kernel>
bool chain_next(const Kernel& core, uint32_t* words) {
    if (!km::chain_peek(words, split_chain.ahead_on)) {
        return false;
    }
    return (
        km::chain_member(words, core.me()) && words[kChainArguments + 4] != kSplitOp &&
        words[kChainArguments + 5] < kSplitPlans
    );
}

// This record's W blocks this mover read in the record before (0: none).
inline uint32_t chain_ahead_taken() {
    const uint32_t tag = chain_now.tag_base + chain_now.record;
    const uint32_t block = split_chain.ahead_tag == tag ? split_chain.ahead_block : 0;
    split_chain.ahead_tag = 0;
    return block;
}

// The next record's W read early up to block `block` (0: none).
inline void chain_ahead_keep(uint32_t block) {
    split_chain.ahead_tag = block != 0 ? chain_now.tag_base + chain_now.record + 1 : 0;
    split_chain.ahead_block = block;
}

// The L1 a core's K range of a kChainDataMark record's input lands in: its arrival
// buffer from page kChainInputPage on; and the tiles it holds.
inline uint32_t chain_input_area() {
    const LocalCBInterface& arrival = get_local_cb_interface(kCbArrive);
    return (
        arrival.fifo_limit - arrival.fifo_size +
        kChainInputPage * arrival.fifo_page_size
    );
}

// A group's root writer, its output written: the framework's marks, a sent input from
// kCbSplitOut's base (kept until its next credits) into the next cores' arrival areas.
template <typename Writer>
bool chain_mark(const Writer& writer, const km_command_t& command) {
    const LocalCBInterface& out = get_local_cb_interface(kCbSplitOut);
    const LocalCBInterface& arrival = get_local_cb_interface(kCbArrive);
    const km::ChainDirect direct = {
        out.fifo_limit - out.fifo_size,
        chain_input_area(),
        (arrival.fifo_size - kChainInputPage * arrival.fifo_page_size) / kTileBytes,
        kTileBytes
    };
    return km::chain_mark(SplitShape{}, writer, command, direct, split_chain.data_dram);
}

// Reader: the framework's chain, the split's a1 bits first.
template <typename Product>
void split_chain_reader(
    km::Reader& reader, const km_command_t& command, Product product
) {
    split_chain.ahead_on = (command.a1 & kChainAheadOff) == 0;
    split_chain.ahead_tag = 0;
    km::chain_reader(reader, command, product);
}

// Writer: the framework's chain, the split's a1 bits first.
template <typename Product>
bool chain_writer(km::Writer& writer, const km_command_t& command, Product product) {
    split_chain.ahead_on = (command.a1 & kChainAheadOff) == 0;
    split_chain.data_dram = (command.a1 & kChainDataDram) != 0;
    split_chain.ahead_tag = 0;
    return km::chain_writer(SplitShape{}, writer, command, product);
}

#if defined(KM_CHAIN_STAMPS)
// Writer, SPLIT's kChainStampsMode: this core's stamps to page `me` at a0, then zero.
template <typename Writer>
bool chain_stamps_out(const Writer& writer, const km_command_t& command) {
    return km::chain_stamps_out(writer, command.a0, kChainStampPage);
}
#endif
#endif
#else
using km::chain_compute;
#endif
