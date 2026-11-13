// Broadcast: a sender multicasts block `block` into slot `block % window` of every
// receiver (with a word, or a word alone); a receiver marks it in the DONE-BY table.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_command.hpp"
#include "km_worker.hpp"

namespace km {

    constexpr uint32_t kBroadcastNone = 0xFF;  // rank of a core that is not in the broadcast
    constexpr uint32_t kBroadcastWindowMax = 8;
    constexpr uint32_t kBroadcastBlocksMax = 32767;
    constexpr uint32_t kFlushReads = 0;   // Broadcast::flushed: reads have landed
    constexpr uint32_t kFlushSent = 1;    // writes have left L1
    constexpr uint32_t kFlushWrites = 2;  // writes are acknowledged

    // A broadcast's start-up arguments: kBroadcastArgs program words from the index the
    // program gives. The host computes and checks them (komorebi.layout).
    struct BroadcastArgs {
        uint32_t
            broadcast;   // broadcast id, 0 to KM_BROADCASTS - 1: its control-block area
        uint32_t group;  // first | count << 16: the group the layout was built for
        uint32_t rank;   // 0: sender, 1 to receivers: receiver, kBroadcastNone: neither
        uint32_t receivers;  // cores in the multicast rectangle
        uint32_t sender;     // core index of the broadcast's sender
        uint32_t mc_start;   // multicast rectangle corners, NoC x | y << 16, in the
        uint32_t mc_end;     // order of the broadcast's NoC
        uint32_t noc;        // the broadcast's NoC; the whole group runs it on that NoC
        uint32_t cb;         // broadcast buffer: window * pages pages
        uint32_t window;     // B: blocks in flight, 1 to kBroadcastWindowMax
        uint32_t pages;      // P: pages per block; window and pages 0: no buffer
    };

    constexpr uint32_t kBroadcastArgs = sizeof(BroadcastArgs) / 4;

    // A marker: the command's sequence number (low 16 bits) and a count of blocks.
    inline uint32_t broadcast_marker(uint32_t sequence, uint32_t blocks) {
        return (sequence & 0xFFFFu) << 16 | (blocks & 0xFFFFu);
    }

    // True when `marker` belongs to command `sequence` and counts at least `blocks`.
    inline bool broadcast_reached(uint32_t marker, uint32_t sequence, uint32_t blocks) {
        return (
            (marker >> 16) == (sequence & 0xFFFFu) &&
            static_cast<int16_t>(static_cast<uint16_t>(marker - blocks)) >= 0
        );
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // One broadcast in one command, in the DM kernel on the broadcast's NoC. Per block:
    // the sender reserve, fill, cast; a receiver arrive; after the last, finish().
    class Broadcast {
    public:
        Broadcast(const Reader& reader, const km_command_t& command, uint32_t arg0) :
            Broadcast(static_cast<const Core&>(reader), kStReader, command, arg0) {
        }

        Broadcast(const Writer& writer, const km_command_t& command, uint32_t arg0) :
            Broadcast(static_cast<const Core&>(writer), kStWriter, command, arg0) {
        }

        // False: the arguments, the command or the buffer did not fit (recorded).
        bool ok() const {
            return ok_;
        }

        uint32_t rank() const {
            return args_.rank;
        }

        bool sender() const {
            return args_.rank == 0;
        }

        bool included() const {
            return args_.rank != kBroadcastNone;
        }

        // P, pages per block, and B, blocks in flight.
        uint32_t pages() const {
            return args_.pages;
        }

        uint32_t window() const {
            return args_.window;
        }

        // Blocks multicast or arrived in this command.
        uint32_t blocks() const {
            return block_count_;
        }

        // The word of the block cast or arrived last, 0 if that block carried none.
        uint32_t word() const {
            return received_;
        }

        // Sender: timed wait until its compute freed slot `block % window`; returns the
        // slot's L1 address (the same on every core of the group), 0 after a give-up.
        uint32_t reserve(uint32_t block) {
            if (!in_order(block)) {
                return 0;
            }
            if (args_.pages == 0) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, block);
                return 0;
            }
            if (!wait_free(args_.pages, block)) {
                return 0;
            }
            if (
                get_write_ptr(args_.cb) != slot(block) ||
                !item_fits(args_.cb, get_write_ptr(args_.cb), args_.pages)
            ) {
                fail(kGaveBadItem, kWhereBroadcastFree, block);
                return 0;
            }
            cb_reserve_back(args_.cb, args_.pages);
            reserved_ = true;
            return slot(block);
        }

        // Sender, `block` filled: timed wait for every receiver to be done with block
        // `block - window`, multicasts it, barrier, VALID, push to compute.
        bool cast(uint32_t block) {
            return send(block, 0, false);
        }

        // The same, and every receiver takes `word` in arrive(block). Without a buffer
        // the word goes alone: no reserve. Each receiver took the last word first.
        bool cast(uint32_t block, uint32_t word) {
            return send(block, word, true);
        }

        // Receiver: waits (with a timeout) for VALID >= `block + 1`, marking the blocks
        // its compute popped meanwhile, takes the block's word, pushes it to compute.
        bool arrive(uint32_t block) {
            if (!in_order(block)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* valid = area_word(KM_BROADCAST_VALID);
            uint32_t marker = 0;
            uint32_t word = 0;
            uint32_t carries = 0;
            const uint64_t started = wall64();
            const uint32_t code = core_.poll(
                [&] {
                    marker_freed();
                    marker = valid[0];
                    if (!broadcast_reached(marker, sequence_, block + 1)) {
                        return false;
                    }
                    word = valid[KM_BROADCAST_WORD];
                    carries = valid[KM_BROADCAST_CARRIES];
                    return valid[0] == marker;  // the three words of one VALID
                },
                core_.args().wait_limit
            );
            add_wait(wall64() - started);
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastValid, block);
                return false;
            }
            // A block with a word stays the last VALID until every receiver took it.
            received_ = 0;
            if (carries != 0 && (marker & 0xFFFFu) == ((block + 1) & 0xFFFFu)) {
                received_ = word;
                if (!took_word(block)) {
                    return false;
                }
            }
            if (args_.pages == 0) {
                block_count_++;
                return true;
            }
            if (!wait_free(args_.pages, block)) {
                return false;
            }
            if (
                get_write_ptr(args_.cb) != slot(block) ||
                !item_fits(args_.cb, get_write_ptr(args_.cb), args_.pages)
            ) {
                fail(kGaveBadItem, kWhereBroadcastFree, block);
                return false;
            }
            cb_reserve_back(args_.cb, args_.pages);
            cb_push_back(args_.cb, args_.pages);
            block_count_++;
            return true;
        }

        // After the last block (timed): compute popped all, padding to whole windows,
        // last marker; a sender that multicast waits for all receivers'. Clears words.
        bool finish() {
            if (!ok_ || !included()) {
                return ok_;
            }
            if (!sender()) {
                *area_word(
                    KM_BROADCAST_VALID
                ) = 0;  // every VALID of this command has arrived
            }
            if (args_.pages != 0 && !drain()) {
                return false;
            }
            if (
                sender() && block_count_ != 0
            ) {  // no block multicast: no receiver markers
                if (
                    word_blocks_ != 0 &&
                    !wait_done(word_blocks_, block_count_, KM_BROADCAST_ARRIVED)
                ) {
                    return false;
                }
                if (args_.pages != 0 && !wait_done(block_count_, block_count_)) {
                    return false;
                }
                for (uint32_t receiver = 1; receiver <= args_.receivers; receiver++) {
                    volatile tt_l1_ptr uint32_t* done = (
                        area_word(KM_BROADCAST_DONE + 16 * receiver)
                    );
                    done[0] = 0;
                    done[KM_BROADCAST_ARRIVED] = 0;
                }
            }
            if (!flushed(kFlushWrites, kWhereBroadcastFree, block_count_)) {
                return false;
            }
            add_sat(area_word(KM_BROADCAST_STATS), block_count_);
            return true;
        }

    private:
        // Sender: cast(block) and cast(block, word); `carries` says which.
        bool send(uint32_t block, uint32_t word, bool carries) {
            const bool tiles = args_.pages != 0;
            if (tiles ? !reserved_ : !carries) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, block);
                return false;
            }
            if (!in_order(block)) {
                return false;
            }
            if (tiles) {
                if (!flushed(
                    kFlushReads, kWhereBroadcastDone, block
                )) {  // the block's reads have landed
                    return false;
                }
                if (
                    block >= args_.window && !wait_done(block - args_.window + 1, block)
                ) {
                    return false;
                }
            }
            if (args_.receivers != 0) {
                if (tiles) {
                    const uint32_t source = slot(block);
                    noc_async_write_multicast(
                        source, multicast_address(source), block_bytes_, args_.receivers
                    );
                }
                if (!flushed(
                    tiles ? kFlushWrites : kFlushSent, kWhereBroadcastDone, block
                )) {  // the block landed before its VALID; the last VALID left
                    return false;
                }
                if (word_blocks_ != 0) {  // every receiver took the last word
                    if (!wait_done(word_blocks_, block, KM_BROADCAST_ARRIVED)) {
                        return false;
                    }
                    word_blocks_ = 0;
                }
                volatile tt_l1_ptr uint32_t* out = area_word(KM_BROADCAST_OUT);
                out[KM_BROADCAST_WORD] = word;
                out[KM_BROADCAST_CARRIES] = carries ? 1 : 0;
                out[0] = broadcast_marker(sequence_, block + 1);
                noc_async_write_multicast(
                    area_ + KM_BROADCAST_OUT,
                    multicast_address(area_ + KM_BROADCAST_VALID),
                    16,
                    args_.receivers
                );
                if (carries) {
                    word_blocks_ = block + 1;
                }
            }
            if (tiles) {
                cb_push_back(args_.cb, args_.pages);
                reserved_ = false;
            }
            received_ = carries ? word : 0;
            block_count_++;
            return true;
        }

        // Receiver: marks in the sender's area that it took the word of `block`.
        bool took_word(uint32_t block) {
            if (!flushed(
                kFlushSent, kWhereBroadcastValid, block
            )) {  // the last marker has left KM_BROADCAST_OUT
                return false;
            }
            const uint32_t offset = 4 * KM_BROADCAST_ARRIVED;
            *area_word(KM_BROADCAST_OUT + offset) = broadcast_marker(
                sequence_, block + 1
            );
            noc_async_write(
                area_ + KM_BROADCAST_OUT + offset,
                core_.core_noc(
                    args_.sender, area_ + KM_BROADCAST_DONE + 16 * args_.rank + offset
                ),
                4
            );
            return true;
        }

        // finish() with a buffer: compute popped every block, then the padding.
        bool drain() {
            const uint32_t code = core_.poll(
                [&] {
                    marker_freed();
                    return popped() == block_count_;
                },
                core_.args().wait_limit
            );
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastFree, block_count_);
                return false;
            }
            marker_freed();  // the last pop may have come after the last marker
            if (!ok_) {
                return false;
            }
            const uint32_t padding = (
                (args_.window - block_count_ % args_.window) % args_.window *
                args_.pages
            );
            if (padding != 0) {
                if (!item_fits(args_.cb, get_write_ptr(args_.cb), padding)) {
                    fail(kGaveBadItem, kWhereBroadcastFree, block_count_);
                    return false;
                }
                cb_reserve_back(args_.cb, padding);  // free: every block was popped
                cb_push_back(args_.cb, padding);
            }
            return true;
        }

        Broadcast(
            const Core& core, uint32_t word, const km_command_t& command, uint32_t arg0
        ) :
            core_(core),
            word_(word),
            sequence_(command.sequence),
            slot_(km::slot(command)) {
            uint32_t* words = &args_.broadcast;
            for (uint32_t i = 0; i < kBroadcastArgs; i++) {
                words[i] = core.arg(arg0 + i);
            }
            area_ = (
                core.args().control + KM_CONTROL_BROADCAST +
                (args_.broadcast % KM_BROADCASTS) * KM_BROADCAST_BYTES
            );
            ok_ = begin(command);
        }

        // Checks the arguments against the command and the buffer, then a timed wait
        // until the buffer is empty at slot 0 (the last command's padding was popped).
        bool begin(const km_command_t& command) {
            if (args_.broadcast >= KM_BROADCASTS || args_.group != command.group) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                return false;
            }
            if (!included()) {
                return true;
            }
            if (args_.window == 0 && args_.pages == 0) {  // no buffer: words alone
                if (
                    args_.receivers >= KM_BROADCAST_RANKS ||
                    args_.rank > args_.receivers || args_.noc != noc_index
                ) {
                    fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                    return false;
                }
                return true;
            }
            const LocalCBInterface& circular_buffer = get_local_cb_interface(args_.cb);
            block_bytes_ = args_.pages * circular_buffer.fifo_page_size;
            if (
                args_.window == 0 || args_.window > kBroadcastWindowMax ||
                args_.pages == 0 || args_.receivers >= KM_BROADCAST_RANKS ||
                args_.rank > args_.receivers || args_.noc != noc_index ||
                circular_buffer.fifo_num_pages != args_.window * args_.pages ||
                circular_buffer.fifo_num_pages > 0xFFFFu ||
                circular_buffer.fifo_size != args_.window * block_bytes_
            ) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                return false;
            }
            if (!wait_free(circular_buffer.fifo_num_pages, 0)) {
                return false;
            }
            if (
                get_write_ptr(args_.cb) !=
                circular_buffer.fifo_limit - circular_buffer.fifo_size
            ) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                return false;
            }
            return true;
        }

        volatile tt_l1_ptr uint32_t* area_word(uint32_t offset) const {
            return l1(area_ + offset);
        }

        uint32_t slot(uint32_t block) const {
            const LocalCBInterface& circular_buffer = get_local_cb_interface(args_.cb);
            return (
                circular_buffer.fifo_limit - circular_buffer.fifo_size +
                (block % args_.window) * block_bytes_
            );
        }

        uint64_t multicast_address(uint32_t address) const {
            return (
                get_noc_multicast_addr(
                    args_.mc_start & 0xFFFFu,
                    args_.mc_start >> 16,
                    args_.mc_end & 0xFFFFu,
                    args_.mc_end >> 16,
                    address
                )
            );
        }

        // Blocks of this command compute has popped: those pushed less the pages still
        // held, a 16-bit difference that cannot wrap (under 2^16 pages, padding after).
        uint32_t popped() const {
            const uint32_t acked = (
                reg_read(reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(args_.cb)))
            );
            const uint16_t held = (
                static_cast<uint16_t>(get_cb_tiles_received_ptr(args_.cb)[0] - acked)
            );
            return (block_count_ * args_.pages - held) / args_.pages;
        }

        // Receiver: a DONE marker to the sender when compute popped more blocks.
        void marker_freed() {
            if (sender() || args_.pages == 0) {
                return;
            }
            const uint32_t popped_blocks = popped();
            if (popped_blocks == marked_) {
                return;
            }
            if (!flushed(
                kFlushSent, kWhereBroadcastFree, popped_blocks
            )) {  // the last marker has left KM_BROADCAST_OUT
                return;
            }
            *area_word(KM_BROADCAST_OUT) = broadcast_marker(sequence_, popped_blocks);
            noc_async_write(
                area_ + KM_BROADCAST_OUT,
                core_.core_noc(
                    args_.sender, area_ + KM_BROADCAST_DONE + 16 * args_.rank
                ),
                4
            );
            marked_ = popped_blocks;
        }

        // Sender: waits (with a timeout) until every receiver's DONE word (or the word
        // `word` of its slot) counts `blocks`.
        bool wait_done(uint32_t blocks, uint32_t block, uint32_t word = 0) {
            uint32_t receiver = 1;
            const uint64_t started = wall64();
            const uint32_t code = core_.poll(
                [&] {
                    while (
                        receiver <= args_.receivers &&
                        broadcast_reached(
                            area_word(KM_BROADCAST_DONE + 16 * receiver)[word],
                            sequence_,
                            blocks
                        )
                    ) {
                        receiver++;
                    }
                    return receiver > args_.receivers;
                },
                core_.args().wait_limit
            );
            add_wait(wall64() - started);
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastDone, block);
                return false;
            }
            return true;
        }

        // Timed NoC barrier of this kernel: reads landed, writes sent from L1, or
        // writes acknowledged. A stop flag set meanwhile gives it one more timeout.
        bool flushed(uint32_t what, uint32_t where, uint32_t block) {
            const uint32_t code = core_.poll(
                [&] {
                    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
                        return (
                            what == kFlushReads
                                ? ncrisc_dynamic_noc_reads_flushed(noc_index)
                                : what == kFlushSent
                                    ? ncrisc_dynamic_noc_nonposted_writes_sent(
                                        noc_index
                                    )
                                    : ncrisc_dynamic_noc_nonposted_writes_flushed(
                                        noc_index
                                    )
                        );
                    } else {
                        return (
                            what == kFlushReads
                                ? ncrisc_noc_reads_flushed(noc_index)
                                : what == kFlushSent
                                    ? ncrisc_noc_nonposted_writes_sent(noc_index)
                                    : ncrisc_noc_nonposted_writes_flushed(noc_index)
                        );
                    }
                },
                core_.args().wait_limit,
                core_.args().wait_limit
            );
            if (code != kGaveNone) {
                fail(code, where, block);
                return false;
            }
            return true;
        }

        // Waits (with a timeout) until `pages` pages of the broadcast buffer are free.
        bool wait_free(uint32_t pages, uint32_t block) {
            const uint32_t code = core_.poll(
                [&] {
                    marker_freed();
                    return cb_pages_reservable_at_back(args_.cb, pages);
                },
                core_.args().wait_limit
            );
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastFree, block);
                return false;
            }
            return true;
        }

        bool in_order(uint32_t block) {
            if (
                !ok_ || !included() || block != block_count_ ||
                block >= kBroadcastBlocksMax
            ) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, block);
                return false;
            }
            return true;
        }

        void add_wait(uint64_t cycles) const {
            volatile tt_l1_ptr uint32_t* stats = area_word(KM_BROADCAST_STATS);
            const uint64_t waited = (
                (static_cast<uint64_t>(stats[2]) << 32 | stats[1]) + cycles
            );
            stats[1] = static_cast<uint32_t>(waited);
            stats[2] = static_cast<uint32_t>(waited >> 32);
        }

        // The first broadcast give-up of this core names broadcast and block; the tile
        // stops.
        void fail(uint32_t code, uint32_t where, uint32_t block) {
            volatile tt_l1_ptr uint32_t* gave = area_word(KM_BROADCAST_GAVE);
            if (gave[0] == 0) {
                gave[0] = where | args_.broadcast << 8;
                gave[1] = block;
                gave[2] = sequence_;
            }
            core_.gave_up(word_, code, where, slot_, sequence_);
            ok_ = false;
        }

        const Core& core_;
        const uint32_t word_;
        const uint32_t sequence_;
        const uint32_t slot_;
        BroadcastArgs args_;
        uint32_t area_ = 0;
        uint32_t block_bytes_ = 0;
        uint32_t block_count_ = 0;
        bool reserved_ = false;
        uint32_t marked_ = 0;
        uint32_t word_blocks_ = 0;  // sender: blocks up to the last word cast, 0: taken
        uint32_t received_ = 0;
        bool ok_ = false;
    };

}  // namespace km

#else
namespace km {

    // Compute: a block's pages, by index 0 to pages - 1 (timed wait).
    inline bool broadcast_receive(const Compute& compute, uint32_t cb, uint32_t pages) {
        return compute.wait_front(cb, pages);
    }

    // Compute: done with the block; the broadcast's data-movement kernel marks the
    // sender.
    inline void broadcast_done(uint32_t cb, uint32_t pages) {
        cb_pop_front(cb, pages);
    }

    // Compute, after the command's last block: pops the padding that finish() pushes.
    inline bool broadcast_end(
        const Compute& compute,
        uint32_t cb,
        uint32_t pages,
        uint32_t window,
        uint32_t blocks
    ) {
        const uint32_t pad = (window - blocks % window) % window * pages;
        if (pad == 0) {
            return true;
        }
        if (!compute.wait_front(cb, pad)) {
            return false;
        }
        cb_pop_front(cb, pad);
        return true;
    }

}  // namespace km
#endif
