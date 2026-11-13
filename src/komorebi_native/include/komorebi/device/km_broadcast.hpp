// Broadcast: a sender multicasts block k into slot k mod B of every receiver's buffer;
// a receiver writes its marker in the sender's DONE-BY table once compute popped it.
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
        uint32_t pages;      // P: pages per block
    };

    constexpr uint32_t kBroadcastArgs = sizeof(BroadcastArgs) / 4;

    // A marker: the command's sequence number (low 16 bits) and a count of blocks.
    inline uint32_t broadcast_marker(uint32_t seq, uint32_t blocks) {
        return (seq & 0xFFFFu) << 16 | (blocks & 0xFFFFu);
    }

    // True when marker v belongs to command `seq` and counts at least `blocks`.
    inline bool broadcast_reached(uint32_t v, uint32_t seq, uint32_t blocks) {
        return (
            (v >> 16) == (seq & 0xFFFFu) &&
            static_cast<int16_t>(static_cast<uint16_t>(v - blocks)) >= 0
        );
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // One broadcast in one command, in the DM kernel on the broadcast's NoC. In block
    // order: the sender reserve(k), fill, cast(k); a receiver arrive(k); then finish().
    class Broadcast {
    public:
        Broadcast(const Reader& k, const km_command_t& r, uint32_t arg0)
            : Broadcast(static_cast<const Core&>(k), kStReader, r, arg0) {
        }

        Broadcast(const Writer& k, const km_command_t& r, uint32_t arg0)
            : Broadcast(static_cast<const Core&>(k), kStWriter, r, arg0) {
        }

        // False: the arguments, the command or the buffer did not fit (recorded).
        bool ok() const {
            return ok_;
        }

        uint32_t rank() const {
            return a_.rank;
        }

        bool sender() const {
            return a_.rank == 0;
        }

        bool included() const {
            return a_.rank != kBroadcastNone;
        }

        // P, pages per block, and B, blocks in flight.
        uint32_t pages() const {
            return a_.pages;
        }

        uint32_t window() const {
            return a_.window;
        }

        // Blocks multicast or arrived in this command.
        uint32_t blocks() const {
            return n_;
        }

        // Sender: timed wait until its compute freed slot k mod B; returns the slot's
        // L1 address (the same on every core of the group), 0 after a give-up.
        uint32_t reserve(uint32_t k) {
            if (!in_order(k)) {
                return 0;
            }
            if (!wait_free(a_.pages, k)) {
                return 0;
            }
            if (get_write_ptr(a_.cb) != slot(k) ||
                !item_fits(a_.cb, get_write_ptr(a_.cb), a_.pages)) {
                fail(kGaveBadItem, kWhereBroadcastFree, k);
                return 0;
            }
            cb_reserve_back(a_.cb, a_.pages);
            reserved_ = true;
            return slot(k);
        }

        // Sender, block k filled: timed wait until every receiver is done with block
        // k - B, multicasts it, barrier, multicasts VALID, pushes it to compute.
        bool cast(uint32_t k) {
            if (!reserved_) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, k);
                return false;
            }
            if (!in_order(k)) {
                return false;
            }
            if (!flushed(
                    kFlushReads, kWhereBroadcastDone, k
                )) {  // the block's reads have landed
                return false;
            }
            if (k >= a_.window && !wait_done(k - a_.window + 1, k)) {
                return false;
            }
            if (a_.receivers != 0) {
                const uint32_t src = slot(k);
                noc_async_write_multicast(src, mc(src), block_bytes_, a_.receivers);
                if (!flushed(
                        kFlushWrites, kWhereBroadcastDone, k
                    )) {  // block k landed before its VALID
                    return false;
                }
                *at(KM_BROADCAST_OUT) = broadcast_marker(seq_, k + 1);
                noc_semaphore_set_multicast(
                    area_ + KM_BROADCAST_OUT,
                    mc(area_ + KM_BROADCAST_VALID),
                    a_.receivers
                );
            }
            cb_push_back(a_.cb, a_.pages);
            reserved_ = false;
            n_++;
            return true;
        }

        // Receiver: waits (with a timeout) for VALID >= k + 1, marking the blocks its
        // compute popped meanwhile, then pushes block k to compute.
        bool arrive(uint32_t k) {
            if (!in_order(k)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* valid = at(KM_BROADCAST_VALID);
            const uint64_t t0 = wall64();
            const uint32_t code = c_.poll(
                [&] {
                    marker_freed();
                    return broadcast_reached(*valid, seq_, k + 1);
                },
                c_.args().wait_limit
            );
            add_wait(wall64() - t0);
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastValid, k);
                return false;
            }
            if (!wait_free(a_.pages, k)) {
                return false;
            }
            if (get_write_ptr(a_.cb) != slot(k) ||
                !item_fits(a_.cb, get_write_ptr(a_.cb), a_.pages)) {
                fail(kGaveBadItem, kWhereBroadcastFree, k);
                return false;
            }
            cb_reserve_back(a_.cb, a_.pages);
            cb_push_back(a_.cb, a_.pages);
            n_++;
            return true;
        }

        // After the last block (timed): compute popped all, padding to whole windows,
        // last marker; a sender that multicast waits for all receivers'. Clears words.
        bool finish() {
            if (!ok_ || !included()) {
                return ok_;
            }
            if (!sender()) {
                *at(KM_BROADCAST_VALID) = 0;  // every VALID of this command has arrived
            }
            const uint32_t code = c_.poll(
                [&] {
                    marker_freed();
                    return popped() == n_;
                },
                c_.args().wait_limit
            );
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastFree, n_);
                return false;
            }
            marker_freed();  // the last pop may have come after the last marker
            if (!ok_) {
                return false;
            }
            const uint32_t pad = (a_.window - n_ % a_.window) % a_.window * a_.pages;
            if (pad != 0) {
                if (!item_fits(a_.cb, get_write_ptr(a_.cb), pad)) {
                    fail(kGaveBadItem, kWhereBroadcastFree, n_);
                    return false;
                }
                cb_reserve_back(a_.cb, pad);  // free: every block was popped
                cb_push_back(a_.cb, pad);
            }
            if (sender() && n_ != 0) {  // no block multicast: no receiver markers
                if (!wait_done(n_, n_)) {
                    return false;
                }
                for (uint32_t q = 1; q <= a_.receivers; q++) {
                    *at(KM_BROADCAST_DONE + 16 * q) = 0;
                }
            }
            if (!flushed(kFlushWrites, kWhereBroadcastFree, n_)) {
                return false;
            }
            add_sat(at(KM_BROADCAST_STATS), n_);
            return true;
        }

    private:
        Broadcast(const Core& c, uint32_t word, const km_command_t& r, uint32_t arg0)
            : c_(c), word_(word), seq_(r.seq), slot_(km::slot(r)) {
            uint32_t* f = &a_.broadcast;
            for (uint32_t i = 0; i < kBroadcastArgs; i++) {
                f[i] = c.arg(arg0 + i);
            }
            area_ =
                (c.args().ctl + KM_CTL_BROADCAST +
                 (a_.broadcast % KM_BROADCASTS) * KM_BROADCAST_BYTES);
            ok_ = begin(r);
        }

        // Checks the arguments against the command and the buffer, then a timed wait
        // until the buffer is empty at slot 0 (the last command's padding was popped).
        bool begin(const km_command_t& r) {
            if (a_.broadcast >= KM_BROADCASTS || a_.group != r.group) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                return false;
            }
            if (!included()) {
                return true;
            }
            const LocalCBInterface& i = get_local_cb_interface(a_.cb);
            block_bytes_ = a_.pages * i.fifo_page_size;
            if (a_.window == 0 || a_.window > kBroadcastWindowMax || a_.pages == 0 ||
                a_.receivers >= KM_BROADCAST_RANKS || a_.rank > a_.receivers ||
                a_.noc != noc_index || i.fifo_num_pages != a_.window * a_.pages ||
                i.fifo_num_pages > 0xFFFFu || i.fifo_size != a_.window * block_bytes_) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                return false;
            }
            if (!wait_free(i.fifo_num_pages, 0)) {
                return false;
            }
            if (get_write_ptr(a_.cb) != i.fifo_limit - i.fifo_size) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, 0);
                return false;
            }
            return true;
        }

        volatile tt_l1_ptr uint32_t* at(uint32_t off) const {
            return l1(area_ + off);
        }

        uint32_t slot(uint32_t k) const {
            const LocalCBInterface& i = get_local_cb_interface(a_.cb);
            return i.fifo_limit - i.fifo_size + (k % a_.window) * block_bytes_;
        }

        uint64_t mc(uint32_t addr) const {
            return (
                get_noc_multicast_addr(
                    a_.mc_start & 0xFFFFu,
                    a_.mc_start >> 16,
                    a_.mc_end & 0xFFFFu,
                    a_.mc_end >> 16,
                    addr
                )
            );
        }

        // Blocks of this command compute has popped: those pushed less the pages still
        // held, a 16-bit difference that cannot wrap (under 2^16 pages, padding after).
        uint32_t popped() const {
            const uint32_t acked = (
                reg_read(reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(a_.cb)))
            );
            const uint16_t held = (
                static_cast<uint16_t>(get_cb_tiles_received_ptr(a_.cb)[0] - acked)
            );
            return (n_ * a_.pages - held) / a_.pages;
        }

        // Receiver: a DONE marker to the sender when compute popped more blocks.
        void marker_freed() {
            if (sender()) {
                return;
            }
            const uint32_t b = popped();
            if (b == marked_) {
                return;
            }
            if (!flushed(
                    kFlushSent, kWhereBroadcastFree, b
                )) {  // the last marker has left KM_BROADCAST_OUT
                return;
            }
            *at(KM_BROADCAST_OUT) = broadcast_marker(seq_, b);
            noc_async_write(
                area_ + KM_BROADCAST_OUT,
                c_.core_noc(a_.sender, area_ + KM_BROADCAST_DONE + 16 * a_.rank),
                4
            );
            marked_ = b;
        }

        // Sender: waits (with a timeout) until every receiver's DONE counts `blocks`.
        bool wait_done(uint32_t blocks, uint32_t k) {
            uint32_t q = 1;
            const uint64_t t0 = wall64();
            const uint32_t code = c_.poll(
                [&] {
                    while (
                        q <= a_.receivers &&
                        broadcast_reached(*at(KM_BROADCAST_DONE + 16 * q), seq_, blocks)
                    ) {
                        q++;
                    }
                    return q > a_.receivers;
                },
                c_.args().wait_limit
            );
            add_wait(wall64() - t0);
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastDone, k);
                return false;
            }
            return true;
        }

        // Timed NoC barrier of this kernel: reads landed, writes sent from L1, or
        // writes acknowledged. A stop flag set meanwhile gives it one more timeout.
        bool flushed(uint32_t what, uint32_t where, uint32_t k) {
            const uint32_t code = c_.poll(
                [&] {
                    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
                        return (
                            what == kFlushReads
                                ? ncrisc_dynamic_noc_reads_flushed(noc_index)
                                : what == kFlushSent
                                ? ncrisc_dynamic_noc_nonposted_writes_sent(noc_index)
                                : ncrisc_dynamic_noc_nonposted_writes_flushed(noc_index)
                        );
                    } else {
                        return (
                            what == kFlushReads ? ncrisc_noc_reads_flushed(noc_index)
                                : what == kFlushSent
                                ? ncrisc_noc_nonposted_writes_sent(noc_index)
                                : ncrisc_noc_nonposted_writes_flushed(noc_index)
                        );
                    }
                },
                c_.args().wait_limit,
                c_.args().wait_limit
            );
            if (code != kGaveNone) {
                fail(code, where, k);
                return false;
            }
            return true;
        }

        // Waits (with a timeout) until `pages` pages of the broadcast buffer are free.
        bool wait_free(uint32_t pages, uint32_t k) {
            const uint32_t code = c_.poll(
                [&] {
                    marker_freed();
                    return cb_pages_reservable_at_back(a_.cb, pages);
                },
                c_.args().wait_limit
            );
            if (code != kGaveNone) {
                fail(code, kWhereBroadcastFree, k);
                return false;
            }
            return true;
        }

        bool in_order(uint32_t k) {
            if (!ok_ || !included() || k != n_ || k >= kBroadcastBlocksMax) {
                fail(kGaveBadBroadcast, kWhereBroadcastBegin, k);
                return false;
            }
            return true;
        }

        void add_wait(uint64_t cycles) const {
            volatile tt_l1_ptr uint32_t* s = at(KM_BROADCAST_STATS);
            const uint64_t w = (static_cast<uint64_t>(s[2]) << 32 | s[1]) + cycles;
            s[1] = static_cast<uint32_t>(w);
            s[2] = static_cast<uint32_t>(w >> 32);
        }

        // The first broadcast give-up of this core names broadcast and block; the tile
        // stops.
        void fail(uint32_t code, uint32_t where, uint32_t k) {
            volatile tt_l1_ptr uint32_t* g = at(KM_BROADCAST_GAVE);
            if (g[0] == 0) {
                g[0] = where | a_.broadcast << 8;
                g[1] = k;
                g[2] = seq_;
            }
            c_.gave_up(word_, code, where, slot_, seq_);
            ok_ = false;
        }

        const Core& c_;
        const uint32_t word_;
        const uint32_t seq_;
        const uint32_t slot_;
        BroadcastArgs a_;
        uint32_t area_ = 0;
        uint32_t block_bytes_ = 0;
        uint32_t n_ = 0;
        bool reserved_ = false;
        uint32_t marked_ = 0;
        bool ok_ = false;
    };

}  // namespace km

#else
namespace km {

    // Compute: block k's pages, by index 0 to pages - 1 (timed wait).
    inline bool broadcast_receive(const Compute& cp, uint32_t cb, uint32_t pages) {
        return cp.wait_front(cb, pages);
    }

    // Compute: done with the block; the broadcast's data-movement kernel marks the
    // sender.
    inline void broadcast_done(uint32_t cb, uint32_t pages) {
        cb_pop_front(cb, pages);
    }

    // Compute, after the command's last block: pops the padding that finish() pushes.
    inline bool broadcast_end(
        const Compute& cp, uint32_t cb, uint32_t pages, uint32_t window, uint32_t blocks
    ) {
        const uint32_t pad = (window - blocks % window) % window * pages;
        if (pad == 0) {
            return true;
        }
        if (!cp.wait_front(cb, pad)) {
            return false;
        }
        cb_pop_front(cb, pad);
        return true;
    }

}  // namespace km
#endif
