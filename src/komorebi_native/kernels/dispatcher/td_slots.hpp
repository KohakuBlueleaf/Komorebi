// The Tensix dispatcher's busy slots, shared by its two loops through the hand-over in
// L1 (dispatcher.h, kTx*): "out" fills a slot, "in" sweeps it and frees it.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

    // A busy slot's command, written by out before the slot's issued word.
    struct Record {
        uint32_t request;  // the host's command number
        uint32_t sequence;
        uint32_t group;
        uint32_t flags;  // kRecord*
        uint32_t submitted_lo, submitted_hi;
        uint32_t deadline_lo, deadline_hi;  // ticks from its submit
    };

    static_assert(sizeof(Record) == 32, "a record is 32 bytes");
    constexpr uint32_t kRecordResult = 1;  // KM_FLAG_RESULT: the ack's second word, ticks
    constexpr uint32_t kRecordSilent = 2;  // a dynamic run's end: its slot freed, no event
    constexpr uint32_t kRecordGiveUp = 4;  // given up at its deadline, the run going on
    constexpr uint32_t kRecordThread = 8;  // the thread's (source 1): its event to its ring
    constexpr uint32_t kRecordEngine = 16;  // the engines' (source 2): its event to theirs

    // The hand-over at `hand` (the block's kTbHand). A word has one writer; a reader
    // invalidates its L1 cache, then reads the word that publishes, then the rest.
    struct Hand {
        uint32_t base = 0;

        volatile tt_l1_ptr uint32_t* issued() const {
            return km::l1(base + kTxIssued);
        }

        volatile tt_l1_ptr uint32_t* done() const {
            return km::l1(base + kTxDone);
        }

        volatile tt_l1_ptr Record* record(uint32_t slot) const {
            return (
                reinterpret_cast<volatile tt_l1_ptr Record*>(
                    base + kTxRecords + slot * sizeof(Record)
                )
            );
        }

        volatile tt_l1_ptr uint32_t* event(uint32_t index) const {
            return km::l1(base + kTxEvents + (index % kTxEventsMax) * 16);
        }

        volatile tt_l1_ptr uint32_t& word(uint32_t which) const {
            return km::l1(base + kTxWords)[which];
        }
    };

    inline uint64_t wide(uint32_t lo, uint32_t hi) {
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

    // Out's side: which slots are free, and filling one. Out keeps its own copy of what
    // it issued in local memory and reads only in's done words from L1.
    struct OutSlots {
        Hand hand;
        uint32_t depth = 0, in_flight = 0, most = 0, issued_count = 0;
        uint32_t busy_bits[2] = {0, 0};  // slots 0-31, 32-63
        uint32_t sequence[KM_RING_SLOTS];
        uint32_t group[KM_RING_SLOTS];

        void init(uint32_t slots, Hand at) {
            depth = slots;
            hand = at;
            for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
                sequence[slot] = 0;
                group[slot] = 0;
            }
        }

        bool busy(uint32_t slot) const {
            return (busy_bits[slot >> 5] >> (slot & 31u)) & 1u;
        }

        // Frees each busy slot in has freed (after an L1 invalidate); returns how many.
        uint32_t refresh() {
            uint32_t freed = 0;
            volatile tt_l1_ptr uint32_t* done = hand.done();
            for (uint32_t half = 0; half < 2; half++) {
                uint32_t bits = busy_bits[half];
                for (uint32_t slot = half * 32; bits != 0; slot++, bits >>= 1) {
                    if ((bits & 1u) && done[slot] == sequence[slot]) {
                        busy_bits[half] &= ~(1u << (slot & 31u));
                        freed++;
                    }
                }
            }
            in_flight -= freed;
            return freed;
        }

        // The first free slot, or depth when every slot is busy.
        uint32_t free_slot() const {
            uint32_t slot = 0;
            while (slot < depth && busy(slot)) {
                slot++;
            }
            return slot;
        }

        // Whether a busy slot's group shares a core with `other`.
        bool overlaps(uint32_t other) const {
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (busy(slot) && km::groups_overlap(group[slot], other)) {
                    return true;
                }
            }
            return false;
        }

        // The slot's last sequence: a new one differs from it.
        uint32_t last(uint32_t slot) const {
            return sequence[slot];
        }

        void take(
            uint32_t slot,
            uint32_t request,
            uint32_t sequence_number,
            uint32_t group_word,
            uint32_t flags,
            uint64_t at,
            uint64_t deadline
        ) {  // fmt: skip
            volatile tt_l1_ptr Record* record = hand.record(slot);
            record->request = request;
            record->sequence = sequence_number;
            record->group = group_word;
            record->flags = flags;
            record->submitted_lo = static_cast<uint32_t>(at);
            record->submitted_hi = static_cast<uint32_t>(at >> 32);
            record->deadline_lo = static_cast<uint32_t>(deadline);
            record->deadline_hi = static_cast<uint32_t>(deadline >> 32);
            asm volatile("" ::: "memory");
            hand.issued()[slot] = sequence_number;  // the record is the slot's from
                                                    // here
            sequence[slot] = sequence_number;
            group[slot] = group_word;
            busy_bits[slot >> 5] |= 1u << (slot & 31u);
            issued_count++;
            in_flight++;
            most = in_flight > most ? in_flight : most;
        }

        // Busy slots 0 to 31 as bits.
        uint32_t busy_low() const {
            return busy_bits[0];
        }
    };

    // In's side: the sweep of each busy slot's ack word, then its deadline.
    struct InSlots {
        Hand hand;
        uint32_t depth = 0, freed = 0;
        uint32_t seen[KM_RING_SLOTS];  // the sequence in last freed in each slot
        volatile tt_l1_ptr uint32_t* ack = nullptr;

        void init(uint32_t slots, Hand at, volatile tt_l1_ptr uint32_t* acks) {
            depth = slots;
            hand = at;
            ack = acks;
            for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
                seen[slot] = 0;
            }
        }

        // Each acknowledged slot: done(request, status, ticks, flags), then freed; a
        // slot past its deadline: late(slot, flags), false when late ends the run.
        template <typename Done, typename Late>
        bool sweep(uint64_t now, Done done, Late late) {
            volatile tt_l1_ptr uint32_t* issued = hand.issued();
            for (uint32_t slot = 0; slot < depth; slot++) {
                const uint32_t sequence = issued[slot];
                if (sequence == seen[slot]) {
                    continue;
                }
                volatile tt_l1_ptr Record* record = hand.record(slot);
                const uint64_t at = wide(record->submitted_lo, record->submitted_hi);
                const uint64_t since = now > at ? now - at : 0;  // out's clock may lead
                if (ack[slot * 4] == KM_ACK(sequence)) {
                    const uint32_t flags = record->flags;
                    const bool result = (flags & kRecordResult) != 0;
                    const uint32_t ticks = (
                        result ? ack[slot * 4 + 1] : static_cast<uint32_t>(since)
                    );
                    const uint32_t status = (
                        KM_EVENT_DONE | (record->group & 0xFFu) << 8 |
                        (sequence & KM_EVENT_SEQUENCE_MASK) << 16
                    );
                    const uint32_t request = record->request;
                    seen[slot] = sequence;
                    hand.done()[slot] = sequence;
                    freed++;
                    done(request, status, ticks, flags);
                } else if (
                    since > wide(record->deadline_lo, record->deadline_hi) &&
                    late(slot, record->flags)
                ) {
                    return false;
                }
            }
            return true;
        }
    };

}  // namespace td
