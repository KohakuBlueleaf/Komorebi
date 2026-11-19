// The Tensix dispatcher's "in" loop: the slots' sweep, the events and the host words.
// On NCRISC beside "out" (dispatcher.cpp), or inside out's loop.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/tensors/km_transfer.hpp"
#include "td_slots.hpp"

namespace td {

    inline uint32_t arg(uint32_t i) {
        return get_arg_val<uint32_t>(i);
    }

    // Empty cycles between two polls of L1 (the movers share L1 with the TRISCs).
    inline void pace() {
        for (uint32_t i = 0; i < kTdPace; i++) {
            asm volatile("nop");
        }
    }

    // The host channel's NoC address on this processor's NoC: Blackhole's PCIe
    // coordinates are translated (either NoC), Wormhole's NoC 0's, mirrored on NoC 1.
    inline uint64_t pcie() {
        if constexpr (km::kPcieEitherNoc) {
            return km::pcie_noc();
        }
        return (
            static_cast<uint64_t>(NOC_XY_PCIE_ENCODING(
                NOC_X_PHYS_COORD(PCIE_NOC_X), NOC_Y_PHYS_COORD(PCIE_NOC_Y)
            ))
        );
    }

    // Whether the block argument lies in this core's own buffer (kTdBlockCb), as the
    // host placed it; else the core writes nothing there and ends.
    inline bool block_placed(uint32_t block) {
        const uint32_t at = get_write_ptr(kTdBlockCb);
        return block >= at && block - at < 64;
    }

    // The slots in use: the depth argument, 0 or too many meaning every slot.
    inline uint32_t depth_arg() {
        const uint32_t depth = arg(kTdDepth);
        return depth == 0 || depth > KM_RING_SLOTS ? KM_RING_SLOTS : depth;
    }

    struct In {
        Hand hand;
        InSlots slots;
        uint32_t block = 0, event_depth = 0;
        uint64_t area = 0;
        uint32_t pushed = 0, told = 0, published = 0, taken = 0;
        uint32_t results = 0, last_result = 0, passes = 0, sweeps = 0;
#ifdef KM_TD_ACK_COUNT
        uint32_t acked = 0;  // the workers' ack count at the last sweep
#endif
        uint64_t last_event = 0;
        bool late = false;
        volatile tt_l1_ptr uint32_t* words = nullptr;
        volatile tt_l1_ptr uint32_t* events = nullptr;
        volatile tt_l1_ptr uint32_t* result = nullptr;

        void init() {
            block = arg(kTdBlock);
            event_depth = arg(kTdEventDepth);
            area = pcie() | arg(kTdArea);
            hand.base = block + kTbHand;
            words = km::l1(block + kTbWords);
            events = km::l1(block + kTbEvents);
            result = km::l1(block + kTbResult);
            slots.init(depth_arg(), hand, km::l1(arg(kTdControl) + KM_CONTROL_DONE));
        }

        // The host words {state, consumed, events, error}: one 16-byte write once the
        // writes before it left (an event before the count that tells it).
        void put_words(uint32_t consumed) {
            noc_async_writes_flushed();
            words[KMD_HOST_STATE] = KMD_STATE_READY;
            words[KMD_HOST_CONSUMED] = consumed;
            words[KMD_HOST_EVENTS] = pushed;
            words[KMD_HOST_ERROR] = KMD_ERROR_NONE;
            noc_async_write_pcie(block + kTbWords, area, 16);
            published = consumed;
            told = pushed;
        }

        // Writes event {index, request, status, ticks} into the host's ring: ++pushed,
        // or for the self-fed bench's own commands the next one's place, index 0.
        void push(uint32_t request, uint32_t status, uint32_t ticks) {
#ifdef KM_TD_SELF_FEED
            const uint32_t index = request == 0 ? pushed + 1 : ++pushed;
#else
            const uint32_t index = ++pushed;
#endif
            const uint32_t at = (index & (event_depth - 1u)) * 16;
            volatile tt_l1_ptr uint32_t* event = events + at / 4;
            event[0] = request != 0 ? index : 0;  // index 0: the host never reads it
            event[1] = request;
            event[2] = status;
            event[3] = ticks;
            noc_async_write_pcie(block + kTbEvents + at, area + KMD_HOST_RING + at, 16);
        }

        // One pass: the sweep, the events out posted, the host words when they changed.
        // Whether it did anything.
        bool pass(uint64_t now) {
            bool active = false;
            passes++;
#ifdef KM_TD_ACK_COUNT
            // no sweep while the workers' count stands, but one each 64 passes (a
            // deadline, an ack whose add landed first)
            const uint32_t count = slots.ack[kTdAckCountAt / 4];
            const bool look = count != acked || (passes & 63u) == 0;
            acked = count;
#else
            const bool look = true;
#endif
            if (!late && look) {
                sweeps++;
                late = !slots.sweep(
                    now,
                    [&](uint32_t request, uint32_t status, uint32_t ticks, bool with) {
                        push(request, status, ticks);
                        if (with) {
                            results++;
                            last_result = ticks;
                        }
                        last_event = now;
                        active = true;
                    },
                    [&](uint32_t slot) {
                        volatile tt_l1_ptr Record* record = hand.record(slot);
                        result[KMD_R_SLOT] = slot;
                        result[KMD_R_SEQUENCE] = record->sequence;
                        result[KMD_R_REQ] = record->request;
                        push(
                            record->request,
                            KM_EVENT_TIMEOUT | slot << 8 |
                                (record->sequence & KM_EVENT_SEQUENCE_MASK) << 16,
                            0
                        );
                        hand.word(kTwLate) = slot + 1;
                    }
                );
            }
            const uint32_t posted = hand.word(kTwPosted);
            while (taken != posted) {
                volatile tt_l1_ptr uint32_t* event = hand.event(taken);
                push(event[0], event[1], event[2]);
                hand.word(kTwTaken) = ++taken;
                last_event = now;
                active = true;
            }
            const uint32_t consumed = hand.word(kTwConsumed);
            if (consumed != published || pushed != told) {
                put_words(consumed);
            }
            return active;
        }

        // The end: every write landed, then the final words for out's result block.
        void over() {
            noc_async_write_barrier();
            hand.word(kTwPushed) = pushed;
            hand.word(kTwResults) = results;
            hand.word(kTwLastResult) = last_result;
            hand.word(kTwLastLo) = static_cast<uint32_t>(last_event);
            hand.word(kTwLastHi) = static_cast<uint32_t>(last_event >> 32);
            hand.word(kTwPasses) = passes;
            hand.word(kTwSweeps) = sweeps;
            asm volatile("" ::: "memory");
            hand.word(kTwOver) = arg(kTdTag);
        }
    };

}  // namespace td
