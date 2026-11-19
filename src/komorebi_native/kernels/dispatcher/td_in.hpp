// The Tensix dispatcher's "in" loop: the slots' sweep, the events and the host words.
// On NCRISC beside "out" (dispatcher.cpp), or inside out's loop.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/tensors/km_transfer.hpp"
#include "td_messages.hpp"
#include "td_slots.hpp"
#include "td_sources.hpp"

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
        uint32_t scale = 0;  // a thread's clock (thread_ticks): its events' times
        uint64_t area = 0;
        uint32_t pushed = 0, told = 0, published = 0, taken = 0;
        ThreadArea works[kClusters];  // the threads' (sources 1 to 3): their events
        ThreadArea engines;           // the engines' (source 4)
        uint32_t thread_told[kClusters] = {}, engines_told = 0;
        uint32_t link = 0, link_told = 0;  // the link block (0: none), its events
        uint32_t results = 0, last_result = 0, passes = 0, sweeps = 0;
#ifdef KM_TD_ACK_COUNT
        uint32_t acked = 0;  // the workers' ack count at the last sweep
#endif
        uint64_t last_event = 0;
        bool late = false;
        // the dynamic run in flight (out's kTwRun* words) while its end is not yet owed
        uint32_t run = 0, run_slot = 0, run_group = 0, checks = 0, quiet_at = 0;
        bool run_on = false;
        // the deadline being given up (out's inspect, abort and leave): its sequence
        uint32_t giving = 0, giving_slot = 0, given_ups = 0;
        // the watch's stalls reported (KMD_EVENT_STALL): SEEN and BACK
        uint32_t stalls_seen = 0, stalls_back = 0;
        volatile tt_l1_ptr uint32_t* watch = nullptr;
        volatile tt_l1_ptr uint32_t* words = nullptr;
        volatile tt_l1_ptr uint32_t* events = nullptr;
        volatile tt_l1_ptr uint32_t* result = nullptr;

        void init() {
            block = arg(kTdBlock);
            event_depth = arg(kTdEventDepth);
            scale = arg(kTdThreadScale);
            area = pcie() | arg(kTdArea);
            hand.base = block + kTbHand;
            words = km::l1(block + kTbWords);
            events = km::l1(block + kTbEvents);
            result = km::l1(block + kTbResult);
            watch = km::l1(block + kTbWatch);
            for (uint32_t c = 0; c < kClusters; c++) {
                const uint32_t tile = thread_tile_arg(c + 1);
                works[c].base = thread_area(block, c + 1) + kTaWork;
                works[c].event_depth = kTaEventDepth;
                works[c].place(arg(tile), arg(tile + 1));
            }
            engines.base = block + kTbBytes + kTaBytes + kTeWork;
            engines.event_depth = kTeEventDepth;
            link = arg(kTdLinkIsland) != 0 ? arg(kTdLinkBase) : 0;
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

        // An event into its source's ring: the thread's or the engines' (words, then
        // index; out counted its room), the link's (its block, for out), else host's.
        void push_to(
            uint32_t source, uint32_t request, uint32_t status, uint32_t ticks
        ) {
            if (source == kSourceLink && link != 0) {
                const uint32_t index = ++link_told;
                volatile tt_l1_ptr uint32_t* event = (
                    km::l1(link + kTlEvents + (index % kTlEventDepth) * 16)
                );
                event[0] = index;
                event[1] = request;
                event[2] = status;
                event[3] = ticks;
                asm volatile("" ::: "memory");
                km::l1(link + kTlTold)[0] = index;
                return;
            }
            const bool thread = is_thread(source);
            if (!thread && source != kSourceEngine) {
                push(request, status, ticks);
                return;
            }
            const uint32_t c = thread ? source - kSourceThread : 0;
            const ThreadArea& to = thread ? works[c] : engines;
            uint32_t& told_count = thread ? thread_told[c] : engines_told;
            const uint32_t index = ++told_count;
            volatile tt_l1_ptr uint32_t* event = to.event(index);
            event[1] = request;
            event[2] = status;
            event[3] = ticks;
            asm volatile("" ::: "memory");
            event[0] = index;
            if (
                thread && to.remote != 0
            ) {  // the event landed there before the count telling it
                to.mirror(KMD_WORK_EVENTS + (index & (to.event_depth - 1u)) * 16, 16);
                noc_async_write_barrier();
            }
            *to.told() = index;
            if (thread) {
                to.mirror(KMD_WORK_TOLD, 4);
            }
        }

        // A TIMEOUT's auxiliary: the slot, or for a run across islands the island its
        // waves wait for (KM_EVENT_ISLAND + it), as the X280's.
        uint32_t timeout_of(uint32_t slot) {
            volatile tt_l1_ptr uint32_t* words = km::l1(link + kTlTold);
            const uint32_t waited = (
                link != 0 && run_on && slot == run_slot && words[kTlSpans] == run
                    ? words[kTlWaited]
                    : 0
            );
            return waited != 0 ? KM_EVENT_ISLAND + waited - 1u : slot;
        }

        // The command in `slot` is late (or its dynamic run gave up): its TIMEOUT
        // event, its words in the result, and out told; in sweeps no more.
        void late_slot(uint32_t slot) {
            volatile tt_l1_ptr Record* record = hand.record(slot);
            result[KMD_R_SLOT] = slot;
            result[KMD_R_SEQUENCE] = record->sequence;
            result[KMD_R_REQ] = record->request;
            push_to(  // a link record's to the link: the host has no such command
                record->flags & kRecordLink ? kSourceLink : kSourceHost,
                record->request,
                KM_EVENT_TIMEOUT | timeout_of(slot) << 8 |
                    (record->sequence & KM_EVENT_SEQUENCE_MASK) << 16,
                0
            );
            hand.word(kTwLate) = slot + 1;
            late = true;
        }

        // A pass during a dynamic run: its RUN done (nothing to end), its lines quiet
        // (out told to send its end), or a line that gave up (the RUN late now).
        void run_turn(uint64_t now) {
            const uint32_t sequence = hand.word(kTwRunSequence);
            if (sequence != 0 && sequence != run) {
                run = sequence;
                run_slot = hand.word(kTwRunSlot);
                run_group = hand.word(kTwRunGroup);
                run_on = true;
            }
            if (!run_on || late) {
                return;
            }
            if (slots.seen[run_slot] == run) {
                run_on = false;  // the RUN completed or ended otherwise
                return;
            }
            uint32_t gave = 0;
            uint64_t sent = 0, consumed = 0;
            const int found = sample_lines(block, run, run_group, gave, sent, consumed);
            checks++;
            // a run across islands: its samples to out, whose waves end it
            // (td_link.hpp)
            const bool spans = link != 0 && km::l1(link + kTlTold)[kTlSpans] == run;
            if (found < 0) {
                hand.word(kTwRunGave) = gave + 1;
                run_on = false;
                late_slot(run_slot);
            } else if (spans) {
                volatile tt_l1_ptr uint32_t* words = km::l1(link + kTlTold);
                words[kTlQuiet] = found > 0 ? 1u : 0u;
                words[kTlSent] = static_cast<uint32_t>(sent);
                words[kTlSent + 1] = static_cast<uint32_t>(sent >> 32);
                words[kTlConsumed] = static_cast<uint32_t>(consumed);
                words[kTlConsumed + 1] = static_cast<uint32_t>(consumed >> 32);
                asm volatile("" ::: "memory");
                words[kTlSamples] = words[kTlSamples] + 1;
            } else if (found > 0 && sent == consumed) {
                hand.word(kTwRunQuiet) = run;
                quiet_at = static_cast<uint32_t>(now);
                run_on = false;
            }
        }

        // Slot `slot` past its deadline: given up by out when it may be (one at a time,
        // the others wait), else late (true: the run ends).
        bool late_at(uint32_t slot, uint32_t flags) {
            const uint32_t sequence = hand.record(slot)->sequence;
            if ((flags & kRecordGiveUp) == 0) {
                late_slot(slot);
                return true;
            }
            if (giving == 0) {
                giving = sequence;
                giving_slot = slot;
                hand.word(kTwGiveSlot) = slot;
                asm volatile("" ::: "memory");
                hand.word(kTwGiveSequence) = sequence;
            }
            return false;
        }

        // Out's answer to the give-up: every core left (TIMEOUT, the slot freed, the
        // run goes on) or one did not (late). An ack that came first ends it as done.
        void give_turn(uint64_t now) {
            const uint32_t slot = giving_slot;
            if (slots.seen[slot] == giving) {
                giving = 0;
                return;
            }
            if (hand.word(kTwGiveKept) == giving) {
                giving = 0;
                late_slot(slot);
                return;
            }
            if (hand.word(kTwGivenUp) != giving) {
                return;
            }
            volatile tt_l1_ptr Record* record = hand.record(slot);
            const uint64_t at = wide(record->submitted_lo, record->submitted_hi);
            if (
                (record->flags & kRecordLink) == 0
            ) {  // before the event the host reads
                hand.word(kTwGiveUps) = ++given_ups;
            }
            const uint32_t to = record_source(record->flags);
            const uint32_t time = static_cast<uint32_t>(now > at ? now - at : 0);
            push_to(
                to,
                record->request,
                KM_EVENT_TIMEOUT | timeout_of(slot) << 8 |
                    (giving & KM_EVENT_SEQUENCE_MASK) << 16,
                is_thread(to) ? thread_ticks(time, scale) : time
            );
            slots.seen[slot] = giving;
            hand.done()[slot] = giving;
            last_event = now;
            giving = 0;
        }

        // The watch's stalls as events: SEEN (req the next host command, ticks the gap)
        // and BACK (req the loops, 1 out | 2 in, ticks since SEEN), one each a pass.
        void stall_turn() {
            const uint32_t seen = watch[kTkSeen], back = watch[kTkBack];
            const uint32_t loops = watch[kTkLoops], ticks = watch[kTkTicks];
            if (seen != stalls_seen) {
                stalls_seen = seen;
                push(
                    hand.word(kTwConsumed) + 1,
                    KMD_EVENT_STALL | KMD_STALL_SEEN << 8,
                    ticks
                );
            }
            if (back != stalls_back && back == seen) {
                stalls_back = back;
                push(loops, KMD_EVENT_STALL | KMD_STALL_BACK << 8, ticks);
            }
        }

        // One pass: the sweep, the events out posted, the host words when they changed.
        // Whether it did anything.
        bool pass(uint64_t now) {
            bool active = false;
            passes++;
            watch[kTkIn] = passes;  // the watch's beat
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
                slots.sweep(
                    now,
                    [&](
                        uint32_t request,
                        uint32_t status,
                        uint32_t ticks,
                        uint32_t flags
                    ) {
                        if ((flags & kRecordSilent) == 0) {
                            const uint32_t to = record_source(flags);
                            push_to(
                                to,
                                request,
                                status,
                                is_thread(to) && (flags & kRecordResult) == 0
                                    ? thread_ticks(ticks, scale)
                                    : ticks
                            );  // fmt: skip
                        }
                        if (
                            (
                                flags &
                                (
                                    kRecordResult | kRecordThread | kRecordEngine |
                                    kRecordLink
                                )
                            ) == kRecordResult
                        ) {
                            results++;
                            last_result = ticks;
                        }
                        last_event = now;
                        active = true;
                    },
                    [&](uint32_t slot, uint32_t flags) {
                        return late_at(slot, flags);
                    }
                );
            }
            if (giving != 0) {
                give_turn(now);
            }
            if (run_on || hand.word(kTwRunSequence) != run) {
                run_turn(now);
            }
            if (watch[kTkSeen] != stalls_seen || watch[kTkBack] != stalls_back) {
                stall_turn();
            }
            const uint32_t posted = hand.word(kTwPosted);
            while (taken != posted) {
                volatile tt_l1_ptr uint32_t* event = hand.event(taken);
                if (
                    event[3] == kSourceHost && (event[1] & 0xFFu) == KM_EVENT_TIMEOUT
                ) {  // a crossing's give-up (td_link.hpp):
                                             // counted first, as a slot's
                    hand.word(kTwGiveUps) = ++given_ups;
                }
                push_to(event[3], event[0], event[1], event[2]);
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
            hand.word(kTwRunChecks) = checks;
            hand.word(kTwRunQuietAt) = quiet_at;
            asm volatile("" ::: "memory");
            hand.word(kTwOver) = arg(kTdTag);
        }
    };

}  // namespace td
