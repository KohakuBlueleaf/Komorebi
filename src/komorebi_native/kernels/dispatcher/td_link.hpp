// Commands across islands under the Tensix dispatcher, as the X280's (islands/link.h):
// a command whose group reaches other islands runs its part here and sends each other
// island its line as a packet, staged on link core 0 and carried by a PAGES LINE
// command into that island's inbox; that island's dispatcher runs it as source 5 and
// sends DONE back. One event to the command's source once every part is back, or a
// TIMEOUT naming the lowest island not back at its deadline.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_sources.hpp"

namespace td {

    constexpr uint32_t kLinkCrossings = 8;  // crossings in flight (KMD_CROSSINGS)
    constexpr uint32_t kLinkOut = 16;       // packets waiting for link core 0
    constexpr uint32_t kLinkOrigins = 16;   // commands from other islands not answered
    constexpr uint32_t kIslandNone = 0xFFFFFFFFu;
    // A link record's request (kRecordLink): a crossing's local part, a LINE's stage,
    // else the link ring's number of a command from another island.
    constexpr uint32_t kRequestCrossing = 0x80000000u;
    constexpr uint32_t kRequestStage = 0x40000000u;

    // A packet: the command line (16 words, its landing words its number), then the
    // crossing line (KMD_X_*).
    struct Packet {
        uint32_t word[32];
    };

    // A command of this island's sources with a part on another island.
    struct Crossing {
        uint32_t request, source, parts, generation, sequence, first, status;
        uint32_t word;    // the result word, from the part of the group's first rank
        uint32_t result;  // KM_FLAG_RESULT: 1, 2 once that part gave the word
        uint32_t late;    // its TIMEOUT went out; the rest of its parts send none
        uint32_t next, last, holder;
        uint32_t pending;  // bit i: island i's part not back
        uint32_t ended;    // every part back: its event waits for room to post
        uint32_t ticks;    // that event's ticks
        uint64_t submitted, deadline;
        uint32_t line[16];
    };

    // A command from another island in the link ring: whose it is (number 0: free).
    struct Origin {
        uint32_t number, island, ticket;
    };

    // Out's link state, in the link block (kTlState); out alone reads and writes it.
    struct Link {
        uint32_t island, islands, core, area, slots, part_bytes, base;
        uint32_t entries[3 * kTdLinkIslandsMax];  // ranks, threads, dispatcher tile
        uint32_t sent[kTdLinkIslandsMax];         // packets sent to island i
        uint32_t read[kTdLinkIslandsMax];         // packets read from island i
        uint32_t busy;                            // bit e: crossing e in flight
        Crossing crossing[kLinkCrossings];
        Packet out[kLinkOut];
        uint8_t out_to[kLinkOut];
        uint32_t out_head, out_tail;
        uint32_t stages;  // bit s: stage s holds a packet in flight
        uint32_t stage_ticket[KM_LINK_STAGES];  // 1 + its crossing's ticket, 0: a DONE
        Origin origin[kLinkOrigins];
        uint32_t stored;          // lines put into the link ring
        uint32_t events_read;     // the link ring's events read
        uint32_t left[kTlDepth];  // each line's ticks left of its deadline
        uint32_t crossed, received, answered, gave_up;

        // From the arguments; false: no link (one island, or no link line).
        bool open(uint32_t block) {
            const uint32_t word = get_arg_val<uint32_t>(kTdLinkIsland);
            if (word == 0) {
                return false;
            }
            island = word & 0xFFFFu;
            islands = word >> 16;
            core = get_arg_val<uint32_t>(kTdLinkCore);
            area = get_arg_val<uint32_t>(kTdLinkArea);
            base = block;
            slots = get_arg_val<uint32_t>(kTdLinkSlots);
            part_bytes = KMD_LINK_COUNT_BYTES + slots * KM_LINK_PACKET_BYTES;
            for (uint32_t i = 0; i < 3 * kTdLinkIslandsMax; i++) {
                entries[i] = get_arg_val<uint32_t>(kTdLinkEntries + i);
            }
            for (uint32_t i = 0; i < kTdLinkIslandsMax; i++) {
                sent[i] = 0;
                read[i] = 0;
            }
            busy = 0;
            for (uint32_t e = 0; e < kLinkCrossings; e++) {
                crossing[e].generation = 0;
            }
            out_head = out_tail = 0;
            stages = 0;
            for (uint32_t k = 0; k < kLinkOrigins; k++) {
                origin[k].number = 0;
            }
            stored = events_read = 0;
            crossed = received = answered = gave_up = 0;
            return true;
        }

        // Island `from`'s part of island `to`'s inbox: its count line's offset.
        uint32_t part_of(uint32_t to, uint32_t from) const {
            return (from < to ? from : from - 1u) * part_bytes;
        }

        // The island whose ranks hold `rank`, or kIslandNone.
        uint32_t find(uint32_t rank) const {
            for (uint32_t i = 0; i < islands; i++) {
                const uint32_t range = entries[3 * i];
                if (rank - (range & 0xFFFFu) < (range >> 16)) {
                    return i;
                }
            }
            return kIslandNone;
        }

        // Whether a group (flat ranks) reaches past this island's; a link group never.
        bool reaches_out(uint32_t group) const {
            const uint32_t first = group & 0xFFFFu, count = group >> 16;
            if (KM_GROUP_IS_LINK(first) || first >= KM_GROUP_CLUSTER_BASE) {
                return false;
            }
            const uint32_t range = entries[3 * island];
            return (
                first < (range & 0xFFFFu) ||
                first + count > (range & 0xFFFFu) + (range >> 16)
            );
        }

        uint32_t out_room() const {
            return kLinkOut - (out_tail - out_head);
        }

        bool active() const {
            return busy != 0 || out_head != out_tail || stages != 0;
        }

        // Queues a packet for island `to`: `command` (16 words, or none for a DONE) and
        // the crossing line's words.
        void queue(
            uint32_t to,
            const uint32_t* command,
            uint32_t kind,
            uint32_t ticket,
            uint32_t left_word,
            uint32_t ticks
        ) {  // fmt: skip
            Packet& packet = out[out_tail % kLinkOut];
            for (uint32_t w = 0; w < 32; w++) {
                packet.word[w] = w < 16 && command != nullptr ? command[w] : 0;
            }
            packet.word[16 + KMD_X_KIND] = kind | island << 8;
            packet.word[16 + KMD_X_TICKET] = ticket;
            packet.word[16 + KMD_X_LEFT] = left_word;
            packet.word[16 + KMD_X_TICKS] = ticks;
            out_to[out_tail % kLinkOut] = static_cast<uint8_t>(to);
            out_tail++;
        }

        // Crossing `index`'s part from island `from` (or the local part) back: its
        // status and ticks; the last part ends it (its event then waits to be posted).
        void part(uint32_t index, uint32_t from, uint32_t status, uint32_t ticks) {
            Crossing& c = crossing[index];
            const uint32_t kind = status & 0xFFu;
            if (from != island) {
                c.pending &= ~(1u << from);
            }
            if (kind != KM_EVENT_DONE && c.status == KM_EVENT_DONE) {
                c.status = (
                    kind == KM_EVENT_TIMEOUT
                        ? KM_EVENT_TIMEOUT | (KM_EVENT_ISLAND + from) << 8
                        : status
                );
            }
            if (kind == KM_EVENT_DONE && c.result == 1u && from == c.holder) {
                c.word = ticks;
                c.result = 2u;
            }
            if (--c.parts != 0) {
                return;
            }
            if (c.late) {
                busy &= ~(1u << index);
                return;
            }
            c.ended = 1;
        }

        // The event of ended crossing `index`: status and ticks, given `now`.
        uint32_t event_status(uint32_t index, uint64_t now, uint32_t& ticks) const {
            const Crossing& c = crossing[index];
            const uint32_t sequence = (
                (c.sequence & KM_EVENT_SEQUENCE_MASK) << KM_EVENT_SEQUENCE_SHIFT
            );
            ticks = static_cast<uint32_t>(now - c.submitted);
            if ((c.status & 0xFFu) == KM_EVENT_DONE) {
                ticks = c.result != 0 ? c.word : ticks;
                return KM_EVENT_DONE | ((c.first & 0xFFu) | sequence) << 8;
            }
            if ((c.status & 0xFFu) == KM_EVENT_TIMEOUT) {
                return (c.status & 0xFFFFu) | sequence << 8;
            }
            return c.status;
        }

        // Parts not yet queued while the out queue has room, each with the ticks left
        // of its deadline; a crossing given up queues none of the rest (they count as
        // back).
        void feed(uint64_t now) {
            for (uint32_t index = 0; index < kLinkCrossings; index++) {
                Crossing& c = crossing[index];
                if (!(busy & (1u << index)) || c.ended) {
                    continue;
                }
                const uint32_t ticket = index | c.generation << 8;
                while (c.next <= c.last) {
                    const uint32_t to = c.next;
                    if (to != island && c.late) {
                        c.next++;
                        part(index, to, KM_EVENT_DONE, 0);
                        if (!(busy & (1u << index))) {
                            break;
                        }
                        continue;
                    }
                    if (to != island) {
                        if (out_room() == 0) {
                            return;
                        }
                        const uint64_t left_ticks = (
                            c.deadline > now ? c.deadline - now : 1u
                        );
                        queue(
                            to,
                            c.line,
                            KMD_CROSS_COMMAND,
                            ticket,
                            left_ticks > 0xFFFFFFFFull
                                ? 0xFFFFFFFFu
                                : static_cast<uint32_t>(left_ticks),
                            0
                        );  // fmt: skip
                    }
                    c.next++;
                }
            }
        }

        // Reads island `from`'s inbox part: commands into the link ring (whose, time
        // left), DONEs to their crossings; a packet not yet whole, or no room, waits.
        // `taken`: the link ring's next number to take.
        void receive_from(uint32_t from, uint32_t taken) {
            const uint32_t at = base + kTlInbox + part_of(island, from);
            const uint32_t landed = km::l1(at)[0];
            while (read[from] != landed) {
                const uint32_t number = read[from] + 1u;
                volatile tt_l1_ptr uint32_t* words = (
                    km::l1(
                        at + KMD_LINK_COUNT_BYTES +
                        (number % slots) * KM_LINK_PACKET_BYTES
                    )
                );
                const uint32_t* cross = nullptr;
                uint32_t packet[32];
                for (uint32_t w = 0; w < 32; w++) {
                    packet[w] = words[w];
                }
                cross = packet + 16;
                if (
                    packet[0] != number || packet[4] != number || packet[8] != number ||
                    packet[12] != number || cross[KMD_X_NUMBER] != number ||
                    cross[KMD_X_ROW] != number
                ) {
                    return;  // not whole yet
                }
                const uint32_t kind = cross[KMD_X_KIND] & 0xFFu;
                const uint32_t sender = (cross[KMD_X_KIND] >> 8) & 0xFFu;
                if (kind == KMD_CROSS_COMMAND) {
                    uint32_t free_origin = kLinkOrigins;
                    for (uint32_t k = 0; k < kLinkOrigins; k++) {
                        if (origin[k].number == 0) {
                            free_origin = k;
                            break;
                        }
                    }
                    if (
                        stored + 1u - taken >= kTlDepth || free_origin == kLinkOrigins
                    ) {
                        return;  // waits for room
                    }
                    const uint32_t line_number = ++stored;
                    volatile tt_l1_ptr uint32_t* line = (
                        km::l1(base + kTlRing + (line_number & (kTlDepth - 1u)) * 64)
                    );
                    for (uint32_t w = 0; w < 16; w++) {
                        line[w] = w % 4 == 0 ? line_number : packet[w];
                    }
                    left[line_number % kTlDepth] = cross[KMD_X_LEFT];
                    origin[free_origin].number = line_number;
                    origin[free_origin].island = sender;
                    origin[free_origin].ticket = cross[KMD_X_TICKET];
                    received++;
                } else if (kind == KMD_CROSS_DONE) {
                    const uint32_t ticket = cross[KMD_X_TICKET];
                    const uint32_t index = ticket & 0xFFu;
                    if (
                        index < kLinkCrossings && (busy & (1u << index)) &&
                        !crossing[index].ended &&
                        crossing[index].generation == ticket >> 8
                    ) {
                        part(index, sender, cross[KMD_X_LEFT], cross[KMD_X_TICKS]);
                    }
                }  // a dynamic run's waves: not served here
                read[from] = number;
            }
        }
    };

    static_assert(kTlState + sizeof(Link) <= kTlInbox, "the link's state");

}  // namespace td
