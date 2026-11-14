// The link service: a core of the run's group that no handler addresses joins its
// island to another: its writer sends its outbox over the fabric, its reader forwards.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_messages.hpp"

#if !defined(COMPILE_FOR_TRISC)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#endif

namespace km {

    // Reader and writer program arguments after MessagesArgs (the program's own
    // follow).
    struct LinkArgs {
        uint32_t core;        // the link core in the core table; kLinkNone: no link
        uint32_t region;      // L1 address of the link's words, outbox and inbox
        uint32_t outbox;      // outbox entries: a power of two, kLinkOutboxMin or more
        uint32_t inbox;       // inbox packets, 1 to kLinkInboxMax
        uint32_t packet;      // entries a packet at most (1: one message a packet)
        uint32_t island;      // this island's number
        uint32_t far;         // the far link core's NoC coordinates, x | y << 16
        uint32_t far_region;  // its region's L1 address
        uint32_t channel;   // the fabric's Ethernet channel toward it (the connection's
        uint32_t teardown;  // arguments, in this order: channel, the L1 addresses of
        uint32_t cursor;    // the teardown answer and of the producer cursor)
        uint32_t token;     // this load's hello, the same on both sides (never 0)
    };

    constexpr uint32_t kLinkArgs = sizeof(LinkArgs) / 4;
    constexpr uint32_t kLinkArgChannel = 8;  // LinkArgs::channel's word
    constexpr uint32_t kLinkNone = 0xFFFFFFFF;
    constexpr uint32_t kLinkOutboxMin = kMessagesPacket * 2;
    constexpr uint32_t kLinkInboxMax = 16;
    constexpr uint32_t kLinkPacketBytes = 4352;  // the fabric's largest payload

    // The region, from its address: 16-byte words, each with one writer.
    constexpr uint32_t kLinkOutTail = 0x00;  // outbox positions reserved (atomic adds)
    constexpr uint32_t kLinkOutHead = 0x10;  // outbox positions sent (writer)
    constexpr uint32_t kLinkLanded = 0x20;   // inbox packets landed (the far writer)
    constexpr uint32_t kLinkCredit = 0x30;   // far inbox packets freed (the far writer)
    constexpr uint32_t kLinkReader = 0x40;  // packets taken, messages issued, run, issued in it
    constexpr uint32_t kLinkWriter = 0x50;  // packets sent, freed count sent, run, sent in it
    constexpr uint32_t kLinkTeardown = 0x60;  // the connection's teardown answer
    constexpr uint32_t kLinkCursor = 0x70;    // the connection's producer cursor
    constexpr uint32_t kLinkWaits = 0x80;   // reader: -; writer: polls without credit,
                                            // router space, an entry; credit packets
    constexpr uint32_t kLinkGave = 0x90;    // reader's first give-up (where, position,
                                            // run, word); the writer's at 0xA0
    constexpr uint32_t kLinkHello = 0xB0;   // the far writer's hello: its token
    constexpr uint32_t kLinkStamps = 0xC0;  // wall clock (low word) at: the first send,
                                            // the first data, the first count, link up
    constexpr uint32_t kLinkHeaders = 0xD0;  // the writer's packet headers: data,
                                             // credit, count
    constexpr uint32_t kLinkHeaderBytes = 64;
    constexpr uint32_t kLinkOutbox = kLinkHeaders + 3 * kLinkHeaderBytes;
    constexpr uint32_t kLinkPrefix = 16;  // an entry: {sequence, count} of its packet
                                          // (first entry), then the message

    // Words of kLinkStamps (0 until it happens).
    constexpr uint32_t kStampSent = 0;
    constexpr uint32_t kStampData = 1;
    constexpr uint32_t kStampCount = 2;
    constexpr uint32_t kStampUp = 3;

    // Words of kLinkReader and kLinkWriter.
    constexpr uint32_t kLinkTaken = 0;   // reader: inbox packets taken and freed
    constexpr uint32_t kLinkIssued = 1;  // reader: messages issued again, all runs
    constexpr uint32_t kLinkSent = 0;    // writer: packets sent, all runs
    constexpr uint32_t kLinkFreed = 1;               // writer: the reader's taken count last sent back
    constexpr uint32_t kLinkRun = 2;  // the run the next word counts
    constexpr uint32_t kLinkRunCount = 3;  // messages issued (reader) or sent (writer) in it

    // Where codes of a link (KM_MESSAGES_GAVE and kLinkGave word 0).
    constexpr uint32_t kWhereLinkArgs = 50;  // its arguments, or the router's packet size
    constexpr uint32_t kWhereLinkSpace = 51;   // a writer: no outbox space in time
    constexpr uint32_t kWhereLinkCredit = 52;  // link writer: no far inbox packet free
    constexpr uint32_t kWhereLinkRouter = 53;  // link writer: no space at the router
    constexpr uint32_t kWhereLinkSent = 54;  // link writer: a packet's writes did not leave
    constexpr uint32_t kWhereLinkGap = 55;  // link writer: a reserved entry not written
    constexpr uint32_t kWhereLinkSequence = 56;  // link reader: a packet's sequence, count
    constexpr uint32_t kWhereLinkIsland = 57;  // link reader: an entry not for its island
    constexpr uint32_t kWhereLinkOverflow = 58;              // link reader: more landed than it holds
    constexpr uint32_t kWhereLinkClose = 59;  // link writer: the connection's close
    constexpr uint32_t kWhereLinkUp = 60;     // link writer: no hello from the far side
    static_assert(kWhereLinkArgs > kWhereClear, "link where codes after the runtime's");

    // A message's route word (word 1) while destinations are explicit words: the
    // island above, the rank of the run's group there below.
    constexpr uint32_t kLinkRouteWord = 1;
    constexpr uint32_t kLinkIslandShift = 16;
    constexpr uint32_t kLinkRankMask = 0xFFFF;

    // Bytes of an outbox entry, and the entries a packet holds, for `message_bytes`.
    inline uint32_t link_entry(uint32_t message_bytes) {
        return kLinkPrefix + message_bytes;
    }

    inline uint32_t link_packet_max(uint32_t message_bytes) {
        return kLinkPacketBytes / link_entry(message_bytes);
    }

    // Region bytes: the words, the outbox, the inbox and the writer's packet.
    inline uint32_t link_bytes(
        uint32_t message_bytes, uint32_t outbox, uint32_t inbox
    ) {
        return (
            kLinkOutbox + outbox * link_entry(message_bytes) +
            (inbox + 1) * kLinkPacketBytes
        );
    }

    // What the run that ends at a quiet link expects to cross each way (0, 0: none
    // given, the link is quiet whenever it holds nothing).
    struct LinkEnd {
        uint32_t out;
        uint32_t in;
    };

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    template <typename CoreType>
    inline LinkArgs link_args(const CoreType& core) {
        LinkArgs link_arguments;
        uint32_t* words = &link_arguments.core;
        for (uint32_t index = 0; index < kLinkArgs; index++) {
            words[index] = core.arg(kMessagesArgs + index);
        }
        return link_arguments;
    }

    // Whether `link` is a usable link of messages of `message_bytes` (the shapes the
    // reader, the writer and the island's writers rely on).
    inline bool link_valid(const LinkArgs& link, uint32_t message_bytes) {
        return (
            link.region % 16 == 0 && link.far_region % 16 == 0 &&
            link.outbox >= kLinkOutboxMin && (link.outbox & (link.outbox - 1)) == 0 &&
            link.inbox >= 1 && link.inbox <= kLinkInboxMax && link.packet >= 1 &&
            link.packet <= link_packet_max(message_bytes) && message_bytes % 16 == 0 &&
            link.token != 0
        );
    }

    // The link core's start-up, before the first command: its words, outbox marks and
    // inbox prefixes zero (nothing crosses before both islands are loaded).
    inline void link_zero(const LinkArgs& link, uint32_t message_bytes) {
        const uint32_t bytes = link_bytes(message_bytes, link.outbox, link.inbox);
        for (uint32_t word = 0; word < bytes / 4; word++) {
            l1(link.region)[word] = 0;
        }
        invalidate_l1_cache();
    }

    // A writer's bin for the link core goes to its outbox as to a queue: one atomic add
    // reserves, a wait for space at the link's head, the messages, then the first mark.
    struct LinkOut {
        LinkArgs link;
        uint32_t message_bytes;
        uint32_t entry;
        uint32_t shift;
        uint32_t head = 0;  // the link writer's head as last read

        template <typename WriterType>
        explicit LinkOut(const WriterType& writer) :
            link(link_args(writer)),
            message_bytes(messages_args(writer).message_bytes),
            entry(link_entry(message_bytes)),
            shift(__builtin_ctz(link.outbox == 0 ? 1 : link.outbox)) {
        }

        bool takes(uint32_t destination) const {
            return link.core != kLinkNone && destination == link.core;
        }

        template <typename WriterType>
        uint32_t put(
            WriterType& writer,
            const km_command_t&,
            uint32_t,
            uint32_t base,
            uint32_t stride,
            uint32_t count
        ) {
            if (!link_valid(link, message_bytes) || count > link.outbox) {
                return kWhereLinkArgs;
            }
            const Args& arguments = writer.args();
            const uint32_t returned_address = (
                arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_RETURNED
            );
            volatile tt_l1_ptr uint32_t* returned = l1(returned_address);
            returned[0] = kMessagesEnd;
            invalidate_l1_cache();  // a fence: the store lands before the answer can
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index,
                write_at_cmd_buf,
                writer.core_noc(link.core, link.region + kLinkOutTail),
                NOC_UNICAST_WRITE_VC,
                count,
                31,
                false,
                false,
                returned_address
            );
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            })) {
                return kWhereMessagesAtomic;
            }
            if (!noc_timeout(writer, [&] {
                invalidate_l1_cache();
                return returned[0] != kMessagesEnd;
            })) {
                return kWhereMessagesLost;
            }
            const uint32_t first_position = returned[0];
            // Space: the positions are less than an outbox ahead of the link's head.
            const uint32_t scratch = (
                arguments.control + KM_CONTROL_TAILS + KM_TAILS_WRITER
            );
            if (first_position + count - head > link.outbox) {
                if (!noc_timeout(writer, [&] {
                    noc_async_read(
                        writer.core_noc(link.core, link.region + kLinkOutHead),
                        scratch,
                        16
                    );
                    for (
                        uint32_t polls = 0; !ncrisc_noc_reads_flushed(noc_index);
                        polls++
                    ) {
                        if (polls > (1u << 20)) {
                            return false;
                        }
                    }
                    invalidate_l1_cache();
                    head = l1(scratch)[0];
                    return first_position + count - head <= link.outbox;
                })) {
                    return kWhereLinkSpace;
                }
            }
            const MessageMarks marks(first_position, shift, count);
            *message_mark_half(base) = 0;
            for (uint32_t i = 1; i < count; i++) {
                *message_mark_half(base + i * stride) = marks(i);
            }
            invalidate_l1_cache();  // a fence: the NoC reads the stored marks
            auto at = [&](uint32_t i) {
                return (
                    writer.core_noc(
                        link.core,
                        link.region + kLinkOutbox +
                            ((first_position + i) & (link.outbox - 1)) * entry + kLinkPrefix
                    )
                );
            };
            for (uint32_t i = 0; i < count; i++) {
                noc_async_write(base + i * stride, at(i), message_bytes);
            }
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                return kWhereMessagesFlush;
            }
            *message_mark_half(base) = marks(0);
            invalidate_l1_cache();
            noc_async_write(base, at(0), 16);
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                return kWhereMessagesFlush;
            }
            return 0;
        }
    };

    // `count` messages `stride` bytes apart from `base` to local core `destination`:
    // reserve on its counter tile, write with the first mark 0, set it. 0 or where.
    template <typename CoreType>
    inline uint32_t link_put(
        const CoreType& core,
        const km_command_t& command,
        const MessagesArgs& messages_arguments,
        uint32_t destination,
        uint32_t base,
        uint32_t stride,
        uint32_t count,
        uint32_t returned_address
    ) {
        const Args& arguments = core.args();
        const uint32_t message_bytes = messages_arguments.message_bytes;
        const uint32_t shift = __builtin_ctz(messages_arguments.cap);
        const uint32_t island_first = islands_at(arguments.control).first();
        const InterleavedAddrGen<true> queue_addresses = {
            .bank_base_address = messages_arguments.queue,
            .page_size = kMessagesPageMessages * message_bytes
        };
        auto at = [&](uint32_t position) {
            const uint32_t in_queue = position & (messages_arguments.cap - 1);
            return (
                queue_addresses.get_noc_addr(
                    messages_page(destination, position, messages_arguments.cap),
                    in_queue % kMessagesPageMessages * message_bytes
                )
            );
        };
        volatile tt_l1_ptr uint32_t* returned = l1(returned_address);
        returned[0] = kMessagesEnd;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(
                messages_counter(
                    command,
                    messages_arguments.counters,
                    island_first + destination,
                    island_first
                ),
                arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE + 16 * destination
            ),
            NOC_UNICAST_WRITE_VC,
            count,
            31,
            false,
            false,
            returned_address
        );
        if (!noc_timeout(core, [] {
            return ncrisc_noc_nonposted_atomics_flushed(noc_index);
        })) {
            return kWhereMessagesAtomic;
        }
        // On the card the answer can stay away although its count moved.
        if (!noc_timeout(core, [&] {
            invalidate_l1_cache();
            return returned[0] != kMessagesEnd;
        })) {
            return kWhereMessagesLost;
        }
        const uint32_t first_position = returned[0];
        const MessageMarks marks(first_position, shift, count);
        *message_mark_half(base) = 0;
        for (uint32_t i = 1; i < count; i++) {
            *message_mark_half(base + i * stride) = marks(i);
        }
        invalidate_l1_cache();  // a fence: the NoC reads the stored marks
        for (uint32_t i = 0; i < count; i++) {
            noc_async_write(base + i * stride, at(first_position + i), message_bytes);
        }
        if (!noc_timeout(core, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        })) {
            return kWhereMessagesFlush;
        }
        *message_mark_half(base) = marks(0);
        invalidate_l1_cache();
        noc_async_write(base, at(first_position), 16);
        if (!noc_timeout(core, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        })) {
            return kWhereMessagesFlush;
        }
        return 0;
    }

    // The first give-up of one side of a link: its words in the region and the run's.
    inline void link_gave_up(
        uint32_t control,
        const LinkArgs& link,
        uint32_t side,
        uint32_t where,
        uint32_t position,
        uint32_t run,
        uint32_t seen
    ) {
        volatile tt_l1_ptr uint32_t* gave = l1(link.region + kLinkGave + 16 * side);
        if (gave[0] == 0) {
            gave[1] = position;
            gave[2] = run;
            gave[3] = seen;
            gave[0] = where;
        }
        messages_gave_up(control, where, position, run, seen);
    }

    // Link reader: each landed packet goes on to the ranks its route words name. Line:
    // issued as sent, sent across as consumed; idle once `end` is met.
    template <typename ReaderType>
    inline void link_receiver(
        ReaderType& reader, const km_command_t& command, const LinkEnd& end
    ) {
        const Args& arguments = reader.args();
        const MessagesArgs messages_arguments = messages_args(reader);
        const LinkArgs link = link_args(reader);
        const uint32_t run = command.sequence;
        const uint32_t entry = link_entry(messages_arguments.message_bytes);
        const uint32_t inbox = link.region + kLinkOutbox + link.outbox * entry;
        const uint32_t returned_address = (
            arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_RETURNED
        );
        volatile tt_l1_ptr uint32_t* messages = reader.control(KM_CONTROL_MESSAGES);
        volatile tt_l1_ptr uint32_t* out = messages + KM_MESSAGES_OUT / 4;
        volatile tt_l1_ptr uint32_t* landed = l1(link.region + kLinkLanded);
        volatile tt_l1_ptr uint32_t* mine = l1(link.region + kLinkReader);
        volatile tt_l1_ptr uint32_t* writer_words = l1(link.region + kLinkWriter);
        volatile tt_l1_ptr uint32_t* tail = l1(link.region + kLinkOutTail);
        volatile tt_l1_ptr uint32_t* head = l1(link.region + kLinkOutHead);
        volatile tt_l1_ptr uint32_t* stamps = l1(link.region + kLinkStamps);
        const uint64_t line = (
            reader.disp_noc(messages_arguments.lines + 16 * arguments.me)
        );
        const uint32_t counter_tiles = messages_arguments.counters;
        const uint32_t ranks = messages_ranks(command, counter_tiles);
        const uint32_t island_first = islands_at(arguments.control).first();
        const uint32_t me = island_first + arguments.me;  // this core's flat rank
        uint32_t where = 0, seen = 0, issued = 0;
        mine[kLinkRunCount] = 0;
        mine[kLinkRun] = run;
        if (
            !link_valid(link, messages_arguments.message_bytes) ||
            link.core != arguments.me || counter_tiles == 0 ||
            counter_tiles > KM_COUNTERS_MAX || ranks == 0
        ) {
            where = kWhereLinkArgs;
        }
        for (uint32_t tile = 0; tile < counter_tiles && where == 0; tile++) {
            if (!messages_ready(
                reader,
                first(command) - island_first + ranks + tile,
                KM_TAILS_READER,
                run
            )) {
                where = kWhereMessagesCounter;
            }
        }
        // One 16-byte store of {run, state, sent, consumed} at the submitter.
        auto status = [&](uint32_t state, uint32_t sent, uint32_t consumed) {
            out[0] = run;
            out[1] = state;
            out[2] = sent;
            out[3] = consumed;
            invalidate_l1_cache();  // a fence: the NoC reads the stored line
            noc_async_write(
                arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_OUT, line, 16
            );
            return noc_timeout(reader, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            });
        };
        // Messages this run sent across: the writer's count once it counts this run.
        auto sent_across = [&] {
            return writer_words[kLinkRun] == run ? writer_words[kLinkRunCount] : 0u;
        };
        bool idle_shown = false, armed = false;
        uint32_t key[3] = {0, 0, 0};
        uint64_t since = 0;
        while (where == 0) {
            invalidate_l1_cache();
            if (reader.stop()) {
                where = kWhereMessagesStopped;
                seen = *landed;
                break;
            }
            const uint32_t taken = mine[kLinkTaken];
            const uint32_t landed_count = *landed;
            const uint32_t slot_count = (
                l1(inbox + (taken % link.inbox) * kLinkPacketBytes)[1]
            );
            if (stamps[kStampData] == 0 && slot_count != 0) {
                stamps[kStampData] = wall() | 1;
            }
            if (stamps[kStampCount] == 0 && landed_count != taken) {
                stamps[kStampCount] = wall() | 1;
            }
            if (landed_count - taken > link.inbox) {
                where = kWhereLinkOverflow;
                seen = landed_count;
                break;
            }
            if (landed_count == taken) {
                if (messages_ended(reader, run, me)) {
                    break;
                }
                const uint32_t across = sent_across();
                const bool quiet = (
                    *tail == *head && writer_words[kLinkFreed] == taken &&
                    (
                        (end.out == 0 && end.in == 0) ||
                        (across == end.out && issued == end.in)
                    )
                );
                if (!quiet || idle_shown) {
                    armed = armed && quiet;
                    if (!quiet && idle_shown) {
                        if (!status(KM_MESSAGES_BUSY, 0, 0)) {
                            where = kWhereMessagesLine;
                        }
                        idle_shown = false;
                    }
                    continue;
                }
                if (!armed || key[0] != taken || key[1] != *tail || key[2] != across) {
                    armed = true;
                    key[0] = taken;
                    key[1] = *tail;
                    key[2] = across;
                    since = wall64();
                    continue;
                }
                if (wall64() - since < messages_arguments.idle) {
                    continue;
                }
                if (!status(KM_MESSAGES_IDLE, issued, across)) {
                    where = kWhereMessagesLine;
                    break;
                }
                idle_shown = true;
                continue;
            }
            armed = false;
            if (idle_shown) {  // busy has landed before the first message is issued
                if (!status(KM_MESSAGES_BUSY, 0, 0)) {
                    where = kWhereMessagesLine;
                    break;
                }
                idle_shown = false;
            }
            const uint32_t packet = inbox + (taken % link.inbox) * kLinkPacketBytes;
            const uint32_t sequence = l1(packet)[0];
            const uint32_t count = l1(packet)[1];
            if (sequence != taken || count == 0 || count > link.packet) {
                where = kWhereLinkSequence;
                seen = sequence;
                break;
            }
            for (uint32_t index = 0; index < count && where == 0;) {
                const uint32_t route = (
                    l1(packet + index * entry + kLinkPrefix)[kLinkRouteWord]
                );
                const uint32_t rank = route & kLinkRankMask;
                if (route >> kLinkIslandShift != link.island || rank >= ranks) {
                    where = kWhereLinkIsland;
                    seen = route;
                    break;
                }
                uint32_t row = 1;
                while (
                    index + row < count &&
                    l1(packet + (index + row) * entry + kLinkPrefix)[kLinkRouteWord] ==
                        route
                ) {
                    row++;
                }
                where = (
                    link_put(
                        reader,
                        command,
                        messages_arguments,
                        first(command) - island_first + rank,
                        packet + index * entry + kLinkPrefix,
                        entry,
                        row,
                        returned_address
                    )
                );
                seen = route;
                index += row;
            }
            if (where != 0) {
                break;
            }
            issued += count;
            mine[kLinkIssued] += count;
            mine[kLinkRunCount] = issued;
            mine[kLinkTaken] = taken + 1;  // the slot is free: the writer sends it back
        }
        if (where != 0) {
            link_gave_up(
                arguments.control, link, 0, where, mine[kLinkTaken], run, seen
            );
            status(KM_MESSAGES_GAVE_UP, mine[kLinkTaken], where);
            reader.gave_up(kStReader, kGaveStop, kWhereData, slot(command), run);
        }
        // Compute leaves its loop on the end packet (or on the stop flag).
        if (reader.poll(
            [] {
                return cb_pages_reservable_at_back(kCbMessagesIn, 1);
            },
            arguments.wait_limit,
            arguments.wait_limit
        ) == kGaveNone) {
            cb_reserve_back(kCbMessagesIn, 1);
            l1(get_write_ptr(kCbMessagesIn))[0] = kMessagesEnd;
            cb_push_back(kCbMessagesIn, 1);
        }
    }

    // The link core's fabric connection, held by its writer from start-up to its end:
    // opened once, a hello both ways before the outbox moves, closed at the end.
    struct LinkWire {
        LinkArgs link;
        uint32_t entry;
        uint32_t staging;  // the writer's packet: a payload is copied here first
        tt::tt_fabric::WorkerToFabricEdmSender sender;
        bool open = false;
        bool up = false;     // the far side's hello has been read
        uint32_t where = 0;  // why the connection could not be opened (0: it is)
        uint32_t seen = 0;

        template <typename WriterType>
        explicit LinkWire(WriterType& writer) :
            link(link_args(writer)),
            entry(link_entry(messages_args(writer).message_bytes)),
            staging(
                link.region + kLinkOutbox + link.outbox * entry +
                link.inbox * kLinkPacketBytes
            ) {
        }

        // Start-up: the region zeroed, the connection opened, one hello sent.
        template <typename WriterType>
        void start(WriterType& writer) {
            using namespace tt::tt_fabric;
            const uint32_t message_bytes = messages_args(writer).message_bytes;
            if (!link_valid(link, message_bytes) || link.core != writer.args().me) {
                where = kWhereLinkArgs;
                return;
            }
            link_zero(link, message_bytes);
            size_t index = kArgs + kMessagesArgs + kLinkArgChannel;
            sender = WorkerToFabricEdmSender::
                build_from_args<ProgrammableCoreType::TENSIX, true>(index);
            if (
                sender.buffer_size_bytes <
                    sizeof(PACKET_HEADER_TYPE) + link.packet * entry ||
                sizeof(PACKET_HEADER_TYPE) > kLinkHeaderBytes
            ) {
                where = kWhereLinkArgs;
                seen = sender.buffer_size_bytes;
                return;
            }
            sender.open();
            open = true;
            if (!hello(writer)) {
                where = kWhereLinkSent;
            }
        }

        // Header `which` zeroed, routed one chip on; the caller sets its NoC command.
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header(uint32_t which) {
            const uint32_t address = (
                link.region + kLinkHeaders + which * kLinkHeaderBytes
            );
            for (uint32_t word = 0; word < sizeof(PACKET_HEADER_TYPE) / 4; word++) {
                l1(address)[word] = 0;
            }
            volatile tt_l1_ptr PACKET_HEADER_TYPE*
                result = (
                    reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(address)
                );
            result->to_chip_unicast(1);
            return result;
        }

        // A NoC address `offset` bytes into the far region.
        uint64_t far(uint32_t offset) const {
            return (
                safe_get_noc_addr(
                    link.far & 0xFFFF,
                    link.far >> 16,
                    link.far_region + offset,
                    tt::tt_fabric::edm_to_local_chip_noc
                )
            );
        }

        // One packet: space, the payload (if any) from `source`, the header, then its
        // writes sent and flushed.
        template <typename WriterType>
        bool send(WriterType& writer, uint32_t which, uint32_t source, uint32_t bytes) {
            if (!noc_timeout(writer, [&] {
                return sender.edm_has_space_for_packet<1>();
            })) {
                return false;
            }
            if (bytes != 0) {
                sender.send_payload_without_header_non_blocking_from_address(
                    source, bytes
                );
            }
            sender.send_payload_flush_non_blocking_from_address(
                link.region + kLinkHeaders + which * kLinkHeaderBytes,
                sizeof(PACKET_HEADER_TYPE)
            );
            const bool sent = noc_timeout(writer, [] {
                return (
                    ncrisc_noc_nonposted_writes_sent(noc_index) &&
                    ncrisc_noc_nonposted_writes_sent(
                        tt::tt_fabric::get_fabric_worker_noc()
                    )
                );
            });
            return sent && noc_timeout(writer, [] {
                return (
                    ncrisc_noc_nonposted_writes_flushed(noc_index) &&
                    ncrisc_noc_nonposted_writes_flushed(
                        tt::tt_fabric::get_fabric_worker_noc()
                    )
                );
            });
        }

        // 16 bytes, `word` then zeros, to `offset` of the far region.
        template <typename WriterType>
        bool put_word(
            WriterType& writer, uint32_t which, uint32_t offset, uint32_t word
        ) {
            for (uint32_t index = 0; index < 4; index++) {
                l1(staging)[index] = index == 0 ? word : 0;
            }
            header(which)->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{far(offset)}, 16
            );
            invalidate_l1_cache();  // a fence: the NoC reads the stored words
            return send(writer, which, staging, 16);
        }

        template <typename WriterType>
        bool hello(WriterType& writer) {
            return put_word(writer, 1, kLinkHello, link.token);
        }

        // The end: the connection closed, bounded. 0 or where.
        template <typename WriterType>
        uint32_t close(WriterType& writer) {
            if (!open) {
                return 0;
            }
            open = false;
            sender.close_start();
            const bool closed = noc_timeout(
                writer,
                [] {
                    return (
                        ncrisc_noc_nonposted_writes_flushed(
                            tt::tt_fabric::get_fabric_worker_noc()
                        )
                    );
                }
            ) &&
                noc_timeout(writer, [&] {
                    invalidate_l1_cache();
                    return *sender.worker_teardown_addr == 1;
                });
            *sender.worker_teardown_addr = 0;
            return closed ? 0 : kWhereLinkClose;
        }
    };

    // Link writer: the outbox held, hellos repeated, until the far hello is read; then
    // freed counts back, entries in packets of `packet` (0: most). True: compute ended.
    template <typename WriterType>
    inline bool link_sender(
        WriterType& writer, const km_command_t& command, LinkWire& wire, uint32_t packet
    ) {
        using namespace tt::tt_fabric;
        const Args& arguments = writer.args();
        const LinkArgs& link = wire.link;
        const uint32_t run = command.sequence;
        const uint32_t entry = wire.entry;
        const uint32_t staging = wire.staging;
        const uint32_t shift = __builtin_ctz(link.outbox == 0 ? 1 : link.outbox);
        volatile tt_l1_ptr uint32_t* mine = l1(link.region + kLinkWriter);
        volatile tt_l1_ptr uint32_t* reader_words = l1(link.region + kLinkReader);
        volatile tt_l1_ptr uint32_t* tail = l1(link.region + kLinkOutTail);
        volatile tt_l1_ptr uint32_t* head_word = l1(link.region + kLinkOutHead);
        volatile tt_l1_ptr uint32_t* credit = l1(link.region + kLinkCredit);
        volatile tt_l1_ptr uint32_t* waits = l1(link.region + kLinkWaits);
        volatile tt_l1_ptr uint32_t* stamps = l1(link.region + kLinkStamps);
        volatile tt_l1_ptr uint32_t* far_hello = l1(link.region + kLinkHello);
        uint32_t where = wire.where, seen = wire.seen;
        packet = packet == 0 || packet > link.packet ? link.packet : packet;
        mine[kLinkRunCount] = 0;
        invalidate_l1_cache();  // a fence: the count is zero before the run names it
        mine[kLinkRun] = run;
        const uint64_t wait = static_cast<uint64_t>(arguments.wait_limit) << 10;
        const uint64_t entered = wall64();
        uint64_t hello_at = entered;
        uint64_t stalled_from = 0;
        uint32_t stalled_where = 0;
        uint32_t head = *head_word, sent = 0;
        bool ended = false;
        while (where == 0 && !ended) {
            invalidate_l1_cache();
            if (writer.stop()) {
                where = kWhereMessagesStopped;
                break;
            }
            if (cb_pages_available_at_front(kCbMessagesOut, 1)) {
                cb_wait_front(kCbMessagesOut, 1);
                const uint32_t word = l1(get_read_ptr(kCbMessagesOut))[0];
                cb_pop_front(kCbMessagesOut, 1);
                if (word == kMessagesEnd) {
                    ended = true;
                    continue;
                }
                where = kWhereMessagesBadDest;  // the link core's own handler sent
                seen = word;
                break;
            }
            if (!wire.up) {
                // Up once the far hello is read; ours goes once more, after the far
                // side's start-up (its zeroing) has surely passed.
                if (*far_hello == link.token) {
                    wire.up = true;
                    stamps[kStampUp] = wall() | 1;
                    if (!wire.hello(writer)) {
                        where = kWhereLinkSent;
                    }
                } else if (wall64() - entered > wait) {
                    where = kWhereLinkUp;
                    seen = *far_hello;
                } else if (wall64() - hello_at > (1u << 14)) {
                    hello_at = wall64();
                    if (!wire.hello(writer)) {
                        where = kWhereLinkSent;
                    }
                }
                continue;
            }
            bool moved = false;
            uint32_t blocked = 0;
            const uint32_t freed = reader_words[kLinkTaken];
            if (freed != mine[kLinkFreed]) {
                if (wire.sender.edm_has_space_for_packet<1>()) {
                    if (!wire.put_word(writer, 1, kLinkCredit, freed)) {
                        where = kWhereLinkSent;
                        break;
                    }
                    mine[kLinkFreed] = freed;
                    waits[3]++;
                    moved = true;
                } else {
                    blocked = kWhereLinkRouter;
                }
            }
            const uint32_t reserved = *tail - head;
            if (reserved != 0) {
                const uint32_t at = head & (link.outbox - 1);
                uint32_t most = link.outbox - at;
                most = most < reserved ? most : reserved;
                most = most < packet ? most : packet;
                const MessageMarks marks(head, shift, most);
                uint32_t count = 0;
                const uint32_t base = link.region + kLinkOutbox + at * entry;
                while (
                    count < most &&
                    *message_mark_half(base + count * entry + kLinkPrefix) ==
                        marks(count)
                ) {
                    count++;
                }
                const uint32_t packets = mine[kLinkSent];
                if (count == 0) {
                    blocked = kWhereLinkGap;
                    waits[2]++;
                } else if (packets - *credit >= link.inbox) {
                    blocked = kWhereLinkCredit;
                    waits[0]++;
                } else if (!wire.sender.edm_has_space_for_packet<1>()) {
                    blocked = kWhereLinkRouter;
                    waits[1]++;
                } else {
                    l1(base)[0] = packets;
                    l1(base)[1] = count;
                    for (uint32_t word = 0; word < count * entry / 4; word++) {
                        l1(staging)[word] = l1(base)[word];
                    }
                    wire.header(0)->to_noc_unicast_write(
                        NocUnicastCommandHeader{wire.far(
                            kLinkOutbox + link.outbox * entry +
                            (packets % link.inbox) * kLinkPacketBytes
                        )},
                        count * entry
                    );
                    wire.header(2)->to_noc_unicast_atomic_inc(
                        NocUnicastAtomicIncCommandHeader(wire.far(kLinkLanded), 1, true)
                    );
                    invalidate_l1_cache();  // a fence: the NoC reads the stored copy
                    if (stamps[kStampSent] == 0) {
                        stamps[kStampSent] = wall() | 1;
                    }
                    if (
                        !wire.send(writer, 0, staging, count * entry) ||
                        !wire.send(writer, 2, 0, 0)
                    ) {
                        where = kWhereLinkSent;
                        break;
                    }
                    head += count;
                    *head_word = head;
                    sent += count;
                    mine[kLinkSent] = packets + 1;
                    mine[kLinkRunCount] = sent;
                    moved = true;
                }
            }
            if (moved || blocked == 0) {
                stalled_from = 0;
            } else if (stalled_from == 0 || blocked != stalled_where) {
                stalled_from = wall64();
                stalled_where = blocked;
            } else if (wall64() - stalled_from > wait) {
                where = blocked;
                seen = blocked == kWhereLinkCredit ? *credit : head;
            }
        }
        if (where != 0) {
            link_gave_up(arguments.control, link, 1, where, head, run, seen);
            writer.gave_up(kStWriter, kGaveStop, kWhereData, slot(command), run);
            return false;
        }
        return ended;
    }

}  // namespace km
#endif
