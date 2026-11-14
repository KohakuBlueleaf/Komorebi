// The link service: on an island's link core 0, the CPU that holds link 0 sends the
// outbox the island's cores fill over that connection and issues what lands.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_messages.hpp"

#if !defined(COMPILE_FOR_TRISC) && defined(KM_PAGE_LINKS)
#include "km_pages.hpp"
#endif

// The program's own reader and writer arguments before LinkArgs (the same count on
// both; 0: LinkArgs right after MessagesArgs).
#if !defined(KM_LINK_ARGS_AFTER)
#define KM_LINK_ARGS_AFTER 0
#endif

namespace km {

    // Reader and writer program arguments after MessagesArgs and the program's own
    // KM_LINK_ARGS_AFTER.
    struct LinkArgs {
        uint32_t core;        // link core 0 in the core table; kLinkNone: no link
        uint32_t region;      // L1 address of the link's words, outbox and inbox
        uint32_t outbox;      // outbox entries: a power of two, kLinkOutboxMin or more
        uint32_t inbox;       // inbox packets, 1 to kLinkInboxMax
        uint32_t packet;      // entries a packet at most (1: one message a packet)
        uint32_t island;      // this island's number
        uint32_t far;         // the far link core's NoC coordinates, x | y << 16
        uint32_t far_region;  // its region's L1 address
        uint32_t token;       // this load's hello, the same on both sides (never 0)
        uint32_t far1;        // more than two islands: link core 1's far link core,
        uint32_t far2;        // link core 2's (kLinkNone: none)
        uint32_t root;        // island 0's link core 0, x | y << 16: the reports' end
    };

    constexpr uint32_t kLinkArgs = sizeof(LinkArgs) / 4;
    constexpr uint32_t kLinkNone = 0xFFFFFFFF;
    constexpr uint32_t kLinkOutboxMin = kMessagesPacket * 2;
    constexpr uint32_t kLinkInboxMax = 16;
    // A message packet's bytes: tt-metal's default packet, within every island packet
    // the session allows (island_session.LINK_PACKET_BYTES, its least).
    constexpr uint32_t kLinkPacketBytes = 4352;

    // The region, from its address: 16-byte words, each with one writer.
    constexpr uint32_t kLinkOutTail = 0x00;  // outbox positions reserved (atomic adds)
    constexpr uint32_t kLinkOutHead = 0x10;  // outbox positions sent
    constexpr uint32_t kLinkLanded = 0x20;   // inbox packets landed (the far side)
    constexpr uint32_t kLinkCredit = 0x30;   // far inbox packets freed (the far side)
    constexpr uint32_t kLinkReader = 0x40;  // packets taken, messages issued, run, issued in it
    constexpr uint32_t kLinkWriter = 0x50;  // packets sent, freed count sent, run, sent in it
    constexpr uint32_t kLinkFound = 0x60;   // counter tile 0's ready line, as read
    constexpr uint32_t kLinkServed = 0x70;  // the last RUN served (its sequence)
    constexpr uint32_t kLinkWaits = 0x80;   // polls without credit, router space, an
                                            // entry; credit packets
    constexpr uint32_t kLinkGave = 0x90;    // issuing's first give-up (where, position,
                                            // run, word); sending's at 0xA0
    constexpr uint32_t kLinkHello = 0xB0;   // the far side's hello: its token
    constexpr uint32_t kLinkStamps = 0xC0;  // wall clock (low word) at: the first send,
                                            // the first data, the first count, link up
    constexpr uint32_t kLinkHeaders = 0xD0;  // packet headers: data, credit, count
    constexpr uint32_t kLinkHeaderBytes = 64;
    constexpr uint32_t kLinkOutbox = kLinkHeaders + 3 * kLinkHeaderBytes;
    constexpr uint32_t kLinkPrefix = 16;  // an entry: {sequence, count} of its packet
                                          // (first entry), its flat rank, the message
    constexpr uint32_t kLinkPrefixRank = 2;

    // After the sending packet: this side's report {run index | quiet << 31, issued,
    // sent across, version}, the far side's, then two passes of the island's lines.
    constexpr uint32_t kLinkReport = 0x00;
    constexpr uint32_t kLinkFarReport = 0x10;
    constexpr uint32_t kLinkLines = 0x20;
    constexpr uint32_t kLinkLinesMax = 256;
    constexpr uint32_t kLinkQuietBit = 1u << 31;
    constexpr uint32_t kLinkCheckTicks = 1u << 14;  // between two passes over the lines
    // More than two islands, after the lines: each island's report (island 0's link
    // core 0), this core's verdict, its state line (many_pass) and a scratch line.
    constexpr uint32_t kLinkIslands = kLinkLines + 2 * 16 * kLinkLinesMax;
    constexpr uint32_t kLinkVerdict = kLinkIslands + 16 * KM_ISLANDS_MAX;
    constexpr uint32_t kLinkState = kLinkVerdict + 16;
    constexpr uint32_t kLinkScratch = kLinkState + 16;
    constexpr uint32_t kLinkEmpty = 1;  // state: nothing in its outbox, inbox or credit
    constexpr uint32_t kLinkSpread = 2;  // state: the verdict sent to siblings and across

    // Words of kLinkStamps (0 until it happens).
    constexpr uint32_t kStampSent = 0;
    constexpr uint32_t kStampData = 1;
    constexpr uint32_t kStampCount = 2;
    constexpr uint32_t kStampUp = 3;

    // Words of kLinkReader and kLinkWriter.
    constexpr uint32_t kLinkTaken = 0;     // inbox packets taken and freed
    constexpr uint32_t kLinkIssued = 1;    // messages issued again, all runs
    constexpr uint32_t kLinkSent = 0;      // packets sent, all runs
    constexpr uint32_t kLinkFreed = 1;     // the taken count last sent back
    constexpr uint32_t kLinkRun = 2;       // the run the next word counts
    constexpr uint32_t kLinkRunCount = 3;  // messages issued or sent in it

    // Where codes of a link (KM_MESSAGES_GAVE and kLinkGave word 0).
    constexpr uint32_t kWhereLinkArgs = 50;    // its arguments, or the router's packet
    constexpr uint32_t kWhereLinkSpace = 51;   // a writer: no outbox space in time
    constexpr uint32_t kWhereLinkCredit = 52;  // no far inbox packet free
    constexpr uint32_t kWhereLinkRouter = 53;  // no space at the router
    constexpr uint32_t kWhereLinkSent = 54;    // a packet's writes did not leave
    constexpr uint32_t kWhereLinkGap = 55;     // a reserved entry not written
    constexpr uint32_t kWhereLinkSequence = 56;  // a landed packet's sequence, count
    constexpr uint32_t kWhereLinkIsland = 57;    // an entry for no rank of this island
    constexpr uint32_t kWhereLinkOverflow = 58;  // more landed than the inbox holds
    constexpr uint32_t kWhereLinkRunFound = 59;  // the RUN's counter tile not ready
    constexpr uint32_t kWhereLinkUp = 60;        // no hello from the far side
    static_assert(kWhereLinkArgs > kWhereClear, "link where codes after the runtime's");

    // Bytes of an outbox entry, and the entries a packet holds, for `message_bytes`.
    inline uint32_t link_entry(uint32_t message_bytes) {
        return kLinkPrefix + message_bytes;
    }

    inline uint32_t link_packet_max(uint32_t message_bytes) {
        return kLinkPacketBytes / link_entry(message_bytes);
    }

    // Offset of the reports in the region: after the words, the outbox, the inbox and
    // the sending packet.
    inline uint32_t link_reports(
        uint32_t message_bytes, uint32_t outbox, uint32_t inbox
    ) {
        return (
            kLinkOutbox + outbox * link_entry(message_bytes) +
            (inbox + 1) * kLinkPacketBytes
        );
    }

    // Region bytes.
    inline uint32_t link_bytes(
        uint32_t message_bytes, uint32_t outbox, uint32_t inbox
    ) {
        return (link_reports(message_bytes, outbox, inbox) + kLinkScratch + 16);
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    template <typename CoreType>
    inline LinkArgs link_args(const CoreType& core) {
        LinkArgs link_arguments;
        uint32_t* words = &link_arguments.core;
        for (uint32_t index = 0; index < kLinkArgs; index++) {
            words[index] = core.arg(kMessagesArgs + KM_LINK_ARGS_AFTER + index);
        }
        return link_arguments;
    }

    // Whether `link` is a usable link of messages of `message_bytes` (the shapes the
    // link core and the island's writers rely on).
    inline bool link_valid(const LinkArgs& link, uint32_t message_bytes) {
        return (
            link.region % 16 == 0 && link.far_region % 16 == 0 &&
            link.outbox >= kLinkOutboxMin && (link.outbox & (link.outbox - 1)) == 0 &&
            link.inbox >= 1 && link.inbox <= kLinkInboxMax && link.packet >= 1 &&
            link.packet <= link_packet_max(message_bytes) && message_bytes % 16 == 0 &&
            link.token != 0
        );
    }

    // The link core's start-up, before the first command: its region zero (nothing
    // crosses before both islands' connections are held, after their loads).
    inline void link_zero(const LinkArgs& link, uint32_t message_bytes) {
        const uint32_t bytes = link_bytes(message_bytes, link.outbox, link.inbox);
        for (uint32_t word = 0; word < bytes / 4; word++) {
            l1(link.region)[word] = 0;
        }
        invalidate_l1_cache();
    }

    // The link core toward island `island`: link core 0 between two islands; over
    // more, the one of the table's next link (a link core a link).
    inline uint32_t link_toward(
        const LinkArgs& link, const Islands& view, uint32_t island
    ) {
        return view.count() > 2 ? link.core + view.link(island) : link.core;
    }

    // What a wait for outbox room does between two polls (false: stop waiting).
    struct LinkIdle {
        bool operator()() const {
            return true;
        }
    };

    // `count` messages from `base` for flat rank `flat` into link core `to`'s outbox:
    // reserve, wait for room (`head`; `idle` between polls), write, the first mark.
    template <typename CoreType, typename Idle = LinkIdle>
    inline uint32_t link_out_put(
        const CoreType& core,
        const LinkArgs& link,
        uint32_t message_bytes,
        uint32_t to,
        uint32_t& head,
        uint32_t scratch,
        uint32_t flat,
        uint32_t base,
        uint32_t stride,
        uint32_t count,
        Idle idle = Idle()
    ) {
        if (!link_valid(link, message_bytes) || count > link.outbox) {
            return kWhereLinkArgs;
        }
        const uint32_t entry = link_entry(message_bytes);
        const uint32_t shift = __builtin_ctz(link.outbox);
        const Args& arguments = core.args();
        const uint32_t returned_address = (
            arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_RETURNED
        );
        volatile tt_l1_ptr uint32_t* returned = l1(returned_address);
        returned[0] = kMessagesEnd;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(to, link.region + kLinkOutTail),
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
        if (!noc_timeout(core, [&] {
            invalidate_l1_cache();
            return returned[0] != kMessagesEnd;
        })) {
            return kWhereMessagesLost;
        }
        const uint32_t first_position = returned[0];
        // Space: the positions are less than an outbox ahead of the link's head.
        if (first_position + count - head > link.outbox) {
            bool stopped = false;
            if (!noc_timeout(core, [&] {
                noc_async_read(
                    core.core_noc(to, link.region + kLinkOutHead), scratch, 16
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
                if (first_position + count - head <= link.outbox) {
                    return true;
                }
                stopped = !idle();
                return stopped;
            })) {
                return kWhereLinkSpace;
            }
            if (stopped) {
                return kWhereLinkSent;
            }
        }
        const MessageMarks marks(first_position, shift, count);
        *message_mark_half(base) = 0;
        for (uint32_t i = 1; i < count; i++) {
            *message_mark_half(base + i * stride) = marks(i);
        }
        // Each entry's prefix names its flat rank (the head read is done).
        for (uint32_t index = 0; index < 4; index++) {
            l1(scratch)[index] = 0;
        }
        l1(scratch)[kLinkPrefixRank] = flat;
        invalidate_l1_cache();  // a fence: the NoC reads the stored marks
        auto at = [&](uint32_t i) {
            return (
                core.core_noc(
                    to,
                    link.region + kLinkOutbox +
                        ((first_position + i) & (link.outbox - 1)) * entry
                )
            );
        };
        for (uint32_t i = 0; i < count; i++) {
            noc_async_write(scratch, at(i), kLinkPrefix);
            noc_async_write(base + i * stride, at(i) + kLinkPrefix, message_bytes);
        }
        if (!noc_timeout(core, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        })) {
            return kWhereMessagesFlush;
        }
        *message_mark_half(base) = marks(0);
        invalidate_l1_cache();
        noc_async_write(base, at(0) + kLinkPrefix, 16);
        if (!noc_timeout(core, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        })) {
            return kWhereMessagesFlush;
        }
        return 0;
    }

    // A writer's bin for another island goes to the outbox of the link core toward it
    // (link_out_put).
    struct LinkOut {
        LinkArgs link;
        uint32_t message_bytes;
        uint32_t heads[KM_FORWARD_PARTS] = {};  // each link core's head as last read

        template <typename WriterType>
        explicit LinkOut(const WriterType& writer) :
            link(link_args(writer)),
            message_bytes(messages_args(writer).message_bytes) {
        }

        // A rank of another island (kMessagesAway).
        bool takes(uint32_t destination) const {
            return (destination & kMessagesAway) != 0;
        }

        template <typename WriterType>
        uint32_t put(
            WriterType& writer,
            const km_command_t&,
            uint32_t destination,
            uint32_t base,
            uint32_t stride,
            uint32_t count
        ) {
            const uint32_t flat = (
                (destination & kMessagesAway) != 0
                    ? destination & ~kMessagesAway
                    : kLinkNone
            );
            const Islands view = islands_at(writer.args().control);
            const uint32_t to = link_toward(link, view, view.island_of(flat));
            if (to - link.core >= KM_FORWARD_PARTS) {
                return kWhereLinkArgs;
            }
            const uint32_t scratch = (
                writer.args().control + KM_CONTROL_TAILS + KM_TAILS_WRITER
            );
            uint32_t& head = heads[to - link.core];
            return (
                link_out_put(
                    writer,
                    link,
                    message_bytes,
                    to,
                    head,
                    scratch,
                    flat,
                    base,
                    stride,
                    count
                )
            );
        }
    };

    // `count` messages `stride` bytes apart from `base` to local core `destination`:
    // reserve on counter tile `counter`, write, the first mark last. 0 or where.
    template <typename CoreType>
    inline uint32_t link_put(
        const CoreType& core,
        const MessagesArgs& messages_arguments,
        uint32_t counter,
        uint32_t destination,
        uint32_t base,
        uint32_t stride,
        uint32_t count,
        uint32_t returned_address
    ) {
        const Args& arguments = core.args();
        const uint32_t message_bytes = messages_arguments.message_bytes;
        const uint32_t shift = __builtin_ctz(messages_arguments.cap);
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
                counter,
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


    // The first give-up of one side of a link (0 issuing, 1 sending): its words in the
    // region and the run's.
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

    // True when the ring holds the end command of run `run` (any group: the link core
    // is in none of the RUN's).
    template <typename CoreType>
    inline bool link_run_ended(const CoreType& core, uint32_t run) {
        const volatile km_command_t*
            ring = (
                reinterpret_cast<const volatile km_command_t*>(
                    core.control(KM_CONTROL_RING)
                )
            );
        volatile tt_l1_ptr uint32_t* last = core.state() + kStLast;
        for (uint32_t slot_index = 0; slot_index < core.args().depth; slot_index++) {
            const uint32_t sequence = ring[slot_index].sequence;
            if (!newer(sequence, last[slot_index])) {
                continue;
            }
            km_command_t candidate;
            copy_command(&ring[slot_index], &candidate);
            if (
                candidate.sequence == sequence && KM_LANDED(&candidate) &&
                handler(candidate) == KM_HANDLER_MESSAGES_END && candidate.a0 == run
            ) {
                return true;
            }
        }
        return false;
    }

}  // namespace km
#endif

#if !defined(COMPILE_FOR_TRISC) && defined(KM_PAGE_LINKS)
namespace km {

    // The connection link 0's CPU holds (km_pages.hpp): packets one hop on, from
    // headers in the region. Never a second sender on it.
    struct LinkWire {
        LinkArgs link;
        uint32_t entry;
        uint32_t staging;  // the sending packet: a payload is copied here first
        uint32_t reports;  // the reports' offset in both regions

        template <typename Core>
        explicit LinkWire(const Core& core) :
            link(link_args(core)),
            entry(link_entry(messages_args(core).message_bytes)),
            staging(
                link.region + kLinkOutbox + link.outbox * entry +
                link.inbox * kLinkPacketBytes
            ),
            reports(
                link_reports(messages_args(core).message_bytes, link.outbox, link.inbox)
            ) {
            const uint32_t k = core.args().me - link.core;  // link core k's far one
            link.far = k == 1 ? link.far1 : (k == 2 ? link.far2 : link.far);
        }

        static LinkSender& sender() {
            return link_sender();
        }

        // Header `which` zeroed, routed one chip on; the caller sets its NoC command.
        LinkHeaderAt header(uint32_t which) {
            return (
                link_header(link.region + kLinkHeaders + which * kLinkHeaderBytes, 1)
            );
        }

        // A NoC address `offset` bytes into the far region.
        uint64_t far(uint32_t offset) const {
            return (
                link_noc(link.far & 0xFFFF, link.far >> 16, link.far_region + offset)
            );
        }

        // One packet: space, the payload (if any) from `source`, the header, then its
        // writes sent and flushed.
        template <typename Core>
        bool send(const Core& core, uint32_t which, uint32_t source, uint32_t bytes) {
            if (!noc_timeout(core, [&] {
                return sender().space();
            })) {
                return false;
            }
            const LinkHeaderAt made = (
                reinterpret_cast<LinkHeaderAt>(
                    link.region + kLinkHeaders + which * kLinkHeaderBytes
                )
            );
            if (!sender().send_one(made, source, bytes)) {
                return false;
            }
            const bool sent = noc_timeout(core, [] {
                return link_sent();
            });
            return sent && noc_timeout(core, [] {
                return link_flushed();
            });
        }

        // 16 bytes, `words`, to `offset` of the far region.
        template <typename Core>
        bool put_line(
            const Core& core, uint32_t which, uint32_t offset, const uint32_t* words
        ) {
            for (uint32_t index = 0; index < 4; index++) {
                l1(staging)[index] = words[index];
            }
            link_write(header(which), far(offset), 16);
            invalidate_l1_cache();  // a fence: the NoC reads the stored words
            return send(core, which, staging, 16);
        }

        // 16 bytes, `words`, to NoC address `at` (forwarded as forward_route set).
        template <typename Core>
        bool put_line_at(
            const Core& core, uint32_t which, uint64_t at, const uint32_t* words
        ) {
            for (uint32_t index = 0; index < 4; index++) {
                l1(staging)[index] = words[index];
            }
            link_write(header(which), at, 16);
            invalidate_l1_cache();  // a fence: the NoC reads the stored words
            return send(core, which, staging, 16);
        }

        template <typename Core>
        bool put_word(
            const Core& core, uint32_t which, uint32_t offset, uint32_t word
        ) {
            const uint32_t words[4] = {word, 0, 0, 0};
            return put_line(core, which, offset, words);
        }

        template <typename Core>
        bool hello(const Core& core) {
            return put_word(core, 1, kLinkHello, link.token);
        }
    };

    // One spanning RUN's service on link 0's CPU (a0: the RUN's group word): take()
    // issues what landed and keeps the line, send() sends the outbox, until its END.
    template <typename Core>
    class LinkService {
    public:
        LinkService(const Core& core, const km_command_t& command) :
            core_(core),
            arguments_(core.args()),
            messages_arguments_(messages_args(core)),
            wire_(core),
            link_(wire_.link),
            run_command_(group_of(command)),
            view_(islands_at(arguments_.control)),
            ranks_(view_, run_command_, messages_arguments_.counters),
            part_(local_part(view_, run_command_)),
            entry_(wire_.entry),
            inbox_(link_.region + kLinkOutbox + link_.outbox * entry_),
            reports_(link_.region + wire_.reports),
            line_(core.disp_noc(messages_arguments_.lines + 16 * arguments_.me)),
            index_(arguments_.me - link_.core) {
            uint32_t neighbours = 0;
            for (uint32_t i = 0; view_.count() > 2 && i < view_.count(); i++) {
                neighbours += view_.hops(i) == 1 ? 1u : 0u;  // a link core each
            }
            links_ = view_.count() > 2 ? neighbours : 1;
        }

        // The RUN found (counter tile 0 ready for a run this link has not served), the
        // counter tiles ready, this side's first report. False: given up.
        bool start() {
            const uint32_t counters = messages_arguments_.counters;
            if (
                !link_valid(link_, messages_arguments_.message_bytes) ||
                index_ >= links_ || links_ > KM_FORWARD_PARTS || counters == 0 ||
                counters > KM_COUNTERS_MAX || ranks_.ranks_here() == 0 ||
                part_.count + links_ > kLinkLinesMax || !pages_held.open
            ) {
                where_ = kWhereLinkArgs;
                return false;
            }
            volatile tt_l1_ptr uint32_t* found = l1(link_.region + kLinkFound);
            const uint32_t served = l1(link_.region + kLinkServed)[0];
            if (!noc_timeout(core_, [&] {
                noc_async_read(
                    core_.core_noc(
                        ranks_.tile_core(0), arguments_.control + KM_CONTROL_TAILS
                    ),
                    link_.region + kLinkFound,
                    16
                );
                for (uint32_t polls = 0; !ncrisc_noc_reads_flushed(noc_index); polls++) {
                    if (polls > (1u << 20)) {
                        return false;
                    }
                }
                invalidate_l1_cache();
                return found[0] != 0 && found[0] != served && found[1] == 1;
            })) {
                where_ = kWhereLinkRunFound;
                return false;
            }
            run_ = found[0];
            l1(link_.region + kLinkServed)[0] = run_;
            for (uint32_t tile = 0; tile < counters; tile++) {
                if (!messages_ready(core_, ranks_.tile_core(tile), KM_TAILS_READER, run_)) {
                    where_ = kWhereMessagesCounter;
                    return false;
                }
            }
            mine()[kLinkRunCount] = 0;
            mine()[kLinkRun] = run_;
            sending()[kLinkRunCount] = 0;
            invalidate_l1_cache();  // a fence: the count is zero before the run names it
            sending()[kLinkRun] = run_;
            run_index_ = ((report()[0] & ~kLinkQuietBit) + 1) & ~kLinkQuietBit;
            publish(false, 0);
            for (uint32_t i = 0; i < KM_ISLANDS_MAX; i++) {  // the reports as found
                start_[i] = l1(reports_ + kLinkIslands + 16 * i)[3];
            }
            state(false);
            entered_ = hello_at_ = wall64();
            head_ = l1(link_.region + kLinkOutHead)[0];
            return true;
        }

        bool ended() const {
            return link_run_ended(core_, run_);
        }

        uint32_t where() const {
            return where_;
        }

        // One pass of issuing: the next landed packet to the ranks its entries name,
        // or, with nothing landed, the quiet test of both islands and the line.
        void take() {
            volatile tt_l1_ptr uint32_t* landed = l1(link_.region + kLinkLanded);
            volatile tt_l1_ptr uint32_t* stamps = l1(link_.region + kLinkStamps);
            const uint32_t taken = mine()[kLinkTaken];
            const uint32_t landed_count = *landed;
            if (stamps[kStampCount] == 0 && landed_count != taken) {
                stamps[kStampCount] = wall() | 1;
            }
            if (landed_count - taken > link_.inbox) {
                fail(0, kWhereLinkOverflow, landed_count);
                return;
            }
            if (landed_count == taken) {
#if defined(KM_FORWARD)
                if (view_.count() > 2) {
                    many_pass(taken);
                    return;
                }
#endif
                quiet_pass(taken);
                return;
            }
            armed_ = false;
            here_quiet_ = false;
            state(false);
            if (idle_shown_) {  // busy has landed before the first message is issued
                if (!status(KM_MESSAGES_BUSY, 0, 0)) {
                    fail(0, kWhereMessagesLine, 0);
                    return;
                }
                idle_shown_ = false;
            }
            const uint32_t packet = inbox_ + (taken % link_.inbox) * kLinkPacketBytes;
            const uint32_t sequence = l1(packet)[0];
            const uint32_t count = l1(packet)[1];
            if (stamps[kStampData] == 0) {
                stamps[kStampData] = wall() | 1;
            }
            if (sequence != taken || count == 0 || count > link_.packet) {
                fail(0, kWhereLinkSequence, sequence);
                return;
            }
            const uint32_t returned_address = (
                arguments_.control + KM_CONTROL_MESSAGES + KM_MESSAGES_RETURNED
            );
            for (uint32_t index = 0; index < count;) {
                const uint32_t flat = l1(packet + index * entry_)[kLinkPrefixRank];
                const uint32_t island = view_.island_of(flat);
                const uint32_t link = view_.link(island);  // further on: its next link
#if defined(KM_FORWARD)
                const bool onward = island != view_.here();
#else
                const bool onward = false;  // two islands: every message is for here
#endif
                if (
                    onward
                        ? view_.count() <= 2 || link == index_ || link >= links_
                        : ranks_.reserved(flat)
                ) {
                    fail(0, kWhereLinkIsland, flat);
                    return;
                }
                uint32_t row = 1;
                while (
                    index + row < count &&
                    l1(packet + (index + row) * entry_)[kLinkPrefixRank] == flat
                ) {
                    row++;
                }
                const uint32_t message = packet + index * entry_ + kLinkPrefix;
                const uint32_t where = (
                    onward
                        ? link_out_put(
                            core_,
                            link_,
                            messages_arguments_.message_bytes,
                            link_.core + link,
                            heads_[link],
                            reports_ + kLinkScratch,
                            flat,
                            message,
                            entry_,
                            row,
                            [this] {
                                send();  // this outbox drains while the next one is full
                                return where_ == 0;
                            }
                        )
                        : link_put(
                            core_,
                            messages_arguments_,
                            ranks_.counter_core(flat),
                            flat - view_.offset(),
                            message,
                            entry_,
                            row,
                            returned_address
                        )
                );
                if (where != 0) {
                    fail(0, where, flat);
                    return;
                }
                (onward ? forwarded_ : issued_) += row;
                index += row;
            }
            mine()[kLinkIssued] += count;
            mine()[kLinkRunCount] = issued_ + forwarded_;
            mine()[kLinkTaken] = taken + 1;  // the slot is free: credited back
        }

        // One pass of sending: the hello until the link is up, then this side's report,
        // the freed count back, and the outbox's ready entries as one packet.
        void send() {
            volatile tt_l1_ptr uint32_t* stamps = l1(link_.region + kLinkStamps);
            volatile tt_l1_ptr uint32_t* tail = l1(link_.region + kLinkOutTail);
            volatile tt_l1_ptr uint32_t* credit = l1(link_.region + kLinkCredit);
            volatile tt_l1_ptr uint32_t* waits = l1(link_.region + kLinkWaits);
            const uint64_t wait = static_cast<uint64_t>(arguments_.wait_limit) << 10;
            if (stamps[kStampUp] == 0) {
                // Up once the far hello is read; ours goes once more after that.
                if (l1(link_.region + kLinkHello)[0] == link_.token) {
                    stamps[kStampUp] = wall() | 1;
                    if (!wire_.hello(core_)) {
                        fail(1, kWhereLinkSent, 0);
                    }
                } else if (wall64() - entered_ > wait) {
                    fail(1, kWhereLinkUp, l1(link_.region + kLinkHello)[0]);
                } else if (wall64() - hello_at_ > (1u << 14)) {
                    hello_at_ = wall64();
                    if (!wire_.hello(core_)) {
                        fail(1, kWhereLinkSent, 0);
                    }
                }
                return;
            }
            bool moved = false;
            uint32_t blocked = 0;
            if (report()[3] != reported_ && LinkWire::sender().space()) {
                if (!send_report()) {
                    fail(1, kWhereLinkSent, 0);
                    return;
                }
            }
            const uint32_t freed = mine()[kLinkTaken];
            if (freed != sending()[kLinkFreed]) {
                if (LinkWire::sender().space()) {
                    if (!wire_.put_word(core_, 1, kLinkCredit, freed)) {
                        fail(1, kWhereLinkSent, 0);
                        return;
                    }
                    sending()[kLinkFreed] = freed;
                    waits[3]++;
                    moved = true;
                } else {
                    blocked = kWhereLinkRouter;
                }
            }
            const uint32_t reserved = *tail - head_;
            if (reserved != 0) {
                const uint32_t shift = __builtin_ctz(link_.outbox);
                const uint32_t at = head_ & (link_.outbox - 1);
                uint32_t most = link_.outbox - at;
                most = most < reserved ? most : reserved;
                most = most < link_.packet ? most : link_.packet;
                const MessageMarks marks(head_, shift, most);
                uint32_t count = 0;
                const uint32_t base = link_.region + kLinkOutbox + at * entry_;
                while (
                    count < most &&
                    *message_mark_half(base + count * entry_ + kLinkPrefix) ==
                        marks(count)
                ) {
                    count++;
                }
                const uint32_t packets = sending()[kLinkSent];
                if (count == 0) {
                    blocked = kWhereLinkGap;
                    waits[2]++;
                } else if (packets - *credit >= link_.inbox) {
                    blocked = kWhereLinkCredit;
                    waits[0]++;
                } else if (!LinkWire::sender().space()) {
                    blocked = kWhereLinkRouter;
                    waits[1]++;
                } else {
                    l1(base)[0] = packets;
                    l1(base)[1] = count;
                    for (uint32_t word = 0; word < count * entry_ / 4; word++) {
                        l1(wire_.staging)[word] = l1(base)[word];
                    }
                    link_write(
                        wire_.header(0),
                        wire_.far(
                            kLinkOutbox + link_.outbox * entry_ +
                            (packets % link_.inbox) * kLinkPacketBytes
                        ),
                        count * entry_
                    );
                    link_increment(wire_.header(2), wire_.far(kLinkLanded), 1);
                    invalidate_l1_cache();  // a fence: the NoC reads the stored copy
                    if (stamps[kStampSent] == 0) {
                        stamps[kStampSent] = wall() | 1;
                    }
                    if (
                        !wire_.send(core_, 0, wire_.staging, count * entry_) ||
                        !wire_.send(core_, 2, 0, 0)
                    ) {
                        fail(1, kWhereLinkSent, head_);
                        return;
                    }
                    head_ += count;
                    l1(link_.region + kLinkOutHead)[0] = head_;
                    sent_ += count;
                    sending()[kLinkSent] = packets + 1;
                    sending()[kLinkRunCount] = sent_;
                    cross_count(
                        view_.link_area(), KM_CROSS_MESSAGES, count, count * entry_
                    );
                    moved = true;
                }
            }
            if (moved || blocked == 0) {
                stalled_from_ = 0;
            } else if (stalled_from_ == 0 || blocked != stalled_where_) {
                stalled_from_ = wall64();
                stalled_where_ = blocked;
            } else if (wall64() - stalled_from_ > wait) {
                fail(1, blocked, blocked == kWhereLinkCredit ? *credit : head_);
            }
        }

        // The end: the last report sent (the far side may still need it); a give-up's
        // line. kGave* code.
        uint32_t finish() {
            if (
                where_ == 0 && view_.count() <= 2 &&
                l1(link_.region + kLinkStamps)[kStampUp] != 0 && !send_report()
            ) {
                fail(1, kWhereLinkSent, 0);
            }
            if (where_ == 0) {
                return kGaveNone;
            }
            if (where_ != kWhereLinkArgs && where_ != kWhereLinkRunFound) {
                status(KM_MESSAGES_GAVE_UP, mine()[kLinkTaken], where_);
            }
            return kGaveStop;
        }

        void stopped() {
            fail(0, kWhereMessagesStopped, l1(link_.region + kLinkLanded)[0]);
        }

    private:
        static km_command_t group_of(const km_command_t& command) {
            km_command_t run = command;
            run.group = command.a0;
            return run;
        }

        volatile tt_l1_ptr uint32_t* mine() const {
            return l1(link_.region + kLinkReader);
        }

        volatile tt_l1_ptr uint32_t* sending() const {
            return l1(link_.region + kLinkWriter);
        }

        volatile tt_l1_ptr uint32_t* report() const {
            return l1(reports_ + kLinkReport);
        }

        void fail(uint32_t side, uint32_t where, uint32_t seen) {
            if (where_ == 0) {
                where_ = where;
                link_gave_up(
                    arguments_.control, link_, side, where, mine()[kLinkTaken], run_, seen
                );
            }
        }

        // One 16-byte store of {run, state, sent, consumed} at the submitter.
        bool status(uint32_t state, uint32_t sent, uint32_t consumed) {
            volatile tt_l1_ptr uint32_t* out = (
                core_.control(KM_CONTROL_MESSAGES) + KM_MESSAGES_OUT / 4
            );
            out[0] = run_;
            out[1] = state;
            out[2] = sent;
            out[3] = consumed;
            invalidate_l1_cache();  // a fence: the NoC reads the stored line
            noc_async_write(
                arguments_.control + KM_CONTROL_MESSAGES + KM_MESSAGES_OUT, line_, 16
            );
            return noc_timeout(core_, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            });
        }

        // Messages this run sent across.
        uint32_t across() const {
            return sending()[kLinkRun] == run_ ? sending()[kLinkRunCount] : 0u;
        }

        // This side's report for sending: words, then a new version.
        void publish(bool quiet, uint32_t across) {
            volatile tt_l1_ptr uint32_t* words = report();
            const uint32_t first_word = run_index_ | (quiet ? kLinkQuietBit : 0u);
            if (
                words[0] == first_word && words[1] == issued_ && words[2] == across &&
                words[3] != 0
            ) {
                return;
            }
            words[0] = first_word;
            words[1] = issued_;
            words[2] = across;
            invalidate_l1_cache();  // a fence: the version goes last
            words[3] = words[3] + 1 == 0 ? 1 : words[3] + 1;
            invalidate_l1_cache();
        }

        bool send_report() {
            volatile tt_l1_ptr uint32_t* words = report();
            invalidate_l1_cache();
            const uint32_t version = words[3];
            const uint32_t line[4] = {words[0], words[1], words[2], version};
            if (version == reported_) {
                return true;
            }
            if (!wire_.put_line(core_, 1, wire_.reports + kLinkFarReport, line)) {
                return false;
            }
            reported_ = version;
            return true;
        }

        // The island quiet, as its dispatcher sees it: every core of the RUN's part
        // idle in this run over two passes of their lines' sources, and balanced.
        bool island_quiet(uint32_t across) {
            const uint32_t lines = reports_ + kLinkLines;
            for (uint32_t pass = 0; pass < 2; pass++) {
                for (uint32_t core = 0; core < part_.count; core++) {
                    noc_async_read(
                        core_.core_noc(
                            part_.first + core,
                            arguments_.control + KM_CONTROL_MESSAGES + KM_MESSAGES_OUT
                        ),
                        lines + (pass * kLinkLinesMax + core) * 16,
                        16
                    );
                }
                if (!noc_timeout(core_, [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                })) {
                    fail(0, kWhereMessagesRead, 0);
                    return false;
                }
            }
            invalidate_l1_cache();
            uint64_t sent = issued_, consumed = across;
            for (uint32_t core = 0; core < part_.count; core++) {
                volatile tt_l1_ptr uint32_t* one = l1(lines + core * 16);
                volatile tt_l1_ptr uint32_t* two = l1(
                    lines + (kLinkLinesMax + core) * 16
                );
                if (
                    one[0] != run_ || one[1] != KM_MESSAGES_IDLE || two[0] != one[0] ||
                    two[1] != one[1] || two[2] != one[2] || two[3] != one[3]
                ) {
                    return false;
                }
                sent += one[2];
                consumed += one[3];
            }
            return sent == consumed;
        }

        // Both islands quiet: the far report (read twice alike) is quiet in this run
        // and its counts meet ours, or it is of a later run (the far side ended this).
        bool far_quiet(uint32_t across) const {
            volatile tt_l1_ptr uint32_t* far = l1(reports_ + kLinkFarReport);
            const uint32_t words[4] = {far[0], far[1], far[2], far[3]};
            invalidate_l1_cache();
            if (
                far[0] != words[0] || far[1] != words[1] || far[2] != words[2] ||
                far[3] != words[3] || words[3] == 0
            ) {
                return false;
            }
            const uint32_t ahead = (
                ((words[0] & ~kLinkQuietBit) - run_index_) & ~kLinkQuietBit
            );
            if (ahead != 0) {
                return ahead < (kLinkQuietBit >> 1);
            }
            return (
                (words[0] & kLinkQuietBit) != 0 && words[1] == across &&
                words[2] == issued_
            );
        }

        // Nothing landed: this island's quiet (with the link empty), both islands'
        // quiet, then the line idle once that held for the idle cycles.
        void quiet_pass(uint32_t taken) {
            const uint32_t sent_across = across();
            const uint32_t tail = l1(link_.region + kLinkOutTail)[0];
            const bool empty = tail == head_ && sending()[kLinkFreed] == taken;
            if (!empty) {
                here_quiet_ = false;
            } else if (!here_quiet_ && wall64() - checked_ > kLinkCheckTicks) {
                checked_ = wall64();
                here_quiet_ = island_quiet(sent_across);
                if (where_ != 0) {
                    return;
                }
            }
            publish(here_quiet_, sent_across);
            const bool quiet = here_quiet_ && far_quiet(sent_across);
            if (!quiet || idle_shown_) {
                armed_ = armed_ && quiet;
                if (!quiet && idle_shown_) {
                    if (!status(KM_MESSAGES_BUSY, 0, 0)) {
                        fail(0, kWhereMessagesLine, 0);
                    }
                    idle_shown_ = false;
                }
                return;
            }
            if (!armed_ || key_[0] != taken || key_[1] != tail || key_[2] != sent_across) {
                armed_ = true;
                key_[0] = taken;
                key_[1] = tail;
                key_[2] = sent_across;
                since_ = wall64();
                return;
            }
            if (wall64() - since_ < messages_arguments_.idle) {
                return;
            }
            if (!status(KM_MESSAGES_IDLE, issued_, sent_across)) {
                fail(0, kWhereMessagesLine, 0);
                return;
            }
            idle_shown_ = true;
        }

        // More than two islands. This core's state line: {run index, received, sent
        // across, kLinkEmpty | kLinkSpread}.
        void state(bool empty) {
            volatile tt_l1_ptr uint32_t* words = l1(reports_ + kLinkState);
            words[0] = run_index_;
            words[1] = issued_ + forwarded_;
            words[2] = across();
            words[3] = (empty ? kLinkEmpty : 0u) | (spread_ ? kLinkSpread : 0u);
            invalidate_l1_cache();
        }

        // The island quiet (its totals kept): the RUN's cores idle and every link core
        // empty in this run, alike over two passes, and balanced.
        bool many_quiet() {
            const uint32_t lines = reports_ + kLinkLines;
            for (uint32_t pass = 0; pass < 2; pass++) {
                for (uint32_t core = 0; core < part_.count; core++) {
                    noc_async_read(
                        core_.core_noc(
                            part_.first + core,
                            arguments_.control + KM_CONTROL_MESSAGES + KM_MESSAGES_OUT
                        ),
                        lines + (pass * kLinkLinesMax + core) * 16,
                        16
                    );
                }
                for (uint32_t k = 0; k < links_; k++) {
                    noc_async_read(
                        core_.core_noc(link_.core + k, reports_ + kLinkState),
                        lines + (pass * kLinkLinesMax + part_.count + k) * 16,
                        16
                    );
                }
                if (!noc_timeout(core_, [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                })) {
                    fail(0, kWhereMessagesRead, 0);
                    return false;
                }
            }
            invalidate_l1_cache();
            uint64_t sent = 0, consumed = 0, received = 0, out = 0;
            for (uint32_t index = 0; index < part_.count + links_; index++) {
                volatile tt_l1_ptr uint32_t* one = l1(lines + index * 16);
                volatile tt_l1_ptr uint32_t* two = (
                    l1(lines + (kLinkLinesMax + index) * 16)
                );
                if (
                    two[0] != one[0] || two[1] != one[1] || two[2] != one[2] ||
                    two[3] != one[3]
                ) {
                    return false;
                }
                if (index >= part_.count) {  // a link core's state line
                    if (one[0] != run_index_ || (one[3] & kLinkEmpty) == 0) {
                        return false;
                    }
                    received += one[1];
                    out += one[2];
                } else {
                    if (one[0] != run_ || one[1] != KM_MESSAGES_IDLE) {
                        return false;
                    }
                    sent += one[2];
                    consumed += one[3];
                }
            }
            island_received_ = static_cast<uint32_t>(received);
            island_sent_ = static_cast<uint32_t>(out);
            return sent + received == consumed + out;
        }

        // The island's report {run index | quiet, received, sent across, round}, a new
        // round each, into its slot at island 0's link core 0 (sent on between).
        void many_report(bool quiet) {
            volatile tt_l1_ptr uint32_t* words = report();
            words[0] = run_index_ | (quiet ? kLinkQuietBit : 0u);
            words[1] = island_received_;
            words[2] = island_sent_;
            words[3] = words[3] + 1 == 0 ? 1 : words[3] + 1;
            invalidate_l1_cache();
            const uint32_t line[4] = {words[0], words[1], words[2], words[3]};
            const uint32_t slot = reports_ + kLinkIslands + 16 * view_.here();
            if (view_.here() == 0) {
                for (uint32_t index = 0; index < 4; index++) {
                    l1(slot)[index] = line[index];
                }
                invalidate_l1_cache();
                return;
            }
            const uint64_t at = link_noc(link_.root & 0xFFFF, link_.root >> 16, slot);
            forward_route(0, KM_FORWARD_NO_FIELD);
            const bool ok = wire_.put_line_at(core_, 1, at, line);
            forward_route_off();
            if (!ok) {
                fail(1, kWhereLinkSent, 0);
            }
        }

        // Island 0's link core 0: a round once every island reported anew; two rounds
        // in a row all quiet, sums sent and received equal, counts alike: the verdict.
        void collect() {
            uint32_t got[KM_ISLANDS_MAX][4];
            for (uint32_t i = 0; i < view_.count(); i++) {
                volatile tt_l1_ptr uint32_t* slot = (
                    l1(reports_ + kLinkIslands + 16 * i)
                );
                for (uint32_t index = 0; index < 4; index++) {
                    got[i][index] = slot[index];
                }
                invalidate_l1_cache();
                if (
                    slot[0] != got[i][0] || slot[1] != got[i][1] ||
                    slot[2] != got[i][2] || slot[3] != got[i][3] ||
                    got[i][3] == start_[i]
                ) {
                    return;  // not every island reported since the last round
                }
            }
            bool good = true, same = round_good_;
            uint64_t sent = 0, received = 0;
            for (uint32_t i = 0; i < view_.count(); i++) {
                good = good && got[i][0] == (run_index_ | kLinkQuietBit);
                same = same && rounds_[i][0] == got[i][1] && rounds_[i][1] == got[i][2];
                received += got[i][1];
                sent += got[i][2];
                rounds_[i][0] = got[i][1];
                rounds_[i][1] = got[i][2];
                start_[i] = got[i][3];
            }
            good = good && sent == received;
            round_good_ = good;
            if (good && same) {
                l1(reports_ + kLinkVerdict)[0] = run_index_;
                invalidate_l1_cache();
            }
        }

        // The verdict, once: into each other link core of the island, then across.
        void spread() {
            if (spread_) {
                return;
            }
            const uint32_t words[4] = {run_index_, 0, 0, 0};
            for (uint32_t index = 0; index < 4; index++) {
                l1(reports_ + kLinkScratch)[index] = words[index];
            }
            invalidate_l1_cache();  // a fence: the NoC reads the stored words
            for (uint32_t k = 0; k < links_; k++) {
                if (k != index_) {
                    noc_async_write(
                        reports_ + kLinkScratch,
                        core_.core_noc(link_.core + k, reports_ + kLinkVerdict),
                        16
                    );
                }
            }
            const bool written = noc_timeout(core_, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            });
            if (
                !written ||
                !wire_.put_line(core_, 1, wire_.reports + kLinkVerdict, words)
            ) {
                fail(1, kWhereLinkSent, 0);
                return;
            }
            spread_ = true;
        }

        // Every link core of the island spread this run's verdict.
        bool siblings_spread() {
            const uint32_t lines = reports_ + kLinkLines;
            for (uint32_t k = 0; k < links_; k++) {
                noc_async_read(
                    core_.core_noc(link_.core + k, reports_ + kLinkState),
                    lines + k * 16,
                    16
                );
            }
            if (!noc_timeout(core_, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            })) {
                fail(0, kWhereMessagesRead, 0);
                return false;
            }
            invalidate_l1_cache();
            for (uint32_t k = 0; k < links_; k++) {
                volatile tt_l1_ptr uint32_t* one = l1(lines + k * 16);
                if (one[0] != run_index_ || (one[3] & kLinkSpread) == 0) {
                    return false;
                }
            }
            return true;
        }

        // Nothing landed, over more than two islands: the state; link core 0's report
        // until the verdict, then spread, the line idle once every link core spread.
        void many_pass(uint32_t taken) {
            const uint32_t tail = l1(link_.region + kLinkOutTail)[0];
            const bool empty = tail == head_ && sending()[kLinkFreed] == taken;
            invalidate_l1_cache();
            if (l1(reports_ + kLinkVerdict)[0] == run_index_) {
                spread();
                state(empty);
                if (where_ != 0 || index_ != 0 || idle_shown_ || !siblings_spread()) {
                    return;
                }
                if (where_ == 0 && many_quiet()) {
                    if (!status(KM_MESSAGES_IDLE, island_received_, island_sent_)) {
                        fail(0, kWhereMessagesLine, 0);
                        return;
                    }
                    idle_shown_ = true;
                }
                return;
            }
            state(empty);
            if (index_ != 0 || wall64() - checked_ <= kLinkCheckTicks) {
                return;
            }
            checked_ = wall64();
            const bool quiet = empty && many_quiet();
            if (where_ != 0) {
                return;
            }
            many_report(quiet);
            if (where_ == 0 && view_.here() == 0) {
                collect();
            }
        }

        const Core& core_;
        const Args& arguments_;
        const MessagesArgs messages_arguments_;
        LinkWire wire_;
        const LinkArgs& link_;
        const km_command_t run_command_;
        const Islands view_;
        const MessageRanks ranks_;
        const Part part_;
        const uint32_t entry_, inbox_, reports_;
        const uint64_t line_;
        uint32_t where_ = 0, run_ = 0, run_index_ = 0, issued_ = 0, sent_ = 0;
        uint32_t head_ = 0, reported_ = 0, stalled_where_ = 0;
        uint32_t key_[3] = {0, 0, 0};
        uint64_t entered_ = 0, hello_at_ = 0, stalled_from_ = 0, since_ = 0;
        uint64_t checked_ = 0;
        bool idle_shown_ = false, armed_ = false, here_quiet_ = false;
        // over more than two islands (many_pass); island 0's link core 0: each
        // report's round and counts at the last round, and whether it was good
        const uint32_t index_;
        uint32_t links_ = 1, forwarded_ = 0, island_received_ = 0, island_sent_ = 0;
        uint32_t heads_[KM_FORWARD_PARTS] = {};
        uint32_t start_[KM_ISLANDS_MAX] = {};
        uint32_t rounds_[KM_ISLANDS_MAX][2] = {};
        bool round_good_ = false, spread_ = false;
    };

    // A PAGES command of mode KM_PAGES_MESSAGES on link 0's CPU of link core 0 (each,
    // over more than two islands): the link's service of the RUN (a0) until it ends.
    template <typename Core>
    __attribute__((noinline)) uint32_t link_messages(
        const Core& core, const km_command_t& command
    ) {
        LinkService<Core> service(core, command);
        if (service.start()) {
            while (service.where() == 0) {
                invalidate_l1_cache();
                if (core.stop()) {
                    service.stopped();
                    break;
                }
                if (service.ended()) {
                    break;
                }
#if defined(KM_FORWARD)
                forward_turn();  // what this link core sends on for other islands
#endif
                service.take();
                if (service.where() == 0) {
                    service.send();
                }
            }
        }
        return service.finish();
    }

}  // namespace km
#endif
