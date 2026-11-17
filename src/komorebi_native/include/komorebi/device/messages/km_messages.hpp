// Routed messages: a message moves to the queue of another core in DRAM, reserved by
// one atomic add on a counter tile; the generic reader and writer, and compute's calls.
#pragma once

#include <stdint.h>

#include "../core/km_core.hpp"
#include "../core/km_command.hpp"

#if defined(KM_MESSAGE_PARTS)
#include "../overlays/km_overlay.hpp"
#elif defined(KM_MESSAGE_HANDLERS)
#error "the runtime's messages kernels of handlers as parts define KM_MESSAGE_PARTS"
#endif

namespace km {

    // Arguments of a dynamic program: reader and writer program arguments 0 to
    // kMessagesArgs - 1 (the program's own follow), compute program arguments 0 and 1.
    struct MessagesArgs {
        uint32_t handler;  // the dynamic handlers, one per byte (KM_IS_DYNAMIC)
        uint32_t
            queue;  // DRAM address of the queues, kMessagesPageMessages messages a page
        uint32_t
            cap;  // messages per queue: a power of two, at least kMessagesPageMessages
        uint32_t message_bytes;  // a multiple of 64, at most kMessagesMessageMax
        uint32_t lines;     // status lines at the submitter: core me at lines + 16 me
        uint32_t idle;      // cycles the idle condition holds before a line says so
        uint32_t bins;      // writer bins, 1 to kMessagesBinsMax
        uint32_t counters;  // K: the group's last K cores are counter tiles
        uint32_t body;  // DRAM address of the bodies: a page of body_bytes a position
        uint32_t body_bytes;  // the most bytes of a body; 0: the program has none
        uint32_t
            x280;  // X280 ranks after the group's cores (0: none), KM_X280_RANK_CORE
        uint32_t x280_queue;  // bank address of X280 rank 0's queue, cap messages each
        uint32_t x280_bank;   // that bank's DRAM tile, x | y << 16
        uint32_t x280_tile;   // the X280 ranks' cluster: its tile, x | y << 16
        uint32_t x280_credits;  // LIM address there of core 0's credit line (c: + 16 c)
        uint32_t inbox;   // L1 address of this core's inbox, kMessagesInbox messages
        uint32_t direct;  // L1 address of this core's direct area (0: no direct pairs)
        uint32_t stamps;  // L1 address of this core's stamps (0: none)
        uint32_t age;  // cycles: a bin this old is sent while no line comes (0: never)
    };

    constexpr uint32_t kMessagesArgs = sizeof(MessagesArgs) / 4;
    constexpr uint32_t kMessagesCArgMessage = 0;  // compute: message bytes, body bytes
                                                  // above kMessagesCArgBodyShift
    constexpr uint32_t kMessagesCArgBodyShift = 16;
    constexpr uint32_t kMessagesCArgQuota = 1;  // compute: sends at the start of a run
    constexpr uint32_t kMessagesCArgCounters = 2;  // compute: K, as MessagesArgs::counters
    constexpr uint32_t kMessagesCArgX280 = 3;  // compute: X280 ranks, as MessagesArgs::x280
    constexpr uint32_t kMessagesCArgDirect = 4;  // compute: as MessagesArgs::direct
    constexpr uint32_t kMessagesCArgs = 5;

    // Buffers: packets reader -> compute, outgoing messages compute -> writer, bins.
    constexpr uint32_t kCbMessagesIn = 26;
    constexpr uint32_t kCbMessagesOut = 27;
    constexpr uint32_t kCbMessagesBins = 28;
    constexpr uint32_t kMessagesPacket = 32;  // messages of a packet and of a bin
    constexpr uint32_t kMessagesPageMessages = 32;           // messages of a DRAM page of a queue
    constexpr uint32_t kMessagesHeader = 64;  // packet header (n, first position): DRAM
                                              // reads land 64-byte aligned
    constexpr uint32_t kMessagesMessageMax = 256;  // bytes
    constexpr uint32_t kMessagesBinsMax = 16;
    // A core's inbox: the X280 ranks' messages to it, in L1; its reader credits them
    // back every kMessagesInbox / kMessagesCredit messages.
    constexpr uint32_t kMessagesInbox = 32;
    constexpr uint32_t kMessagesCredit = 4;
    static_assert((kMessagesInbox & (kMessagesInbox - 1)) == 0, "an inbox of 2^n");
    constexpr uint32_t kMessagesEnd = 0xFFFFFFFF;  // packet n / outgoing dest: the run ended
    constexpr uint32_t kMessagesWaiting = 1u << 31;  // idle word bit (compute, writer)

    // Idle words (KM_MESSAGES_FLAGS): word 0 the run, word 1 the state.
    constexpr uint32_t kMessagesFlagReader = 0x00;
    constexpr uint32_t kMessagesFlagCompute = 0x10;  // waiting in next() | messages pushed
    constexpr uint32_t kMessagesFlagWriter = 0x20;  // every bin sent | messages taken

    // Positions per core and load; positions never restart. Word 0 of a message: its
    // lap mark (low 16 bits), its handler and its program (0 for now) above it.
    constexpr uint32_t kMessagesPositionMax = 1u << 30;
    constexpr uint32_t kMessagesMarkMask = 0xFFFF;
    constexpr uint32_t kMessagesHandlerShift = 16;
    constexpr uint32_t kMessagesProgramShift = 24;
    constexpr uint32_t kMessagesLaps = 0xFFFF;  // marks 1 to 65535; 0 is empty
    // The word compute hands the writer: the destination rank (16 bits, flat over the
    // islands), the handler above it.
    constexpr uint32_t kMessagesOutHandlerShift = 16;
    constexpr uint32_t kMessagesOutDestination = 0xFFFF;
    // A writer's bin for a rank of another island: this bit above its flat rank.
    constexpr uint32_t kMessagesAway = 1u << 31;

    // A body beside a message (MessagesArgs::body_bytes > 0): word 1 of the message is
    // its body's bytes (0: none); a packet holds at most kMessagesBodyPacket messages.
    constexpr uint32_t kMessagesBodyMax = 4096;
    constexpr uint32_t kMessagesBodyPacket = 4;
    constexpr uint32_t kMessagesBodyWord = 1;
    // Word 1 of an outgoing message: the bytes, the body's L1 address / 64 above.
    constexpr uint32_t kMessagesOutBodyShift = 16;
    constexpr uint32_t kMessagesOutBodyBytes = 0xFFFF;
    constexpr uint32_t kMessagesFlagBodies = 2;  // writer's flag word: bodies written
    constexpr uint32_t kMessagesNoBin = 0xFFFFFFFF;  // the writer: no bin awaits its commit
    constexpr uint32_t kMessagesSettlePolls = 32;  // empty polls before it is committed

    // Stamps (KM_MESSAGES_STAMPS): words of a core's area, wall-clock cycles summed and
    // counts over the load's runs. Writer first, then reader.
    constexpr uint32_t kMessagesStampWords = 32;
    constexpr uint32_t kStampTaken = 0;  // lines taken from compute
    constexpr uint32_t kStampBinWait = 1;  // a line's cycles in its bin, until its flush
    constexpr uint32_t kStampFlushes = 2;  // bins sent
    constexpr uint32_t kStampReserve = 3;  // the reservation's round trip
    constexpr uint32_t kStampLanded = 4;   // waits for written lines to land
    constexpr uint32_t kStampPending = 5;  // a bin written until its commit goes out
    constexpr uint32_t kStampFull = 6;     // flushes: a full bin (or a body)
    constexpr uint32_t kStampEvicted = 7;  // flushes: the bin needed for another rank
    constexpr uint32_t kStampComputeWaits = 8;  // flushes: compute waits (all bins)
    constexpr uint32_t kStampSettled = 9;    // commits after kMessagesSettlePolls polls
    constexpr uint32_t kStampOutEmpty = 10;  // cycles with no outgoing line
    constexpr uint32_t kStampAged = 11;      // flushes: a bin flush_age old
    constexpr uint32_t kStampTails = 16;     // reader: tail reads
    constexpr uint32_t kStampTailCycles = 17;  // reader: their round trips
    constexpr uint32_t kStampPackets = 18;     // reader: queue packets pushed
    constexpr uint32_t kStampMessages = 19;    // reader: their messages
    constexpr uint32_t kStampRead = 20;        // reader: the lines' NoC read
    constexpr uint32_t kStampGaps = 21;  // reader: reads with no valid mark at head
    constexpr uint32_t kStampGapCycles = 22;              // reader: from a new tail to its first mark
    constexpr uint32_t kStampToPush = 23;  // reader: from a new tail to the push
    constexpr uint32_t kStampInPage = 24;  // reader: waits for a free incoming page

    // A core's direct messages (Program.dynamic(direct=)): an L1 area the host writes;
    // its header words are byte offsets in it (program.direct_layout).
    constexpr uint32_t kDirectHeader = 48;
    constexpr uint32_t kDirectReceivers = 0;  // header word: declared receivers
    constexpr uint32_t kDirectSenders = 1;    // header word: declared senders
    constexpr uint32_t kDirectOut = 2;  // a word a receiver: rank | its slot there / 16
    constexpr uint32_t kDirectIn = 3;   // a word a sender: rank | its credit line / 16
    constexpr uint32_t kDirectCredit = 4;   // 16 B a receiver: word 0, lines it took
    constexpr uint32_t kDirectSent = 5;     // a word a receiver: pack's lines to it
    constexpr uint32_t kDirectWritten = 6;  // a word a receiver: the writer's
    constexpr uint32_t kDirectTaken = 7;  // 16 B a sender: word 0, lines this reader took
    constexpr uint32_t kDirectSlots = 8;  // message_bytes a sender, in kDirectIn's order
    constexpr uint32_t kDirectRank = 0xFFFF;
    constexpr uint32_t kDirectFarShift = 16;  // an entry's offset at the far core / 16
    constexpr uint32_t kDirectNone = 0xFFFFFFFF;
    // The word compute hands the writer for a direct line: the receiver's entry, the
    // handler above it.
    constexpr uint32_t kMessagesOutDirect = 1u << 31;
    constexpr uint32_t kMessagesOutEntry = 0xFFFF;

    // A core's direct area at L1 address `base` (0: none).
    struct DirectArea {
        uint32_t base;

        uint32_t word(uint32_t header) const {
            return l1(base)[header];
        }

        uint32_t receivers() const {
            return base == 0 ? 0 : word(kDirectReceivers);
        }

        uint32_t senders() const {
            return base == 0 ? 0 : word(kDirectSenders);
        }

        // Word `index` of the array at header word `header`.
        volatile tt_l1_ptr uint32_t* at(uint32_t header, uint32_t index = 0) const {
            return l1(base + word(header)) + index;
        }

        // The entry of declared receiver `rank`, or kDirectNone (a linear search).
        uint32_t find(uint32_t rank) const {
            const uint32_t count = receivers();
            volatile tt_l1_ptr uint32_t* out = count != 0 ? at(kDirectOut) : nullptr;
            for (uint32_t entry = 0; entry < count; entry++) {
                if ((out[entry] & kDirectRank) == rank) {
                    return entry;
                }
            }
            return kDirectNone;
        }

        // Sender `sender`'s slot (message_bytes) and this reader's taken line for it.
        uint32_t slot(uint32_t sender, uint32_t message_bytes) const {
            return base + word(kDirectSlots) + sender * message_bytes;
        }

        uint32_t taken(uint32_t sender) const {
            return base + word(kDirectTaken) + 16 * sender;
        }
    };

    // The DRAM page of the body at queue position `position` of core `core`.
    inline uint32_t messages_body_page(
        uint32_t core, uint32_t position, uint32_t capacity
    ) {
        return core * capacity + (position & (capacity - 1));
    }

    // A body's bytes as DRAM moves them: whole 64-byte lines.
    inline uint32_t messages_body_lines(uint32_t bytes) {
        return (bytes + 63) & ~63u;
    }

    // The mark of `position` in a queue of 2^shift messages: its lap, 1 to 65535 and
    // round again. A slot holds 0, the previous lap's mark or its own: never confused.
    inline uint32_t messages_mark(uint32_t position, uint32_t shift) {
        const uint32_t lap = position >> shift;
        uint32_t folded = (lap & 0xFFFF) + (lap >> 16);  // lap mod 65535, one fold
        folded = folded >= kMessagesLaps ? folded - kMessagesLaps : folded;
        return folded + 1;
    }

    // The marks of `count` messages (at most a queue) from `position` on, worked out
    // once: the first's lap mark, and the next lap's if the run passes the queue's end.
    struct MessageMarks {
        uint32_t before, after, left;

        MessageMarks(uint32_t position, uint32_t shift, uint32_t count) :
            before(messages_mark(position, shift)),
            after(before),
            left((1u << shift) - (position & ((1u << shift) - 1))) {
            if (left < count) {
                after = messages_mark(position + left, shift);
            }
        }

        uint32_t operator()(uint32_t index) const {
            return index < left ? before : after;
        }
    };

    // Word 0 of the message at `address` as two halves (little endian): the mark, then
    // the handler and program. Each is stored alone, with no read of the other.
    inline volatile tt_l1_ptr uint16_t* message_mark_half(uint32_t address) {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1(address));
    }

    inline volatile tt_l1_ptr uint16_t* message_handler_half(uint32_t address) {
        return message_mark_half(address) + 1;
    }

    // The handler in a message's word 0.
    inline uint32_t messages_handler(uint32_t word) {
        return word >> kMessagesHandlerShift & 0xFF;
    }

    // Handlers as overlay parts (km_message): a RUN names one of the first four. The
    // packet header's groups by handler: their count, each one's end, the messages.
    constexpr uint32_t kMessagesHandlersMax = 16;
    constexpr uint32_t kMessagesGroups = 2;      // header word: groups in the packet
    constexpr uint32_t kMessagesGroupEnds = 16;  // header byte: a byte per group
    constexpr uint32_t kMessagesOrder = 32;      // header byte: a byte per message
    static_assert(
        kMessagesGroupEnds + kMessagesHandlersMax <= kMessagesOrder &&
            kMessagesOrder + kMessagesPacket <= kMessagesHeader,
        "a packet's groups fit its header"
    );

#if defined(KM_MESSAGE_HANDLERS)
    // The overlay of the handler of a message with word 0 `word`: 1 + the handler's
    // place in the program's list (KM_MESSAGE_HANDLERS, from the host); 0: none.
    inline uint32_t message_overlay(uint32_t word) {
        static const uint8_t handlers[] = {KM_MESSAGE_HANDLERS};
        static_assert(sizeof(handlers) <= kMessagesHandlersMax, "too many handlers");
        if (word >> kMessagesProgramShift != 0) {
            return 0;  // another program's
        }
        const uint32_t handler = messages_handler(word);
        for (uint32_t index = 0; index < sizeof(handlers); index++) {
            if (handlers[index] == handler) {
                return index + 1;
            }
        }
        return 0;
    }

    // The reader's message_overlay(): a handler the program's list does not have is
    // overlay n when the live run added it as dynamic handler n (kOverlayMessages).
    inline uint32_t message_number(uint32_t control, uint32_t word) {
        const uint32_t listed = message_overlay(word);
        if (listed != 0 || word >> kMessagesProgramShift != 0) {
            return listed;
        }
        const uint32_t handler = messages_handler(word);
        const uint32_t added = overlay_area(control)[kOverlayMessages / 4];
        return (
            handler <= kMessagesHandlersMax && (added >> handler & 1u) != 0
                ? handler
                : 0
        );
    }
#endif

    // Queue position `position` of core `core`: DRAM page and byte offset in it.
    inline uint32_t messages_page(uint32_t core, uint32_t position, uint32_t capacity) {
        return (
            core * (capacity / kMessagesPageMessages) +
            (position & (capacity - 1)) / kMessagesPageMessages
        );
    }

    // A run's cores: the group's cores before its last `counter_tiles` (counter tiles:
    // a tile that submits atomics can lose an answer; they submit none).
    inline uint32_t messages_ranks(
        const km_command_t& command, uint32_t counter_tiles
    ) {
        return count(command) > counter_tiles ? count(command) - counter_tiles : 0;
    }

    // The counter tile that holds the tail of flat rank `rank`, as this island's core
    // (`island_first`: the island's first rank, held by its core `island_base`).
    inline uint32_t messages_counter(
        const km_command_t& command,
        uint32_t counter_tiles,
        uint32_t rank,
        uint32_t island_first,
        uint32_t island_base
    ) {
        return (
            island_base + first(command) - island_first +
            messages_ranks(command, counter_tiles) +
            (rank - first(command)) % counter_tiles
        );
    }

    // A run's message ranks over the islands its group spans: each island's part gives
    // its cores but its last K, its counter tiles. One island: messages_ranks above.
    class MessageRanks {
    public:
        static constexpr uint32_t kNone = 0xFFFFFFFF;

        MessageRanks(
            const Islands& view, const km_command_t& command, uint32_t counter_tiles
        ) :
            view_(view),
            group_first_(km::first(command)),
            group_end_(km::first(command) + km::count(command)),
            counters_(counter_tiles) {
            for (uint32_t island = 0; island < tables(); island++) {
                uint32_t low = 0, high = 0;
                part(island, low, high);
                parts_ += high > low;
            }
            reserved_ = counters_;
            for (uint32_t island = 0; island < tables(); island++) {
                uint32_t low = 0, high = 0;
                part(island, low, high);
                if (island == here()) {
                    low_ = low;
                    high_ = high;
                    before_ = count_;
                }
                count_ += held(low, high);
            }
        }

        // The run's message ranks; whether its group has parts on several islands.
        uint32_t count() const {
            return count_;
        }

        bool spans() const {
            return parts_ > 1;
        }

        // This island's part, flat: message ranks from low(), then the reserved cores.
        uint32_t low() const {
            return low_;
        }

        uint32_t ranks_here() const {
            return held(low_, high_);
        }

        // Whether flat rank `flat` of this island's part is reserved (no message rank).
        bool reserved(uint32_t flat) const {
            return flat - low_ >= ranks_here();
        }

        // The message rank of flat rank `flat` of this island (reserved: count() on).
        uint32_t rank_of(uint32_t flat) const {
            return (
                reserved(flat)
                    ? count_ + flat - low_ - ranks_here()
                    : before_ + flat - low_
            );
        }

        // The flat rank of message rank `rank` (< count()).
        uint32_t flat_of(uint32_t rank) const {
            for (uint32_t island = 0; island < tables(); island++) {
                uint32_t low = 0, high = 0;
                part(island, low, high);
                if (rank < held(low, high)) {
                    return low + rank;
                }
                rank -= held(low, high);
            }
            return kNone;
        }

        // Counter tile `tile` of this island as its core, and the one that holds flat
        // rank `flat`'s tail (`flat` a message rank of this island).
        uint32_t tile_core(uint32_t tile) const {
            return low_ + ranks_here() + tile - view_.offset();
        }

        uint32_t counter_core(uint32_t flat) const {
            return tile_core((flat - low_) % (counters_ == 0 ? 1 : counters_));
        }

        // Counter tile index of this island's flat rank `flat` (counter tiles: < K).
        uint32_t tile_of(uint32_t flat) const {
            return flat - low_ - ranks_here();
        }

    private:
        // Islands in the table (one, as [0, 2^32), when no table was written).
        uint32_t tables() const {
            return view_.ranks() == 0 ? 1 : view_.count();
        }

        uint32_t here() const {
            return view_.ranks() == 0 ? 0 : view_.here();
        }

        uint32_t held(uint32_t low, uint32_t high) const {
            return high - low > reserved_ ? high - low - reserved_ : 0;
        }

        void part(uint32_t island, uint32_t& low, uint32_t& high) const {
            uint32_t start = 0, end = 0xFFFFFFFF;
            if (view_.ranks() != 0) {
                start = view_.entry_first(island);
                end = start + view_.entry_count(island);
            }
            low = start > group_first_ ? start : group_first_;
            high = end < group_end_ ? end : group_end_;
            high = high > low ? high : low;
        }

        Islands view_;
        uint32_t group_first_, group_end_, counters_;
        uint32_t reserved_ = 0, count_ = 0, before_ = 0, low_ = 0, high_ = 0;
        uint32_t parts_ = 0;
    };

    // Commands the first give-up of a dynamic run (where, position, run).
    inline void messages_gave_up(
        uint32_t control,
        uint32_t where,
        uint32_t position,
        uint32_t run,
        uint32_t seen = 0
    ) {
        volatile tt_l1_ptr uint32_t* gave = (
            l1(control + KM_CONTROL_MESSAGES + KM_MESSAGES_GAVE)
        );
        if (gave[0] == 0) {
            gave[1] = position;
            gave[2] = run;
            gave[3] = seen;
            gave[0] = where;
        }
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {
#if defined(KM_MESSAGES_STAMPS)
#define KM_MSTAMP(statement) statement
#else
#define KM_MSTAMP(statement)
#endif
    // Adds `value` to word `word` of the stamps at `area` (KM_MESSAGES_STAMPS).
    inline void messages_stamp(uint32_t area, uint32_t word, uint32_t value) {
        if (area != 0) {
            volatile tt_l1_ptr uint32_t* words = l1(area);
            words[word] = words[word] + value;
        }
    }

    // Reads the dynamic arguments that follow the start-up arguments.
    inline MessagesArgs messages_args(const Core& core) {
        MessagesArgs messages_arguments;
        uint32_t* words = &messages_arguments.handler;
        for (uint32_t index = 0; index < kMessagesArgs; index++) {
            words[index] = core.arg(index);
        }
        return messages_arguments;
    }

    // Polls a NoC counter `done()` for at most the wait limit. False: it did not.
    template <typename Done>
    inline bool noc_timeout(const Core& core, Done done) {
        const uint64_t end = (
            wall64() + (static_cast<uint64_t>(core.args().wait_limit) << 10)
        );
        for (uint32_t polls = 0;; polls++) {
            if (done()) {
                return true;
            }
            if ((polls & 15) == 15 && wall64() > end) {
                return false;
            }
        }
    }

    // Waits (with a timeout) until counter tile `counter` shows READY for `run`,
    // reading its line into `scratch` (KM_TAILS_READER or _WRITER). False: it did not.
    template <typename CoreType>
    inline bool messages_ready(
        const CoreType& core, uint32_t counter, uint32_t scratch, uint32_t run
    ) {
        const uint32_t control = core.args().control;
        volatile tt_l1_ptr uint32_t* words = l1(control + KM_CONTROL_TAILS + scratch);
        const uint64_t end = (
            wall64() + (static_cast<uint64_t>(core.args().wait_limit) << 10)
        );
        for (;;) {
            noc_async_read(
                core.core_noc(counter, control + KM_CONTROL_TAILS + KM_TAILS_READY),
                control + KM_CONTROL_TAILS + scratch,
                16
            );
            if (!noc_timeout(core, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            })) {
                return false;
            }
            invalidate_l1_cache();
            if (words[0] == run && words[1] == 1) {
                return true;
            }
            if (core.stop() || wall64() > end) {
                return false;
            }
        }
    }

    // True when the next command of rank `me` in the ring is the end of run `run`.
    template <typename CoreType>
    inline bool messages_ended(const CoreType& core, uint32_t run, uint32_t me) {
        const Args& arguments = core.args();
        const volatile km_command_t*
            ring = (
                reinterpret_cast<const volatile km_command_t*>(
                    core.control(KM_CONTROL_RING)
                )
            );
        volatile tt_l1_ptr uint32_t* last = core.state() + kStLast;
        for (uint32_t slot_index = 0; slot_index < arguments.depth; slot_index++) {
            const uint32_t sequence = ring[slot_index].sequence;
            if (!newer(sequence, last[slot_index])) {
                continue;
            }
            km_command_t candidate;
            copy_command(&ring[slot_index], &candidate);
            if (
                candidate.sequence == sequence && KM_LANDED(&candidate) &&
                handler(candidate) == KM_HANDLER_MESSAGES_END && candidate.a0 == run &&
                in_group(candidate, me)
            ) {
                return true;
            }
        }
        return false;
    }

    // Start-up, before the first command: zero-fills this core's queue (mark 0 is
    // empty) from the packet buffer's zeroed first page. False: a write stalled.
    template <typename ReaderType>
    inline bool messages_zero_fill(ReaderType& reader) {
        const MessagesArgs messages_arguments = messages_args(reader);
        const uint32_t page_bytes = (
            kMessagesPageMessages * messages_arguments.message_bytes
        );
        const InterleavedAddrGen<true> queue_addresses = {
            .bank_base_address = messages_arguments.queue, .page_size = page_bytes
        };
        const uint32_t source = get_write_ptr(kCbMessagesIn) + kMessagesHeader;
        for (uint32_t word = 0; word < page_bytes / 4; word++) {
            l1(source)[word] = 0;
        }
        invalidate_l1_cache();  // a fence: the NoC reads the stored zeros
        const uint32_t pages = messages_arguments.cap / kMessagesPageMessages;
        for (uint32_t page = 0; page < pages; page++) {
            noc_async_write(
                source,
                queue_addresses.get_noc_addr(reader.args().me * pages + page),
                page_bytes
            );
            if ((page & 15) == 15 || page + 1 == pages) {
                if (!noc_timeout(reader, [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                })) {
                    return false;
                }
            }
        }
        return true;
    }

#if defined(KM_MESSAGE_HANDLERS)
    // Groups the packet's `count` messages by handler into its header, held handlers
    // first. 0, or the word 0 of a message whose handler is not the program's.
    inline uint32_t message_groups(
        uint32_t control, uint32_t page, uint32_t count, uint32_t message_bytes
    ) {
        uint8_t numbers[kMessagesPacket];
        uint8_t sizes[kMessagesHandlersMax + 1] = {};
        uint8_t starts[kMessagesHandlersMax + 1];
        for (uint32_t index = 0; index < count; index++) {
            const uint32_t word = l1(page + kMessagesHeader + index * message_bytes)[0];
            const uint32_t number = message_number(control, word);
            if (number == 0) {
                return word;
            }
            numbers[index] = number;
            sizes[number]++;
        }
        volatile tt_l1_ptr uint8_t*
            header = (reinterpret_cast<volatile tt_l1_ptr uint8_t*>(page));
        uint32_t groups = 0, at = 0;
        for (uint32_t held = 2; held-- > 0;) {
            for (uint32_t number = 1; number <= kMessagesHandlersMax; number++) {
                if (
                    sizes[number] != 0 &&
                    overlay_members_held(control, number) == (held != 0)
                ) {
                    starts[number] = at;
                    at += sizes[number];
                    header[kMessagesGroupEnds + groups++] = at;
                }
            }
        }
        for (uint32_t index = 0; index < count; index++) {
            header[kMessagesOrder + starts[numbers[index]]++] = index;
        }
        l1(page)[kMessagesGroups] = groups;
        return 0;
    }

    // The overlay of group `group` of the packet at `page`: its first message's.
    inline uint32_t message_group_overlay(
        uint32_t control, uint32_t page, uint32_t group, uint32_t message_bytes
    ) {
        volatile tt_l1_ptr uint8_t*
            header = (reinterpret_cast<volatile tt_l1_ptr uint8_t*>(page));
        const uint32_t first = group == 0 ? 0 : header[kMessagesGroupEnds + group - 1];
        return (
            message_number(
                control,
                l1(
                    page + kMessagesHeader +
                    header[kMessagesOrder + first] * message_bytes
                )[0]
            )
        );
    }

    // The reader loads the part of every group of a packet it has pushed, in order:
    // compute meets it there when the tile does not hold one. False: gave up.
    template <typename ReaderType>
    inline bool message_loads(
        ReaderType& reader, uint32_t page, uint32_t message_bytes
    ) {
        const uint32_t groups = l1(page)[kMessagesGroups];
        for (uint32_t group = 0; group < groups; group++) {
            Overlay<ReaderType> part(
                reader,
                message_group_overlay(reader.args().control, page, group, message_bytes)
            );
            if (!part.load()) {
                return false;
            }
        }
        return true;
    }
#endif

#if defined(KM_DELIVER) && !defined(COMPILE_FOR_TRISC)
    // A direct pair across islands (km_deliver.hpp): the line by km::write_l1, its slot
    // back by a posted km::add_word. kGave* codes.
    template <typename Kernel>
    uint32_t direct_far_line(
        const Kernel&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t
    );
    template <typename Kernel>
    uint32_t direct_far_credit(const Kernel&, uint32_t, uint32_t);
#endif

    // Generic reader: this core's queue in order, to compute in packets; idle and busy
    // lines to the submitter; leaves on the run's end, the stop flag or an error.
    template <typename ReaderType>
    inline void messages_reader(
        ReaderType& reader, const km_command_t& command, bool filled = true
    ) {
        const Args& arguments = reader.args();
        const MessagesArgs messages_arguments = messages_args(reader);
        const uint32_t run = command.sequence;
        volatile tt_l1_ptr uint32_t* messages = reader.control(KM_CONTROL_MESSAGES);
        volatile tt_l1_ptr uint32_t* tail = messages + KM_MESSAGES_IN_TAIL / 4;
        volatile tt_l1_ptr uint32_t* head_word = messages + KM_MESSAGES_IN_HEAD / 4;
        volatile tt_l1_ptr uint32_t*
            compute_flags = (messages + (KM_MESSAGES_FLAGS + kMessagesFlagCompute) / 4);
        volatile tt_l1_ptr uint32_t*
            writer_flags = (messages + (KM_MESSAGES_FLAGS + kMessagesFlagWriter) / 4);
        volatile tt_l1_ptr uint32_t*
            reader_flags = (messages + (KM_MESSAGES_FLAGS + kMessagesFlagReader) / 4);
        volatile tt_l1_ptr uint32_t* out = messages + KM_MESSAGES_OUT / 4;
        const InterleavedAddrGen<true> queue_addresses = {
            .bank_base_address = messages_arguments.queue,
            .page_size = kMessagesPageMessages * messages_arguments.message_bytes
        };
        const InterleavedAddrGen<true> body_addresses = {
            .bank_base_address = messages_arguments.body,
            .page_size = messages_arguments.body_bytes
        };
        const uint32_t in_pages = get_local_cb_interface(kCbMessagesIn).fifo_num_pages;
        const uint64_t line = (
            reader.disp_noc(messages_arguments.lines + 16 * arguments.me)
        );
        const uint64_t wait = static_cast<uint64_t>(arguments.wait_limit) << 10;
        const uint32_t shift = __builtin_ctz(messages_arguments.cap);
        reader_flags[1] = 0;
        reader_flags[0] = run;
        uint32_t head = *head_word, where = filled ? 0 : kWhereMessagesFill;
        if (
            (get_write_ptr(kCbMessagesIn) | messages_arguments.message_bytes) % 64 != 0
        ) {
            where = kWhereMessagesAlign;  // a DRAM read lands 64-byte aligned or not as
                                          // asked
        }
        if (head >= kMessagesPositionMax) {
            where = kWhereMessagesFull;  // positions stay below 2^31 in a load
        }
        // A counter tile sets its cores' tails to their heads, then shows READY; a
        // core waits for its counter tile's READY before it reads its tail there.
        const uint32_t counter_tiles = messages_arguments.counters;
        // a core's flat rank less its core of the card (arguments.me: its core, for
        // queues and lines)
        const uint32_t offset = islands_at(arguments.control).offset();
        const MessageRanks message_ranks(
            islands_at(arguments.control), command, counter_tiles
        );
        const uint32_t ranks = message_ranks.ranks_here();
        const uint32_t me = offset + arguments.me;
        const bool counter = message_ranks.reserved(me);
        if (counter_tiles == 0 || counter_tiles > KM_COUNTERS_MAX || ranks == 0) {
            where = where != 0 ? where : kWhereMessagesCounter;
        } else if (where == 0 && counter) {
            *tail = head;  // nobody sends here: the loop sees an empty queue
            uint32_t served = 0;
            for (  // this tile's ranks: every counter_tiles-th of the run
                uint32_t rank = message_ranks.tile_of(me);
                rank < ranks && message_ranks.tile_of(me) < counter_tiles;
                rank += counter_tiles
            ) {
                const uint32_t core = message_ranks.low() + rank - offset;
                noc_async_read(
                    reader.core_noc(
                        core,
                        arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_IN_HEAD
                    ),
                    arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE + 16 * core,
                    16
                );
                served++;
            }
            if (!noc_timeout(reader, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            })) {
                where = kWhereMessagesRead;
            } else {
                volatile tt_l1_ptr uint32_t* ready = reader.control(KM_CONTROL_TAILS);
                ready[1] = 1;
                ready[2] = served;
                ready[3] = 0;
                invalidate_l1_cache();  // a fence: the run's word goes last
                ready[0] = run;
                invalidate_l1_cache();
            }
        } else if (
            where == 0 &&
            !messages_ready(
                reader, message_ranks.counter_core(me), KM_TAILS_READER, run
            )
        ) {
            where = kWhereMessagesCounter;
        }
#if defined(KM_MESSAGE_HANDLERS)
        // The RUN command's handler seeds: its part, met by compute before the seeds.
        uint32_t refused = 0;
        if (where == 0 && !counter) {
            Overlay<ReaderType> seeds(
                reader, message_overlay(handler(command) << kMessagesHandlerShift)
            );
            where = seeds.load() ? 0 : kWhereMessagesLoad;
        }
#endif
        const uint64_t tail_line = (
            counter
                ? 0
                : reader.core_noc(
                    message_ranks.counter_core(me),
                    arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE +
                        16 * arguments.me
                )
        );
        // A core's tail, read from its counter tile into IN_TAIL (with a timeout).
        KM_MSTAMP(const uint32_t stamps = messages_arguments.stamps);
        KM_MSTAMP(uint32_t tail_seen = 0);  // when a tail beyond head was first read
        auto refresh = [&] {
            if (counter) {
                return true;  // nobody sends to a counter tile: its tail stays its head
            }
            KM_MSTAMP(const uint32_t asked = wall());
            noc_async_read(
                tail_line,
                arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_IN_TAIL,
                16
            );
            const bool ok = noc_timeout(reader, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            });
            invalidate_l1_cache();
            KM_MSTAMP(messages_stamp(stamps, kStampTails, 1));
            KM_MSTAMP(messages_stamp(stamps, kStampTailCycles, wall() - asked));
            return ok;
        };
        // The second source, this core's inbox in L1: the X280 ranks' messages, taken
        // as the prefix of valid marks from its head, credited back to their cluster.
        const bool inbox = messages_arguments.x280 != 0 && !counter;
        volatile tt_l1_ptr uint32_t* inbox_word = messages + KM_MESSAGES_INBOX_HEAD / 4;
        uint32_t inbox_head = *inbox_word, credited = inbox_head;
        auto arrived = [&] {
            uint32_t valid = 0;
            while (
                inbox && valid < kMessagesPacket &&
                (
                    l1(
                        messages_arguments.inbox +
                        ((inbox_head + valid) & (kMessagesInbox - 1)) *
                            messages_arguments.message_bytes
                    )[0] &
                    kMessagesMarkMask
                ) ==
                    messages_mark(inbox_head + valid, __builtin_ctz(kMessagesInbox))
            ) {
                valid++;
            }
            return valid;
        };
        // The inbox's head to this core's credit line in the X280 ranks' cluster.
        auto credit = [&] {
            inbox_word[1] = inbox_word[2] = inbox_word[3] = 0;
            invalidate_l1_cache();  // a fence: the NoC reads the stored line
            noc_async_write(
                arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_INBOX_HEAD,
                get_noc_addr(
                    messages_arguments.x280_tile & 0xFFFF,
                    messages_arguments.x280_tile >> 16,
                    messages_arguments.x280_credits +
                        KM_X280_CREDIT_BYTES * arguments.me
                ),
                KM_X280_CREDIT_BYTES
            );
            credited = inbox_head;
            return noc_timeout(reader, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            });
        };
        // The third source, the slots of this core's declared senders in L1: valid when
        // a slot's mark is that of its next line (the pair's count, as a lap).
        const DirectArea direct{counter ? 0 : messages_arguments.direct};
        const uint32_t direct_senders = direct.senders();
        if (direct_senders + direct.receivers() != 0 && first(command) != 0) {
            where = where != 0 ? where
                               : kWhereMessagesDirect;  // tables of ranks from 0
        }
        const uint32_t slot_bytes = (
            messages_arguments.message_bytes + messages_arguments.body_bytes
        );
        auto direct_valid = [&](uint32_t sender) {
            return (
                (l1(direct.slot(sender, slot_bytes))[0] & kMessagesMarkMask) ==
                messages_mark(l1(direct.taken(sender))[0], 0)
            );
        };
        auto direct_arrived = [&] {
            uint32_t valid = 0;
            for (uint32_t sender = 0; sender < direct_senders; sender++) {
                valid += direct_valid(sender);
            }
            return valid < kMessagesPacket ? valid : kMessagesPacket;
        };
        bool idle_shown = false, busy = true, armed = false;
        uint32_t key[3] = {0, 0, 0};
        uint64_t since = 0, waiting_from = 0;

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
        // The idle condition, as one comparable sample (compute and writer words).
        auto sample = [&](uint32_t tail_value, uint32_t* words) {
            words[0] = tail_value;
            words[1] = compute_flags[1];
            words[2] = writer_flags[1];
            return (
                tail_value == head && arrived() == 0 && direct_arrived() == 0 &&
                compute_flags[0] == run && writer_flags[0] == run &&
                (words[1] & kMessagesWaiting) && (words[2] & kMessagesWaiting) &&
                words[1] == words[2] &&
                cb_pages_reservable_at_back(kCbMessagesIn, in_pages)
            );
        };
        // True when the next command of this core in the ring is this run's end.
        auto end_command = [&] {
            return messages_ended(reader, run, me);
        };

        while (where == 0) {
            invalidate_l1_cache();
            if (reader.stop()) {
                where = kWhereMessagesStopped;
                break;
            }
            if (!refresh()) {
                where = kWhereMessagesRead;
                break;
            }
            const uint32_t tail_value = *tail;
            if (tail_value - head > messages_arguments.cap) {
                where = kWhereMessagesOverflow;
                break;
            }
            KM_MSTAMP(if (tail_value != head && tail_seen == 0) {
                tail_seen = wall();
            });
            const uint32_t from_inbox = arrived();
            const uint32_t from_direct = direct_arrived();
            if (tail_value == head && from_inbox == 0 && from_direct == 0) {
                if (end_command()) {
                    break;
                }
                uint32_t first_sample[3];
                if (!busy || !sample(tail_value, first_sample)) {
                    armed = false;
                    continue;
                }
                const uint32_t sent = messages[KM_MESSAGES_SENT / 4];
                const uint32_t consumed = messages[KM_MESSAGES_CONSUMED / 4];
                uint32_t second_sample[3];
                invalidate_l1_cache();
                if (!refresh()) {
                    where = kWhereMessagesRead;
                    break;
                }
                if (
                    !sample(*tail, second_sample) ||
                    second_sample[0] != first_sample[0] ||
                    second_sample[1] != first_sample[1] ||
                    second_sample[2] != first_sample[2]
                ) {
                    armed = false;
                    continue;
                }
                if (
                    !armed || key[0] != first_sample[0] || key[1] != first_sample[1] ||
                    key[2] != first_sample[2]
                ) {
                    armed = true;
                    key[0] = first_sample[0];
                    key[1] = first_sample[1];
                    key[2] = first_sample[2];
                    since = wall64();
                    continue;
                }
                if (wall64() - since < messages_arguments.idle) {
                    continue;
                }
                if (!status(KM_MESSAGES_IDLE, sent, consumed)) {
                    where = kWhereMessagesLine;
                    break;
                }
                reader_flags[1] = 1;
                idle_shown = true;
                busy = false;  // until the next message is taken
                continue;
            }
            busy = true;
            armed = false;
            if (idle_shown) {  // busy has landed before the first message is taken
                if (!status(KM_MESSAGES_BUSY, 0, 0)) {
                    where = kWhereMessagesLine;
                    break;
                }
                reader_flags[1] = 0;
                idle_shown = false;
            }
            KM_MSTAMP(const uint32_t page_asked = wall());
            if (reader.poll_at(
                kWhereMessagesInPage,
                [&] {
                    return cb_pages_reservable_at_back(kCbMessagesIn, 1);
                },
                arguments.wait_limit
            ) != kGaveNone) {
                where = kWhereMessagesInPage;
                break;
            }
            KM_MSTAMP(messages_stamp(stamps, kStampInPage, wall() - page_asked));
            cb_reserve_back(kCbMessagesIn, 1);
            const uint32_t page = get_write_ptr(kCbMessagesIn);
            if (from_direct != 0) {  // the slots first: each sender waits for its own
                const uint32_t words = messages_arguments.message_bytes / 4;
                const uint32_t most = (
                    messages_arguments.body_bytes != 0
                        ? kMessagesBodyPacket
                        : from_direct
                );
                uint16_t senders[kMessagesPacket];
                uint32_t taken = 0;
                const uint32_t wanted = from_direct < most ? from_direct : most;
                for (uint32_t sender = 0; sender < direct_senders; sender++) {
                    if (taken == wanted || where != 0) {
                        break;
                    }
                    if (!direct_valid(sender)) {
                        continue;
                    }
                    const uint32_t slot = direct.slot(sender, slot_bytes);
                    volatile tt_l1_ptr uint32_t* message = l1(slot);
                    volatile tt_l1_ptr uint32_t* to = (
                        l1(
                            page + kMessagesHeader +
                            taken * messages_arguments.message_bytes
                        )
                    );
                    for (uint32_t word = 0; word < words; word++) {
                        to[word] = message[word];
                    }
                    // A body: from the slot into the packet page, by the NoC.
                    const uint32_t bytes = (
                        messages_arguments.body_bytes != 0
                            ? message[kMessagesBodyWord]
                            : 0
                    );
                    if (bytes > messages_arguments.body_bytes) {
                        where = kWhereMessagesBody;
                    } else if (bytes != 0) {
                        noc_async_read(
                            reader.core_noc(
                                arguments.me, slot + messages_arguments.message_bytes
                            ),
                            page + kMessagesHeader +
                                kMessagesPacket * messages_arguments.message_bytes +
                                taken * messages_arguments.body_bytes,
                            messages_body_lines(bytes)
                        );
                    }
                    senders[taken++] = sender;
                }
                if (where != 0) {
                    break;
                }
                if (!noc_timeout(reader, [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                })) {
                    where = kWhereMessagesRead;
                    break;
                }
                invalidate_l1_cache();
                // The slots go back: this reader's count to each sender's credit line.
                for (uint32_t index = 0; index < taken; index++) {
                    volatile tt_l1_ptr uint32_t* line = (
                        l1(direct.taken(senders[index]))
                    );
                    line[0] = line[0] + 1;
                    invalidate_l1_cache();  // a fence: the NoC reads the stored line
                    const uint32_t entry = direct.at(kDirectIn)[senders[index]];
#if defined(KM_DELIVER)
                    const uint32_t from = message_ranks.flat_of(entry & kDirectRank);
                    if (from - message_ranks.low() >= message_ranks.ranks_here()) {
                        if (direct_far_credit(
                            reader,
                            from,
                            messages_arguments.direct +
                                ((entry >> kDirectFarShift) << 4)
                        ) != kGaveNone) {
                            where = kWhereFar;
                            break;
                        }
                        continue;  // another island's sender: its count + 1 there
                    }
#endif
                    noc_async_write(
                        direct.taken(senders[index]),
                        reader.core_noc(
                            message_ranks.flat_of(entry & kDirectRank) - offset,
                            messages_arguments.direct +
                                ((entry >> kDirectFarShift) << 4)
                        ),
                        16
                    );
                }
                if (!noc_timeout(reader, [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                })) {
                    where = kWhereMessagesFlush;
                    break;
                }
#if defined(KM_MESSAGE_HANDLERS)
                refused = (
                    message_groups(
                        arguments.control, page, taken, messages_arguments.message_bytes
                    )
                );
                if (refused != 0) {
                    where = kWhereMessagesHandler;
                    break;
                }
#endif
                l1(page)[0] = taken;
                l1(page)[1] = 0;
                cb_push_back(kCbMessagesIn, 1);
#if defined(KM_MESSAGE_HANDLERS)
                if (!message_loads(reader, page, messages_arguments.message_bytes)) {
                    where = kWhereMessagesLoad;
                    break;
                }
#endif
                continue;
            }
            if (from_inbox != 0) {  // the inbox first: its room is the X280 ranks' wait
                const uint32_t words = messages_arguments.message_bytes / 4;
                for (uint32_t i = 0; i < from_inbox; i++) {
                    volatile tt_l1_ptr uint32_t* message = (
                        l1(
                            messages_arguments.inbox +
                            ((inbox_head + i) & (kMessagesInbox - 1)) *
                                messages_arguments.message_bytes
                        )
                    );
                    volatile tt_l1_ptr uint32_t* to = (
                        l1(
                            page + kMessagesHeader + i * messages_arguments.message_bytes
                        )
                    );
                    for (uint32_t word = 0; word < words; word++) {
                        to[word] = message[word];
                    }
                }
#if defined(KM_MESSAGE_HANDLERS)
                refused = (
                    message_groups(
                        arguments.control,
                        page,
                        from_inbox,
                        messages_arguments.message_bytes
                    )
                );
                if (refused != 0) {
                    where = kWhereMessagesHandler;
                    break;
                }
#endif
                l1(page)[0] = from_inbox;
                l1(page)[1] = inbox_head;
                cb_push_back(kCbMessagesIn, 1);
                inbox_head += from_inbox;
                *inbox_word = inbox_head;
                if (
                    inbox_head - credited >= kMessagesInbox / kMessagesCredit &&
                    !credit()
                ) {
                    where = kWhereMessagesFlush;
                    break;
                }
#if defined(KM_MESSAGE_HANDLERS)
                if (!message_loads(reader, page, messages_arguments.message_bytes)) {
                    where = kWhereMessagesLoad;
                    break;
                }
#endif
                continue;
            }
            const uint32_t packet = (
                messages_arguments.body_bytes != 0
                    ? kMessagesBodyPacket
                    : kMessagesPacket
            );
            uint32_t batch = tail_value - head < packet ? tail_value - head : packet;
            KM_MSTAMP(const uint32_t read_at = wall());
            for (uint32_t i = 0; i < batch;) {
                const uint32_t position = (head + i) & (messages_arguments.cap - 1);
                const uint32_t in_page = position % kMessagesPageMessages;
                uint32_t chunk = kMessagesPageMessages - in_page;
                chunk = chunk < batch - i ? chunk : batch - i;
                noc_async_read(
                    queue_addresses.get_noc_addr(
                        messages_page(arguments.me, position, messages_arguments.cap),
                        in_page * messages_arguments.message_bytes
                    ),
                    page + kMessagesHeader + i * messages_arguments.message_bytes,
                    chunk * messages_arguments.message_bytes
                );
                i += chunk;
            }
            if (!noc_timeout(reader, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            })) {
                where = kWhereMessagesRead;
                break;
            }
            invalidate_l1_cache();
            KM_MSTAMP(messages_stamp(stamps, kStampRead, wall() - read_at));
            uint32_t valid = 0;
            const MessageMarks marks(head, shift, batch);
            // The prefix of valid marks: a message is valid only after its packet's
            // first message, whose mark is written once the whole packet has landed.
            while (
                valid < batch &&
                (
                    l1(
                        page + kMessagesHeader +
                        valid * messages_arguments.message_bytes
                    )[0] &
                    kMessagesMarkMask
                ) == marks(valid)
            ) {
                valid++;
            }
            if (valid == 0) {  // the message at head has not landed yet
                KM_MSTAMP(messages_stamp(stamps, kStampGaps, 1));
                const uint64_t now = wall64();
                if (waiting_from == 0) {
                    waiting_from = now;
                } else if (now - waiting_from > wait) {
                    where = kWhereMessagesGap;
                    break;
                }
                continue;
            }
            waiting_from = 0;
            KM_MSTAMP(messages_stamp(stamps, kStampGapCycles, wall() - tail_seen));
            if (messages_arguments.body_bytes != 0) {
                // The bodies after the packet's messages, a page each: landed before
                // the packet's mark (the writer's barrier).
                const uint32_t bodies = (
                    page + kMessagesHeader +
                    kMessagesPacket * messages_arguments.message_bytes
                );
                for (uint32_t index = 0; index < valid && where == 0; index++) {
                    const uint32_t bytes = (
                        l1(
                            page + kMessagesHeader +
                            index * messages_arguments.message_bytes
                        )[kMessagesBodyWord]
                    );
                    if (bytes > messages_arguments.body_bytes) {
                        where = kWhereMessagesBody;
                    } else if (bytes != 0) {
                        noc_async_read(
                            body_addresses.get_noc_addr(messages_body_page(
                                arguments.me, head + index, messages_arguments.cap
                            )),
                            bodies + index * messages_arguments.body_bytes,
                            messages_body_lines(bytes)
                        );
                    }
                }
                if (where != 0) {
                    break;
                }
                if (!noc_timeout(reader, [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                })) {
                    where = kWhereMessagesRead;
                    break;
                }
                invalidate_l1_cache();
            }
#if defined(KM_MESSAGE_HANDLERS)
            refused = (
                message_groups(
                    arguments.control, page, valid, messages_arguments.message_bytes
                )
            );
            if (refused != 0) {
                where = kWhereMessagesHandler;
                break;
            }
#endif
            l1(page)[0] = valid;
            l1(page)[1] = head;
            cb_push_back(kCbMessagesIn, 1);
            KM_MSTAMP(messages_stamp(stamps, kStampToPush, wall() - tail_seen));
            KM_MSTAMP(messages_stamp(stamps, kStampPackets, 1));
            KM_MSTAMP(messages_stamp(stamps, kStampMessages, valid));
            KM_MSTAMP(tail_seen = 0);
            head += valid;
            *head_word = head;
#if defined(KM_MESSAGE_HANDLERS)
            if (!message_loads(reader, page, messages_arguments.message_bytes)) {
                where = kWhereMessagesLoad;
                break;
            }
#endif
        }
        if (where != 0) {
            // a gap: word 0 of the message found at the head, for the report
            const uint32_t seen = l1(get_write_ptr(kCbMessagesIn))[kMessagesHeader / 4];
#if defined(KM_MESSAGE_HANDLERS)
            messages_gave_up(
                arguments.control, where, head, run, refused != 0 ? refused : seen
            );
#else
            messages_gave_up(arguments.control, where, head, run, seen);
#endif
            status(KM_MESSAGES_GAVE_UP, head, where);
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

    // Where a writer's bins go when their destination is not a queue of this island:
    // nowhere (every destination is local). km_link.hpp's LinkOut takes the others.
    struct LocalOnly {
        bool takes(uint32_t) const {
            return false;
        }

        template <typename WriterType>
        uint32_t put(
            WriterType&,
            const km_command_t&,
            uint32_t,
            uint32_t,
            uint32_t,
            uint32_t,
            uint32_t = 0,
            uint32_t = 0
        ) {
            return 0;
        }
    };

    // Generic writer: bins by destination, one atomic add a bin reserves positions,
    // then the messages go to that queue, or to `away` if it takes it. True: ended.
    template <typename WriterType, typename Away = LocalOnly>
    inline bool messages_writer(
        WriterType& writer, const km_command_t& command, Away away = Away()
    ) {
        const Args& arguments = writer.args();
        const MessagesArgs messages_arguments = messages_args(writer);
        const uint32_t run = command.sequence;
        const uint32_t words = messages_arguments.message_bytes / 4;
        const uint32_t shift = __builtin_ctz(messages_arguments.cap);
        const uint32_t bins = (
            messages_arguments.bins == 0 || messages_arguments.bins > kMessagesBinsMax
                ? kMessagesBinsMax
                : messages_arguments.bins
        );
        volatile tt_l1_ptr uint32_t* messages = writer.control(KM_CONTROL_MESSAGES);
        volatile tt_l1_ptr uint32_t*
            writer_flags = (messages + (KM_MESSAGES_FLAGS + kMessagesFlagWriter) / 4);
        volatile tt_l1_ptr uint32_t*
            compute_flags = (messages + (KM_MESSAGES_FLAGS + kMessagesFlagCompute) / 4);
        volatile tt_l1_ptr uint32_t* returned = messages + KM_MESSAGES_RETURNED / 4;
        const uint32_t returned_address = (
            arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_RETURNED
        );
        const InterleavedAddrGen<true> queue_addresses = {
            .bank_base_address = messages_arguments.queue,
            .page_size = kMessagesPageMessages * messages_arguments.message_bytes
        };
        const InterleavedAddrGen<true> body_addresses = {
            .bank_base_address = messages_arguments.body,
            .page_size = messages_arguments.body_bytes
        };
        const uint32_t store = get_write_ptr(kCbMessagesBins);
        const uint32_t island_first = islands_at(arguments.control).first();
        // a core's flat rank less its core of the card
        const uint32_t offset = islands_at(arguments.control).offset();
        uint32_t bin_destination[kMessagesBinsMax], bin_fill[kMessagesBinsMax];
        // A bin's body (its last message's; a body is sent at once): L1 address, bytes.
        uint32_t bin_body[kMessagesBinsMax], bin_body_bytes[kMessagesBinsMax];
        uint32_t bin_first[kMessagesBinsMax];  // the wall clock at its first message
        for (uint32_t bin = 0; bin < kMessagesBinsMax; bin++) {
            bin_destination[bin] = kMessagesEnd;
            bin_fill[bin] = 0;
            bin_body_bytes[bin] = 0;
        }
        uint32_t taken = 0, sent = 0, held = 0, where = 0, flag = 0, lost_to = 0;
        uint32_t bodies = 0;
        messages[KM_MESSAGES_SENT / 4] = 0;
        writer_flags[kMessagesFlagBodies] = 0;
        writer_flags[1] = 0;
        writer_flags[0] = run;
        // A core reserves on the counter tiles: it waits until each shows READY.
        const uint32_t counter_tiles = messages_arguments.counters;
        const MessageRanks message_ranks(
            islands_at(arguments.control), command, counter_tiles
        );
        const uint32_t island_ranks = islands_at(arguments.control).ranks();
#if defined(KM_DELIVER)
        const uint32_t island_here = islands_at(arguments.control).here();
#endif
        if (
            counter_tiles == 0 || counter_tiles > KM_COUNTERS_MAX ||
            message_ranks.ranks_here() == 0
        ) {
            where = kWhereMessagesCounter;
        } else if (!message_ranks.reserved(offset + arguments.me)) {
            for (
                uint32_t counter_tile = 0; counter_tile < counter_tiles && where == 0;
                counter_tile++
            ) {
                if (!messages_ready(
                    writer,
                    message_ranks.tile_core(counter_tile),
                    KM_TAILS_WRITER,
                    run
                )) {
                    where = kWhereMessagesCounter;
                }
            }
        }

        // Position `position` of `destination`'s queue: its NoC address, and how many
        // messages from it are contiguous (a DRAM page; an X280 rank's whole queue).
        auto queue_noc = [&](uint32_t destination, uint32_t position) -> uint64_t {
            const uint32_t at = position & (messages_arguments.cap - 1);
            if (destination >= KM_X280_RANK_CORE(0)) {
                return (
                    get_noc_addr(
                        messages_arguments.x280_bank & 0xFFFF,
                        messages_arguments.x280_bank >> 16,
                        messages_arguments.x280_queue +
                            (
                                (destination - KM_X280_RANK_CORE(0)) *
                                    messages_arguments.cap +
                                at
                            ) *
                                messages_arguments.message_bytes
                    )
                );
            }
            return (
                queue_addresses.get_noc_addr(
                    messages_page(destination, position, messages_arguments.cap),
                    at % kMessagesPageMessages * messages_arguments.message_bytes
                )
            );
        };
        auto contiguous = [&](uint32_t destination, uint32_t position) {
            const uint32_t at = position & (messages_arguments.cap - 1);
            return (
                destination >= KM_X280_RANK_CORE(0)
                    ? messages_arguments.cap - at
                    : kMessagesPageMessages - at % kMessagesPageMessages
            );
        };

        // The bin written but not committed (kMessagesNoBin: none); its commit goes
        // out while the next bin's reservation travels.
        uint32_t pending = kMessagesNoBin, pending_mark = 0;
        uint64_t pending_noc = 0;
        KM_MSTAMP(const uint32_t stamps = messages_arguments.stamps);
        KM_MSTAMP(uint32_t bin_taken[kMessagesBinsMax] = {});  // sums of take times
        KM_MSTAMP(uint32_t pending_at = 0);
        KM_MSTAMP(uint32_t empty_from = 0);
        auto writes_landed = [&] {
            KM_MSTAMP(const uint32_t asked = wall());
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                where = kWhereMessagesFlush;
                return false;
            }
            KM_MSTAMP(messages_stamp(stamps, kStampLanded, wall() - asked));
            return true;
        };
        // The bin's messages are in their queue: counted sent, its body free, the bin
        // empty.
        auto sent_bin = [&](uint32_t bin) {
            sent += bin_fill[bin];
            messages[KM_MESSAGES_SENT / 4] = sent;
            if (bin_body_bytes[bin] != 0) {  // compute may change the body's L1 again
                bin_body_bytes[bin] = 0;
                writer_flags[kMessagesFlagBodies] = ++bodies;
            }
            bin_fill[bin] = 0;
            bin_destination[bin] = kMessagesEnd;
            held--;
        };
        // The pending bin's commit: one 16-byte write of its first message's mark,
        // once its messages have landed.
        auto commit = [&] {
            const uint32_t base = (
                store + pending * kMessagesPacket * messages_arguments.message_bytes
            );
            if (!writes_landed()) {
                return false;
            }
            KM_MSTAMP(messages_stamp(stamps, kStampPending, wall() - pending_at));
            *message_mark_half(base) = pending_mark;
            invalidate_l1_cache();
            noc_async_write(base, pending_noc, 16);
            return true;
        };
        // The pending bin committed and landed (with nothing else in flight).
        auto settle = [&] {
            if (pending == kMessagesNoBin) {
                return true;
            }
            if (!commit() || !writes_landed()) {
                return false;
            }
            sent_bin(pending);
            pending = kMessagesNoBin;
            return true;
        };

        // Bin `bin` to its destination: reserve; the pending bin's commit while the
        // answer travels; then this bin's messages go out with the first one's mark 0.
        auto flush = [&](uint32_t bin) {
            const uint32_t bin_messages = bin_fill[bin];
            KM_MSTAMP(messages_stamp(stamps, kStampFlushes, 1));
            KM_MSTAMP(messages_stamp(
                stamps, kStampBinWait, bin_messages * wall() - bin_taken[bin]
            ));
            KM_MSTAMP(bin_taken[bin] = 0);
            if (
                (bin_destination[bin] & kMessagesAway) != 0 &&
                !away.takes(bin_destination[bin])
            ) {
                where = kWhereMessagesBadDest;  // another island's rank and no link
                                                // here
                lost_to = bin_destination[bin];
                return false;
            }
            if (away.takes(bin_destination[bin])) {
                if (!settle()) {
                    return false;
                }
                where = (
                    away.put(
                        writer,
                        command,
                        bin_destination[bin],
                        store + bin * kMessagesPacket * messages_arguments.message_bytes,
                        messages_arguments.message_bytes,
                        bin_messages,
                        bin_body[bin],
                        bin_body_bytes[bin]
                    )
                );
                if (where != 0) {
                    lost_to = bin_destination[bin];  // the rank, so its island
                    return false;
                }
                sent_bin(bin);  // its body has left this L1 too
                return true;
            }
            returned[0] = kMessagesEnd;
            invalidate_l1_cache();  // a fence: the store lands before the answer can
            KM_MSTAMP(const uint32_t reserved_at = wall());
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index,
                write_at_cmd_buf,
                writer.core_noc(
                    message_ranks.counter_core(offset + bin_destination[bin]),
                    arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE +
                        16 * bin_destination[bin]
                ),
                NOC_UNICAST_WRITE_VC,
                bin_messages,
                31,
                false,
                false,
                returned_address
            );
            atomic_return_reset(noc_index, write_at_cmd_buf);
            if (pending != kMessagesNoBin && !commit()) {
                return false;
            }
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            })) {
                where = kWhereMessagesAtomic;
                return false;
            }
            // On the card the answer can stay away although its count moved.
            if (!noc_timeout(writer, [&] {
                invalidate_l1_cache();
                return returned[0] != kMessagesEnd;
            })) {
                where = kWhereMessagesLost;
                lost_to = bin_destination[bin];
                return false;
            }
            KM_MSTAMP(messages_stamp(stamps, kStampReserve, wall() - reserved_at));
            if (
                pending != kMessagesNoBin
            ) {  // the commit landed: nothing else is in flight
                if (!writes_landed()) {
                    return false;
                }
                sent_bin(pending);
                pending = kMessagesNoBin;
            }
            // Commit: the messages go out with the first one's mark zero, then after
            // their barrier one 16-byte write sets it (each word 0 holds its handler).
            const uint32_t first_position = returned[0];
            const uint32_t base = (
                store + bin * kMessagesPacket * messages_arguments.message_bytes
            );
            const MessageMarks marks(first_position, shift, bin_messages);
            if (bin_body_bytes[bin] != 0) {  // before the barrier the commit waits for
                noc_async_write(
                    bin_body[bin],
                    body_addresses.get_noc_addr(messages_body_page(
                        bin_destination[bin],
                        first_position + bin_messages - 1,
                        messages_arguments.cap
                    )),
                    messages_body_lines(bin_body_bytes[bin])
                );
            }
            *message_mark_half(base) = 0;
            for (uint32_t i = 1; i < bin_messages; i++) {
                *message_mark_half(base + i * messages_arguments.message_bytes) = marks(
                    i
                );
            }
            invalidate_l1_cache();  // a fence: the NoC reads the stored marks
            for (uint32_t i = 0; i < bin_messages;) {
                const uint32_t position = first_position + i;
                uint32_t chunk = contiguous(bin_destination[bin], position);
                chunk = chunk < bin_messages - i ? chunk : bin_messages - i;
                noc_async_write(
                    base + i * messages_arguments.message_bytes,
                    queue_noc(bin_destination[bin], position),
                    chunk * messages_arguments.message_bytes
                );
                i += chunk;
            }
            pending = bin;
            KM_MSTAMP(pending_at = wall());
            pending_mark = marks(0);
            pending_noc = queue_noc(bin_destination[bin], first_position);
            return true;
        };
        // A direct line from the outgoing page `source` into its receiver's slot: the
        // line with mark 0, then after its barrier its first 16 bytes with the mark.
        const DirectArea direct{messages_arguments.direct};
        auto direct_write = [&](uint32_t source, uint32_t word) {
            const uint32_t entry = word & kMessagesOutEntry;
            const uint32_t out = direct.at(kDirectOut)[entry];
            const uint64_t slot = (
                writer.core_noc(
                    message_ranks.flat_of(out & kDirectRank) - offset,
                    messages_arguments.direct + ((out >> kDirectFarShift) << 4)
                )
            );
            volatile tt_l1_ptr uint32_t* written = direct.at(kDirectWritten, entry);
#if defined(KM_DELIVER)
            const uint32_t to = message_ranks.flat_of(out & kDirectRank);
            if (to - message_ranks.low() >= message_ranks.ranks_here()) {
                const uint32_t body_word = (
                    messages_arguments.body_bytes != 0
                        ? l1(source)[kMessagesBodyWord]
                        : 0
                );
                if (messages_arguments.body_bytes != 0) {
                    l1(source)[kMessagesBodyWord] = body_word & kMessagesOutBodyBytes;
                }
                *message_handler_half(source) = (word >> kMessagesOutHandlerShift) &
                    0xFF;
                if (direct_far_line(
                    writer,
                    to,
                    messages_arguments.direct + ((out >> kDirectFarShift) << 4),
                    source,
                    messages_arguments.message_bytes,
                    body_word,
                    messages_mark(*written, 0)
                ) != kGaveNone) {
                    where = kWhereFar;
                    return false;
                }
                *written = *written + 1;
                sent++;
                messages[KM_MESSAGES_SENT / 4] = sent;
                if ((body_word & kMessagesOutBodyBytes) != 0) {
                    writer_flags[kMessagesFlagBodies] = ++bodies;
                }
                return true;
            }
#endif
            uint32_t body_bytes = 0;
            if (messages_arguments.body_bytes != 0) {  // the body, then the line
                const uint32_t body_word = l1(source)[kMessagesBodyWord];
                body_bytes = body_word & kMessagesOutBodyBytes;
                l1(source)[kMessagesBodyWord] = body_bytes;
                if (body_bytes != 0) {
                    noc_async_write(
                        (body_word >> kMessagesOutBodyShift) << 6,
                        slot + messages_arguments.message_bytes,
                        messages_body_lines(body_bytes)
                    );
                }
            }
            *message_mark_half(source) = 0;
            *message_handler_half(source) = (word >> kMessagesOutHandlerShift) & 0xFF;
            invalidate_l1_cache();  // a fence: the NoC reads the stored words
            noc_async_write(source, slot, messages_arguments.message_bytes);
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                where = kWhereMessagesFlush;
                return false;
            }
            *message_mark_half(source) = messages_mark(*written, 0);
            invalidate_l1_cache();
            noc_async_write(source, slot, 16);
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                where = kWhereMessagesFlush;
                return false;
            }
            *written = *written + 1;
            sent++;
            messages[KM_MESSAGES_SENT / 4] = sent;
            if (body_bytes != 0) {  // compute may change the body's L1 again
                writer_flags[kMessagesFlagBodies] = ++bodies;
            }
            return true;
        };
        auto flush_all = [&] {
            for (uint32_t bin = 0; bin < bins; bin++) {
                if (bin_fill[bin] != 0 && bin != pending && !flush(bin)) {
                    return false;
                }
            }
            return settle();
        };
        // The first bin whose first message is messages_arguments.age cycles old,
        // sent while no line comes, as a full bin is.
        auto flush_aged = [&] {
            const uint32_t now = wall();
            for (uint32_t bin = 0; bin < bins; bin++) {
                if (
                    bin_fill[bin] != 0 && bin != pending &&
                    now - bin_first[bin] >= messages_arguments.age
                ) {
                    KM_MSTAMP(messages_stamp(stamps, kStampAged, 1));
                    return flush(bin);
                }
            }
            return true;
        };

        bool ended = false;
        const uint64_t limit = (
            arguments.idle_limit == 0
                ? ~0ull
                : static_cast<uint64_t>(arguments.idle_limit) << 10
        );
        uint64_t quiet_from = wall64();
        uint32_t empty = 0;  // polls of an empty outgoing buffer since the last message
        for (uint32_t loop = 0; where == 0 && !ended; loop++) {
            invalidate_l1_cache();
            if (writer.stop()) {
                where = kWhereMessagesStopped;
                break;
            }
            if (!cb_pages_available_at_front(kCbMessagesOut, 1)) {
                KM_MSTAMP(if (empty_from == 0) { empty_from = wall() | 1; });
                if (pending != kMessagesNoBin && ++empty >= kMessagesSettlePolls) {
                    KM_MSTAMP(messages_stamp(stamps, kStampSettled, 1));
                    settle();  // no next bin came to overlap it with
                    continue;
                }
                if (
                    held != 0 && compute_flags[0] == run &&
                    (compute_flags[1] & kMessagesWaiting)
                ) {
                    KM_MSTAMP(messages_stamp(stamps, kStampComputeWaits, 1));
                    flush_all();  // compute waits: nothing stays in a bin
                    continue;
                }
                // only while a bin besides the pending one waits: the pending one's
                // commit counts these polls
                const uint32_t waiting = held - (pending != kMessagesNoBin ? 1 : 0);
                if (waiting != 0 && messages_arguments.age != 0 && !flush_aged()) {
                    continue;  // where says why
                }
                const uint32_t next_flag = (
                    held == 0 ? (taken | kMessagesWaiting) : taken
                );
                if (next_flag != flag) {
                    flag = next_flag;
                    writer_flags[1] = next_flag;
                }
                if ((loop & 15) == 15 && wall64() - quiet_from > limit) {
                    where = kWhereMessagesOutbox;
                }
                continue;
            }
            quiet_from = wall64();
            empty = 0;
            KM_MSTAMP(if (empty_from != 0) {
                messages_stamp(stamps, kStampOutEmpty, wall() - empty_from);
                empty_from = 0;
            });
            cb_wait_front(kCbMessagesOut, 1);
            volatile tt_l1_ptr uint32_t* outgoing = l1(get_read_ptr(kCbMessagesOut));
            const uint32_t outgoing_word = outgoing[0];
            if (outgoing_word == kMessagesEnd) {
                cb_pop_front(kCbMessagesOut, 1);
                ended = flush_all();
                break;
            }
            if (outgoing_word & kMessagesOutDirect) {
                if (!direct_write(get_read_ptr(kCbMessagesOut), outgoing_word)) {
                    break;
                }
                cb_pop_front(kCbMessagesOut, 1);
                taken++;
                flag = taken;
                writer_flags[1] = taken;
                continue;
            }
            // the rank's core of the card (an X280 rank's: its local core); a rank on
            // another island is kMessagesAway above its flat rank, for its link
            const uint32_t flat = outgoing_word & kMessagesOutDestination;
#if defined(KM_DELIVER)  // an X280 rank: island 0's, delivered from the others
            const bool x280 = flat >= KM_X280_FLAT(0);
            uint32_t destination = (
                x280 ? KM_X280_RANK_CORE(flat - KM_X280_FLAT(0)) : flat - offset
            );
            if (
                island_ranks != 0 &&
                (x280 ? island_here != 0 : flat - island_first >= island_ranks)
            ) {
                destination = kMessagesAway | flat;
            }
#else
            const bool x280 = (
                messages_arguments.x280 != 0 &&
                flat - island_first >= KM_X280_RANK_CORE(0)
            );
            uint32_t destination = x280 ? flat - island_first : flat - offset;
            if (island_ranks != 0 && flat - island_first >= island_ranks && !x280) {
                destination = kMessagesAway | flat;
            }
#endif
            const uint32_t bin = destination % bins;
            if (bin == pending && !settle()) {  // its store is still being read
                break;
            }
            const bool evicted = (
                bin_destination[bin] != destination && bin_fill[bin] != 0
            );
            KM_MSTAMP(if (evicted) { messages_stamp(stamps, kStampEvicted, 1); });
            if (evicted && (!flush(bin) || !settle())) {  // the store takes it next
                break;
            }
            if (bin_fill[bin] == 0) {
                bin_destination[bin] = destination;
                bin_first[bin] = wall();
                held++;
            }
            const uint32_t slot_address = (
                store +
                (bin * kMessagesPacket + bin_fill[bin]) *
                    messages_arguments.message_bytes
            );
            volatile tt_l1_ptr uint32_t* slot_words = l1(slot_address);
            // The handler; the flush adds the mark.
            *message_handler_half(slot_address) = outgoing_word >>
                kMessagesOutHandlerShift;
            for (uint32_t word = 1; word < words; word++) {
                slot_words[word] = outgoing[word];
            }
            uint32_t body_word = 0;
            if (messages_arguments.body_bytes != 0) {  // the line keeps the bytes only
                body_word = outgoing[kMessagesBodyWord];
                slot_words[kMessagesBodyWord] = body_word & kMessagesOutBodyBytes;
                bin_body[bin] = (body_word >> kMessagesOutBodyShift) << 6;
                bin_body_bytes[bin] = body_word & kMessagesOutBodyBytes;
            }
            bin_fill[bin]++;
            KM_MSTAMP(messages_stamp(stamps, kStampTaken, 1));
            KM_MSTAMP(bin_taken[bin] += wall());
            cb_pop_front(kCbMessagesOut, 1);
            taken++;
            flag = taken;
            writer_flags[1] = taken;
            const bool full = (
                bin_fill[bin] == kMessagesPacket || (body_word & kMessagesOutBodyBytes)
            );
            KM_MSTAMP(if (full) { messages_stamp(stamps, kStampFull, 1); });
            if (full && !flush(bin)) {
                break;
            }
        }
        if (where != 0) {
            messages_gave_up(arguments.control, where, taken, run, lost_to);
            writer.gave_up(kStWriter, kGaveStop, kWhereData, slot(command), run);
            return false;
        }
        return ended;
    }

}  // namespace km

#else
namespace km {

    // A message's body in L1: its address (64-byte aligned) and bytes; 0 bytes: none.
    struct Body {
        uint32_t address = 0;
        uint32_t bytes = 0;

        volatile tt_l1_ptr uint32_t* words() const {
            return l1(address);
        }

        explicit operator bool() const {
            return bytes != 0;
        }
    };

    // A packet of messages in L1 (the input buffer's page), valid until done().
    struct Packet {
        uint32_t count;
        uint32_t base;  // byte address of message 0
        uint32_t message_bytes;
        uint32_t* current = nullptr;  // Messages: the handler of the message taken last
#if defined(KM_MESSAGE_PARTS)
        // A handler's group: its messages' places in the packet (the header's order).
        const volatile tt_l1_ptr uint8_t* order = nullptr;
#endif
        uint32_t bodies = 0;  // byte address of message 0's body; a page of body_bytes
        uint32_t body_bytes = 0;

        explicit operator bool() const {
            return count != 0;
        }

        // Message `index`; its handler becomes the one forward() and send() keep.
        const volatile tt_l1_ptr uint32_t* operator[](uint32_t index) const {
#if defined(KM_MESSAGE_PARTS)
            index = order != nullptr ? order[index] : index;
#endif
            const volatile tt_l1_ptr uint32_t* message = (
                l1(base + index * message_bytes)
            );
            if (current != nullptr) {
                *current = messages_handler(message[0]);
            }
            return message;
        }

        // The body of message `index` (empty without one), in L1 until done().
        Body body(uint32_t index) const {
#if defined(KM_MESSAGE_PARTS)
            index = order != nullptr ? order[index] : index;
#endif
            if (body_bytes == 0) {
                return {};
            }
            return {
                bodies + index * body_bytes,
                l1(base + index * message_bytes)[kMessagesBodyWord]
            };
        }
    };

    // Compute's side of a dynamic run. The body runs on all three threads alike; pack
    // writes the outgoing messages, unpack takes and frees the packets.
    class Messages {
    public:
        template <typename ComputeType>
        Messages(const ComputeType& compute, const km_command_t& command) :
            control_(get_arg_val<uint32_t>(kCArgControl)),
            idle_(get_arg_val<uint32_t>(kCArgIdle)),
            wait_(get_arg_val<uint32_t>(kCArgWait)),
            run_(command.sequence),
            ranks_(
                islands_at(get_arg_val<uint32_t>(kCArgControl)),
                command,
                compute.arg(kMessagesCArgCounters)
            ),
            count_(ranks_.count()),
            x280_(compute.arg(kMessagesCArgX280)),
            rank_(ranks_.rank_of(compute.me())),
            message_bytes_(compute.arg(kMessagesCArgMessage) & 0xFFFF),
            body_bytes_(compute.arg(kMessagesCArgMessage) >> kMessagesCArgBodyShift),
            quota_(compute.arg(kMessagesCArgQuota)),
            handler_(km::handler(command)),
#if defined(KM_MESSAGE_PARTS)
            command_(command),
#endif
            direct_{rank_ >= count_ ? 0 : compute.arg(kMessagesCArgDirect)},
            seeded_(rank_ >= count_) {
            UNPACK(({
                volatile tt_l1_ptr uint32_t* messages = (
                    l1(control_ + KM_CONTROL_MESSAGES)
                );
                messages[KM_MESSAGES_CONSUMED / 4] = 0;
                messages[KM_MESSAGES_QUOTA / 4] = quota_;
                messages[(KM_MESSAGES_FLAGS + kMessagesFlagCompute) / 4 + 1] = 0;
                messages[(KM_MESSAGES_FLAGS + kMessagesFlagCompute) / 4] = run_;
            }));
        }

        // True until seeded(): the run's own sends go here, once per run and core.
        bool seeding() const {
            return !seeded_;
        }

        void seeded() {
            seeded_ = true;
        }

        // The run's cores (the group without its counter tiles), and this core's.
        uint32_t count() const {
            return count_;
        }

        // The run's X280 ranks: ranks count() to count() + x280() - 1.
        uint32_t x280() const {
            return x280_;
        }

        uint32_t rank() const {
            return rank_;
        }

        // A counter tile: no message comes here; next() returns only the run's end.
        bool counter() const {
            return rank_ >= count_;
        }

        uint32_t quota() const {
            return quota_;
        }

        // The handler of the message taken last from a packet; the RUN command's
        // before.
        uint32_t handler() const {
            return handler_;
        }

#if defined(KM_MESSAGE_PARTS)
        // The run's RUN command (its argument words are the program's).
        const km_command_t& command() const {
            return command_;
        }

        // The program's compute argument `index` (after the runtime's dynamic ones).
        uint32_t arg(uint32_t index) const {
            return get_arg_val<uint32_t>(kComputeArgs + kMessagesCArgs + index);
        }
#endif

        // The next packet; empty when the run ended, the stop flag was set or a wait
        // gave up. Unpack waits and tells math and pack.
        Packet next() {
            seeded_ = true;
            uint32_t packet_count = 0, address = 0;
            PACK((ckernel::mailbox_write(
                ckernel::ThreadId::UnpackThreadId,
                pushed_ | (failed_ ? kMessagesWaiting : 0)
            )));
            UNPACK(({
                const uint32_t pack_word = (
                    ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)
                );
                volatile tt_l1_ptr uint32_t* messages = (
                    l1(control_ + KM_CONTROL_MESSAGES)
                );
                volatile tt_l1_ptr uint32_t* compute_flags = (
                    messages + (KM_MESSAGES_FLAGS + kMessagesFlagCompute) / 4
                );
                if (!(pack_word & kMessagesWaiting) && !failed_) {
                    pushed_ = pack_word;
                    compute_flags[1] = pushed_ | kMessagesWaiting;
                    volatile tt_l1_ptr uint32_t* got = (
                        get_cb_tiles_received_ptr(kCbMessagesIn)
                    );
                    const uint32_t code = poll_stop(
                        l1(control_ + KM_CONTROL_STOP),
                        [&] {
                            return (
                                static_cast<uint16_t>(
                                    ckernel::reg_read(
                                        static_cast<uint32_t>(
                                            reinterpret_cast<uintptr_t>(got)
                                        )
                                    ) -
                                    get_local_cb_interface(kCbMessagesIn).tiles_acked
                                ) >= 1
                            );
                        },
                        idle_,
                        0
                    );
                    if (code == kGaveNone) {
                        cb_wait_front(kCbMessagesIn, 1);
                        invalidate_l1_cache();
                        address = ckernel::get_tile_l1_byte_address(kCbMessagesIn, 0);
                        packet_count = l1(address)[0];
                        if (
                            packet_count == kMessagesEnd || packet_count == 0 ||
                            packet_count > kMessagesPacket
                        ) {
                            cb_pop_front(kCbMessagesIn, 1);
                            packet_count = 0;
                        } else {
                            compute_flags[1] = pushed_;
                        }
                    } else {
                        messages_gave_up(
                            control_,
                            code == kGaveStop
                                ? kWhereMessagesStopped
                                : kWhereMessagesNext,
                            0,
                            run_
                        );
                        command_gave_up(
                            control_, kStCompute, code, kWhereCToken, 0, run_
                        );
                    }
                }
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, packet_count);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, packet_count);
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, address);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, address);
            }));
            MATH(({
                packet_count = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                address = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            }));
            PACK(({
                packet_count = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                address = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            }));
            if (packet_count == 0) {
                PACK((end_marker()));
                return {0, 0, message_bytes_};
            }
            Packet packet{
                packet_count, address + kMessagesHeader, message_bytes_, &handler_
            };
            if (body_bytes_ != 0) {
                packet.bodies = packet.base + kMessagesPacket * message_bytes_;
                packet.body_bytes = body_bytes_;
                packet_low_ = packet.bodies;
                packet_high_ = packet.bodies + kMessagesBodyPacket * body_bytes_;
            }
            return packet;
        }

        // The message (message_bytes / 4 words, word 0 the runtime's) goes on at rank
        // `destination` of the group, keeping the handler of the message being handled.
        void forward(uint32_t destination, const uint32_t* message) {
            forward(destination, message, handler_);
        }

        // The same, handled by `handler` there (one of the program's dynamic handlers).
        void forward(uint32_t destination, const uint32_t* message, uint32_t handler) {
            if (!reaches(destination)) {
                return;
            }
            PACK(({
                if (!failed_) {
                    push_out(
                        (
                            destination < count_ ? ranks_.flat_of(destination)
#if defined(KM_DELIVER)  // island 0's X280 ranks, from every island
                                                 : KM_X280_FLAT(destination - count_)
#else
                                                 : islands_at(control_).first() +
                                    KM_X280_RANK_CORE(destination - count_)
#endif
                        ) | (handler & 0xFF) << kMessagesOutHandlerShift,
                        message,
                        0
                    );
                }
            }));
        }

        // The same with `body` (L1, 64-byte aligned, at most the program's body bytes)
        // beside it. A body of the packet in hand: done() waits for it to leave.
        void forward(
            uint32_t destination,
            const uint32_t* message,
            uint32_t handler,
            const Body& body
        ) {
            if (!reaches(destination) || !body_fits(body)) {
                return;
            }
            // Another buffer: the call returns once the body has left it.
            const bool held = (
                body.address >= packet_low_ && body.address < packet_high_
            );
            // Every thread's writes to the body are in L1 before pack hands it over.
            UNPACK(({
                __sync_synchronize();
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 1);
            }));
            MATH(({
                __sync_synchronize();
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 1);
            }));
            PACK(({
                (void)ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                (void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
                if (!failed_) {
                    push_out(
                        ranks_.flat_of(destination) |
                            (handler & 0xFF) << kMessagesOutHandlerShift,
                        message,
                        body.bytes | (body.address >> 6) << kMessagesOutBodyShift
                    );
                    bodies_++;
                    if (!held) {
                        wait_bodies();
                    }
                }
                ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, 1);
            }));
            UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
            MATH(((void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
        }

        void forward(uint32_t destination, const uint32_t* message, const Body& body) {
            forward(destination, message, handler_, body);
        }

        // The message goes on at rank `destination` through this core's slot there, a
        // declared pair: false, and nothing sent, while the slot holds the last one.
        bool direct(uint32_t destination, const uint32_t* message) {
            return direct(destination, message, handler_);
        }

        // The same, handled by `handler` there.
        bool direct(uint32_t destination, const uint32_t* message, uint32_t handler) {
            uint32_t went = 0;
            PACK(({
                went = direct_out(destination, message, handler, 0);
                ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, went);
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, went);
            }));
            UNPACK((went = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
            MATH((went = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
            return went != 0;
        }

        // The same with `body` beside it (as forward with a body): true once the body
        // has left; false, and nothing sent, while the slot is busy.
        bool direct(
            uint32_t destination,
            const uint32_t* message,
            uint32_t handler,
            const Body& body
        ) {
            if (!body_fits(body)) {
                return false;
            }
            uint32_t went = 0;
            UNPACK(({
                __sync_synchronize();
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 1);
            }));
            MATH(({
                __sync_synchronize();
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 1);
            }));
            PACK(({
                (void)ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                (void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
                went = (
                    direct_out(
                        destination,
                        message,
                        handler,
                        body.bytes | (body.address >> 6) << kMessagesOutBodyShift
                    )
                );
                if (went != 0) {
                    wait_body();
                }
                ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, went);
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, went);
            }));
            UNPACK((went = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
            MATH((went = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
            return went != 0;
        }

        bool direct(uint32_t destination, const uint32_t* message, const Body& body) {
            return direct(destination, message, handler_, body);
        }

        // The message ends here: one send returns to this core's quota.
        void consume() {
            quota_++;
        }

        // A new message at rank `destination`; false when this core has no quota left.
        bool send(uint32_t destination, const uint32_t* message) {
            return send(destination, message, handler_);
        }

        // The same, handled by `handler` there.
        bool send(uint32_t destination, const uint32_t* message, uint32_t handler) {
            if (quota_ == 0) {
                return false;
            }
            quota_--;
            forward(destination, message, handler);
            return true;
        }

        // The same with `body` beside it (as forward with a body).
        bool send(
            uint32_t destination,
            const uint32_t* message,
            uint32_t handler,
            const Body& body
        ) {
            if (quota_ == 0) {
                return false;
            }
            quota_--;
            forward(destination, message, handler, body);
            return true;
        }

        bool send(uint32_t destination, const uint32_t* message, const Body& body) {
            return send(destination, message, handler_, body);
        }

        // The packet's page is free once its bodies sent on have left; CONSUMED counts
        // its messages.
        void done(const Packet& packet) {
            MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1)));
            PACK(({
                if (body_bytes_ != 0 && !failed_) {
                    wait_bodies();
                }
                ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            }));
            UNPACK(({
                (void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
                (void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);
                cb_pop_front(kCbMessagesIn, 1);
                consumed_ += packet.count;
                volatile tt_l1_ptr uint32_t* messages = (
                    l1(control_ + KM_CONTROL_MESSAGES)
                );
                messages[KM_MESSAGES_CONSUMED / 4] = consumed_;
                messages[KM_MESSAGES_QUOTA / 4] = quota_;
            }));
        }

        // Waits for the stop flag (the idle limit its timeout): a body that stalls.
        bool wait_stop() const {
            return poll_stop(
                l1(control_ + KM_CONTROL_STOP),
                [] {
                    return false;
                },
                idle_,
                0
            ) == kGaveStop;
        }

    private:
        // Pack: a direct line to `destination` with `body` (bytes, address / 64 above;
        // 0: none) when its slot is free. 1: it went; 0: busy, or the run given up.
        uint32_t direct_out(
            uint32_t destination,
            const uint32_t* message,
            uint32_t handler,
            uint32_t body
        ) {
            const uint32_t entry = (
                destination < count_ ? direct_.find(destination) : kDirectNone
            );
            if (failed_) {
                return 0;
            }
            if (entry == kDirectNone) {
                failed_ = true;
                messages_gave_up(control_, kWhereMessagesDirect, destination, run_);
                command_gave_up(
                    control_, kStCompute, kGaveStop, kWhereCOutput, 0, run_
                );
                return 0;
            }
            volatile tt_l1_ptr uint32_t* sends = direct_.at(kDirectSent, entry);
            if (direct_.at(kDirectCredit, 4 * entry)[0] != *sends) {
                return 0;
            }
            push_out(
                kMessagesOutDirect | entry |
                    (handler & 0xFF) << kMessagesOutHandlerShift,
                message,
                body
            );
            if (failed_) {
                return 0;
            }
            *sends = *sends + 1;
            return 1;
        }

        // False, and the run given up (reason body), when `body` cannot be sent.
        bool body_fits(const Body& body) {
            if (
                body.bytes != 0 && body.bytes <= body_bytes_ &&
                body.address % 64 == 0 &&
                (body.address >> 6) >> (32 - kMessagesOutBodyShift) == 0
            ) {
                return true;
            }
            failed_ = true;
            PACK((messages_gave_up(control_, kWhereMessagesBody, body.bytes, run_)));
            PACK((
                command_gave_up(control_, kStCompute, kGaveStop, kWhereCOutput, 0, run_)
            ));
            return false;
        }

        // False, and the run given up, when rank `destination` is no core of the group
        // and no X280 rank.
        bool reaches(uint32_t destination) {
            if (destination < count_ + x280_) {
                return true;
            }
            failed_ = true;
            PACK(
                (messages_gave_up(control_, kWhereMessagesBadDest, destination, run_))
            );
            PACK((
                command_gave_up(control_, kStCompute, kGaveStop, kWhereCOutput, 0, run_)
            ));
            return false;
        }

        // Pack: one outgoing message, word 0 the destination core and the handler; in a
        // program with bodies word 1 is `body` (bytes, address / 64 above; 0: none).
        void push_out(uint32_t destination, const uint32_t* message, uint32_t body) {
            if (!reserve()) {
                return;
            }
            volatile tt_l1_ptr uint32_t* outgoing = (
                l1(get_local_cb_interface(kCbMessagesOut).fifo_wr_ptr << cb_addr_shift)
            );
            for (uint32_t word = 1; word < message_bytes_ / 4; word++) {
                outgoing[word] = message[word];
            }
            if (body_bytes_ != 0) {
                outgoing[kMessagesBodyWord] = body;
            }
            outgoing[0] = destination;
            __sync_synchronize();  // the message is in L1 before the push
            cb_push_back(kCbMessagesOut, 1);
            pushed_++;
        }

        // Pack: one more body pushed, and the wait until every one has been written.
        void wait_body() {
            bodies_++;
            wait_bodies();
        }

        // Pack: waits until the writer has written every body pushed so far.
        void wait_bodies() {
            volatile tt_l1_ptr uint32_t* writer_flags = (
                l1(
                    control_ + KM_CONTROL_MESSAGES + KM_MESSAGES_FLAGS + kMessagesFlagWriter
                )
            );
            const uint32_t code = poll_stop(
                l1(control_ + KM_CONTROL_STOP),
                [&] {
                    return (
                        writer_flags[0] == run_ &&
                        static_cast<int32_t>(
                            writer_flags[kMessagesFlagBodies] - bodies_
                        ) >= 0
                    );
                },
                wait_,
                0
            );
            if (code != kGaveNone) {
                failed_ = true;
                messages_gave_up(control_, kWhereMessagesBody, bodies_, run_);
                command_gave_up(control_, kStCompute, code, kWhereCOutput, 0, run_);
            }
        }

        // Pack: the writer leaves its loop on this message.
        void end_marker() {
            if (!reserve()) {
                return;
            }
            l1(
                get_local_cb_interface(kCbMessagesOut).fifo_wr_ptr << cb_addr_shift
            )[0] = kMessagesEnd;
            __sync_synchronize();  // the message is in L1 before the push
            cb_push_back(kCbMessagesOut, 1);
        }

        // Pack: a free outgoing page (with a timeout, observes the stop flag).
        bool reserve() {
            volatile tt_reg_ptr uint32_t* acked = (
                get_cb_tiles_acked_ptr(kCbMessagesOut)
            );
            const uint32_t code = poll_stop(
                l1(control_ + KM_CONTROL_STOP),
                [&] {
                    const uint16_t used = (
                        static_cast<uint16_t>(
                            get_local_cb_interface(kCbMessagesOut).tiles_received -
                            ckernel::reg_read(
                                static_cast<uint32_t>(
                                    reinterpret_cast<uintptr_t>(acked)
                                )
                            )
                        )
                    );
                    return (
                        get_local_cb_interface(kCbMessagesOut).fifo_num_pages - used >=
                        1
                    );
                },
                wait_,
                0
            );
            if (code != kGaveNone) {
                failed_ = true;
                messages_gave_up(control_, kWhereMessagesOutPage, pushed_, run_);
                command_gave_up(control_, kStCompute, code, kWhereCOutput, 0, run_);
                return false;
            }
            cb_reserve_back(kCbMessagesOut, 1);
            return true;
        }

        uint32_t control_;
        uint32_t idle_;
        uint32_t wait_;
        uint32_t run_;
        MessageRanks ranks_;
        uint32_t count_;
        uint32_t x280_;
        uint32_t rank_;
        uint32_t message_bytes_;
        uint32_t body_bytes_;
        uint32_t quota_;
        uint32_t handler_;
#if defined(KM_MESSAGE_PARTS)
        const km_command_t& command_;
#endif
        DirectArea direct_;
        bool seeded_;
        uint32_t pushed_ = 0;
        uint32_t consumed_ = 0;
        uint32_t bodies_ = 0;                        // pack: bodies pushed in this run
        uint32_t packet_low_ = 0, packet_high_ = 0;  // the packet in hand's bodies
        bool failed_ = false;
    };

#if defined(KM_MESSAGE_PARTS)
    // What the runtime's compute hands a handler's part: km_message(messages, packet).
    struct MessageCall {
        Messages* messages;
        Packet* packet;
    };
#endif

}  // namespace km
#endif
