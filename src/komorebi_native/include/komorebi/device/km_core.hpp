// What every persistent kernel of a tile uses: runtime-owned buffers, the state words,
// the start-up arguments, the clock, and the wait (timeout, stop flag).
#pragma once

#include <stdint.h>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "km_command.hpp"
#include "km_island.hpp"

namespace km {

    // The island area of this core's control block: every kernel's argument 0 is the
    // control block's address.
    template <int>
    inline const volatile uint32_t* island_area() {
        return (
            reinterpret_cast<const volatile uint32_t*>(
                get_arg_val<uint32_t>(0) + KM_CONTROL_ISLANDS
            )
        );
    }

    // Runtime-owned circular buffers: the tokens the reader hands to compute and
    // writer, and the per-command trace (only when tracing).
    constexpr uint32_t kCbTokenCompute = 29;
    constexpr uint32_t kCbTokenWriter = 30;
    constexpr uint32_t kCbTrace = 31;
    constexpr uint32_t kTokenBytes = 64;
    // Word of each 16-byte DONE slot (word 0 counts the cores): the last sequence
    // whose compute body returned in that slot, written by unpack, read by the writer.
    constexpr uint32_t kDoneCompute = 1;

    // State words (32-bit, from KM_CONTROL_STATE). A gave-up word is code | where << 8
    // | slot << 16; its sequence number is in the next word. Counters saturate.
    constexpr uint32_t kStLast = 0;  // KM_RING_SLOTS words: last sequence consumed
    constexpr uint32_t kStReported = KM_RING_SLOTS;  // last sequence reported, per slot
    constexpr uint32_t kStReader = 2 * KM_RING_SLOTS;       // reader gave-up word
    constexpr uint32_t kStReaderSequence = kStReader + 1;   // its sequence number
    constexpr uint32_t kStWriter = kStReader + 2;           // writer gave-up word
    constexpr uint32_t kStWriterSequence = kStReader + 3;   // its sequence number
    constexpr uint32_t kStCompute = kStReader + 4;          // compute gave-up word
    constexpr uint32_t kStComputeSequence = kStReader + 5;  // its sequence number
    constexpr uint32_t kStRuns = kStReader + 6;             // commands the reader took
    constexpr uint32_t kStPasses = kStReader + 7;  // ring passes in next(), all calls
    constexpr uint32_t kStNextCycles = kStReader + 8;  // cycles in next(), all, / 16
    constexpr uint32_t kStCompleted = kStReader + 9;   // commands the writer finished
    constexpr uint32_t kStSeen = kStReader + 10;       // commands compute took
    constexpr uint32_t kStWrong = kStReader + 11;      // unpack, math, pack: bad copies
    constexpr uint32_t kStEnds = kStReader + 12;       // reader, writer, compute: kEnd*
    constexpr uint32_t kStMagic = kStReader + 13;      // kStateMagic | tag << 8 | me
    constexpr uint32_t kStSlotWrong = kStReader + 14;  // not run: slot not theirs
    constexpr uint32_t kStTraceDropped = kStReader + 15;  // past the trace's capacity
    constexpr uint32_t kStateWords = kStReader + 16;
    static_assert(kStateWords == KM_STATE_WORDS, "the control block's state words");
    constexpr uint32_t kStateMagic = 0x4B000000;

    // Gave-up codes, where codes and how a kernel left.
    constexpr uint32_t kGaveNone = 0;
    constexpr uint32_t kGaveStop = 1;     // the stop flag was set
    constexpr uint32_t kGaveLimit = 2;    // the wait timed out
    constexpr uint32_t kGaveBadItem = 3;  // a multi-page item would pass the buffer end
    constexpr uint32_t kGaveBadBroadcast = 4;  // a broadcast's arguments do not fit the command
    constexpr uint32_t kGaveBadTransfer = 6;  // a transfer's bounds, or PCIe routing left
    // The dispatcher gave the command up at its deadline (its abort word): the kernels
    // leave its body, clear what it left and take the next command.
    constexpr uint32_t kGaveDeadline = 7;
    constexpr uint32_t kWhereCheckin = 1;   // leader collecting start-up check-ins
    constexpr uint32_t kWhereToken = 2;     // reader waiting for free token pages
    constexpr uint32_t kWhereNext = 3;      // reader waiting for a command
    constexpr uint32_t kWhereStall = 4;     // a handler waiting for the stop flag
    constexpr uint32_t kWhereComplete = 5;  // leader waiting for its cores
    constexpr uint32_t kWhereWriter = 6;    // writer waiting for a token
    constexpr uint32_t kWhereData = 7;      // a handler's timed buffer wait
    constexpr uint32_t kWhereCToken = 8;    // compute (unpack) waiting for a token
    constexpr uint32_t kWhereCInput = 9;    // compute (unpack) waiting for input pages
    constexpr uint32_t kWhereCOutput = 10;  // compute (pack) waiting for output space
    constexpr uint32_t kWhereCDone = 11;    // writer: compute's and reader's bodies
    constexpr uint32_t kEndExit = 1;        // left on an EXIT command
    constexpr uint32_t kEndGaveUp = 2;      // left after its own wait gave up
    constexpr uint32_t kEndAbort = 3;       // left on the EXIT handed after a give-up
    constexpr uint32_t kEndReader = 0;      // bytes of kStEnds; kStWrong: unpack, ...
    constexpr uint32_t kEndWriter = 1;
    constexpr uint32_t kEndCompute = 2;

    // Where codes of broadcast (km_broadcast.hpp).
    constexpr uint32_t kWhereBroadcastBegin = 12;  // broadcast: arguments or buffer at a command
    constexpr uint32_t kWhereBroadcastDone = 13;  // broadcast sender: DONE markers, its NoC barriers
    constexpr uint32_t kWhereBroadcastValid = 14;  // broadcast receiver waiting for VALID
    constexpr uint32_t kWhereBroadcastFree = 15;  // broadcast: its compute's pops, its last markers

    // Where codes of a port's wait (reader) and of an emit (writer): the gave-up word's
    // top byte is the port; the port keeps the count it wanted (KM_PORT_WANT).
    constexpr uint32_t kWherePort = 17;
    constexpr uint32_t kWhereEmit = 18;

    // Where codes of a dynamic run (km_messages.hpp), KM_MESSAGES_GAVE word 0; word 1
    // position, word 2 run, word 3 the word 0 at a gap or a lost answer's destination.
    constexpr uint32_t kWhereMessagesGap = 19;  // reserved message: not landed in time
    constexpr uint32_t kWhereMessagesOverflow = 20;  // IN_TAIL - IN_HEAD passed the capacity
    constexpr uint32_t kWhereMessagesAtomic = 21;  // a reservation was not answered in time
    constexpr uint32_t kWhereMessagesFlush = 22;                // writer: message writes did not leave
    constexpr uint32_t kWhereMessagesRead = 23;  // reader: message reads did not land
    constexpr uint32_t kWhereMessagesInPage = 24;  // reader: no free packet page
    constexpr uint32_t kWhereMessagesLine = 25;  // the status line write did not leave
    constexpr uint32_t kWhereMessagesBadDest = 26;  // forward or send outside the group
    constexpr uint32_t kWhereMessagesNext = 27;     // compute's wait for a packet
    constexpr uint32_t kWhereMessagesOutPage = 28;  // compute's wait for an outgoing page
    constexpr uint32_t kWhereMessagesOutbox = 29;  // writer: wait for an outgoing message
    constexpr uint32_t kWhereMessagesStopped = 30;  // the stop flag ended the run
    constexpr uint32_t kWhereMessagesAlign = 31;  // pages or messages not 64-B aligned
    constexpr uint32_t kWhereMessagesLost = 32;   // a reservation's answer never landed
    constexpr uint32_t kWhereMessagesFill = 33;   // the start-up zero fill of the queue
    constexpr uint32_t kWhereMessagesFull = 34;  // positions reached 2^30: reload the program
    constexpr uint32_t kWhereMessagesCounter = 35;  // counter tile not ready, or no cores
    // Where codes of an overlay (km_overlay.hpp): the gave-up word's top half is the
    // overlay's number. kGaveOverlay: unknown, no part here, too large, no free slot.
    constexpr uint32_t kGaveOverlay = 5;
    constexpr uint32_t kWhereOverlayFind = 36;  // the table or the slots
    constexpr uint32_t kWhereOverlayLoad = 37;  // the read of a part from DRAM
    constexpr uint32_t kWhereOverlayMeet = 38;  // the reader and a follower meet for a load
    constexpr uint32_t kWhereProgramSwitch = 39;  // a load of a program: its buffers
    static_assert(
        kWhereOverlayFind == kWhereMessagesCounter + 1 && kWhereOverlayMeet < 40,
        "overlay where codes: 36 to 38, after the dynamic ones"
    );
    // A dynamic run whose handlers are overlay parts: a message names a handler (or a
    // program) that is not the program's; a handler's part did not load.
    constexpr uint32_t kWhereMessagesHandler = 40;
    constexpr uint32_t kWhereMessagesLoad = 41;
    constexpr uint32_t kWhereProgramStart = 42;  // compute: no start-up for the program
    // A transfer to or from the host channel (km_transfer.hpp), on the reader.
    constexpr uint32_t kWhereToHost = 43;
    constexpr uint32_t kWhereFromHost = 44;
    constexpr uint32_t kWhereCopy = 45;  // a copy between DRAM pages
    // The runtime's own NoC barriers, the compute threads' words to each other, and the
    // clear after a command was given up (a consumer waits for the buffer's producer).
    constexpr uint32_t kWhereNocFlush = 46;
    constexpr uint32_t kWhereCMailbox = 47;  // math or pack waiting for unpack's word
    constexpr uint32_t kWhereCConfirm = 48;  // unpack waiting for math's and pack's
    constexpr uint32_t kWhereClear = 49;
    // A message's body: too long, not aligned, or not written in time.
    constexpr uint32_t kWhereMessagesBody = 50;
    // Pages across a link: a bound, a read, the router's room, its teardown, a count.
    constexpr uint32_t kWherePages = 51;

    // What Core::flushed waits for: reads landed, writes sent from L1, writes or
    // atomics acknowledged.
    constexpr uint32_t kFlushReads = 0;
    constexpr uint32_t kFlushSent = 1;
    constexpr uint32_t kFlushWrites = 2;
    constexpr uint32_t kFlushAtomics = 3;

    // Live words (from KM_CONTROL_LIVE), each with one writer: kernel r (a role) writes
    // word kLv* + r. The dispatcher's inspect reads words 0 to 31 during the run.
    constexpr uint32_t kRoleReader = 0;
    constexpr uint32_t kRoleWriter = 1;
    constexpr uint32_t kRoleUnpack = 2;
    constexpr uint32_t kRoleMath = 3;
    constexpr uint32_t kRolePack = 4;
    constexpr uint32_t kRoles = 5;
    constexpr uint32_t kLvWhere = 0;      // + r: the runtime wait it is in; 0: none
    constexpr uint32_t kLvSequence = 5;   // + r: the command it runs; 0: none
    constexpr uint32_t kLvSlot = 10;      // + r: that command's slot
    constexpr uint32_t kLvFinished = 15;  // + r: commands it has left, a count
    constexpr uint32_t kLvTrace = 20;     // the trace's L1 address (reader); 0: none
    constexpr uint32_t kLvTraceCap = 21;  // sequences the trace holds
    constexpr uint32_t kLvGivenUp = 22;   // commands given up, writer's count
    constexpr uint32_t kLvProduces = 24;  // + p: buffers producer p fills, a bit each
    constexpr uint32_t kLvEnds = 32;      // pages pushed at a command's end (ends())
    constexpr uint32_t kLvWords = 128;
    constexpr uint32_t kProducerReader = 0;  // producers p: the reader, pack, writer
    constexpr uint32_t kProducerPack = 1;
    constexpr uint32_t kProducerWriter = 2;
    constexpr uint32_t kProducers = 3;
    constexpr uint32_t kClearBuffers = 29;  // buffers 0 to 28: the runtime's are above
    static_assert(
        kLvWords * 4 == KM_LIVE_BYTES && kLvEnds * 4 == KM_LIVE_READ &&
            kLvEnds + kProducers * 2 * 16 <= kLvWords && kLvTrace == KM_LIVE_TRACE &&
            kLvSequence == KM_LIVE_SEQUENCE && kRoles == KM_LIVE_ROLES &&
            kLvTraceCap == KM_LIVE_TRACE_CAP,
        "live words"
    );
    static_assert(
        kWhereMessagesGap == kWhereEmit + 1 &&
            kWhereMessagesCounter == kWhereMessagesGap + 16 &&
            kWhereMessagesCounter < 256,
        "dynamic where codes: 19 to 35, after the runtime's, one byte"
    );

    // Slots of KM_CONTROL_OUT (16 bytes each), the sources of small NoC writes:
    // check-in, READY, hello (reader); acknowledgement (writer).
    constexpr uint32_t kOutCheckin = 0;
    constexpr uint32_t kOutReady = 1;
    constexpr uint32_t kOutHello = 2;
    constexpr uint32_t kOutAck = 3;
    constexpr uint32_t kOutSlots = (KM_CONTROL_CORES - KM_CONTROL_OUT) / 16;
    static_assert(
        KM_CONTROL_OUT % 16 == 0 && kOutAck < kOutSlots, "KM_CONTROL_OUT slots"
    );
    static_assert(KM_ACK_STRIDE == 16, "ack words are 16-byte slots");

    // Start-up arguments of the reader and the writer: runtime args 0 to kArgs - 1. The
    // program's own arguments follow. The compute kernel gets the kCArg* words.
    struct Args {
        uint32_t control;  // control block address, the same on every core
        uint32_t me;       // this core's index in the core order
        uint32_t cores;    // DRAM address of the core table (one page)
        uint32_t
            dispatcher_coordinates;  // NoC x | y << 16 of the dispatcher's ack words
        uint32_t ack_base;    // ack word of slot j at ack_base + j * KM_ACK_STRIDE
        uint32_t mailbox;     // hello address at the dispatcher; 0: send none
        uint32_t leader;      // start-up group: the leader's core index
        uint32_t rank;        // this core's check-in slot at the leader
        uint32_t peers;       // leader: cores that check in
        uint32_t ready_slot;  // leader: its READY word, at ready_base + slot * 16
        uint32_t ready_base;  // the dispatcher's READY words (not the ack words)
        uint32_t tag;         // per-process random tag, 16 bits
        uint32_t workers;     // cores 0 to workers - 1 run commands (an EXIT's group)
        uint32_t depth;       // ring slots scanned, 1 to KM_RING_SLOTS
        uint32_t row;         // cores per grid row (completion); 0: one level always
        uint32_t wait_limit;  // timeout of a wait for another core, 1024-cycle units
        uint32_t idle_limit;  // timeout of a wait for the next command; 0: none
        uint32_t trace;       // DRAM address of the trace, a page per core; 0: none
        uint32_t trace_cap;   // sequences traced: 1 to trace_cap
    };

    constexpr uint32_t kArgs = sizeof(Args) / 4;
    constexpr uint32_t kCArgControl = 0;  // compute: control block address
    constexpr uint32_t kCArgMe = 1;       // this core's index
    constexpr uint32_t kCArgWait = 2;     // wait_limit, as for reader and writer
    constexpr uint32_t kCArgIdle = 3;     // idle_limit
    constexpr uint32_t kComputeArgs = 4;

    // Trace command per sequence number: 4 words of wall-clock cycles (low 32 bits).
    constexpr uint32_t kTraceSeen = 0;  // reader took the command
    constexpr uint32_t kTraceDone = 1;  // writer: outputs written
    constexpr uint32_t kTraceSent = 2;  // writer: completion sent (leader: pushed)
    constexpr uint32_t kTraceExtra = 3;  // reader: ring passes it took; with a timestamp
                          // in a9: cycles from the timestamp to taking it
    constexpr uint32_t kTraceWords = 4;
    static_assert(kTraceWords * 4 == KM_TRACE_ENTRY, "a trace entry");

    // A 64-bit count from a live high word read before and after the low word; a carry
    // between them retries, and a second one returns the carry's own instant.
    template <typename High, typename Low>
    inline uint64_t wall64_of(High high, Low low) {
        for (uint32_t i = 0; i < 2; i++) {
            const uint32_t high_word = high();
            const uint32_t low_word = low();
            if (high() == high_word) {
                return static_cast<uint64_t>(high_word) << 32 | low_word;
            }
        }
        return static_cast<uint64_t>(high()) << 32;
    }

    // Wall clock. Not the latched high word: every core of the tile latches it with
    // its own low-word reads, so it can be torn by 2^32 cycles at a carry.
    inline uint64_t wall64() {
        return wall64_of(
            [] {
                return (
                    *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                        RISCV_DEBUG_REG_WALL_CLOCK_1
                    )
                );
            },
            [] {
                return (
                    *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                        RISCV_DEBUG_REG_WALL_CLOCK_L
                    )
                );
            }
        );
    }

    inline uint32_t wall() {
        return (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            )
        );
    }

    inline volatile tt_l1_ptr uint32_t* l1(uint32_t address) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    }

    // Adds `amount` to a counter that stops at 0xFFFFFFFF.
    inline void add_sat(volatile tt_l1_ptr uint32_t* counter, uint32_t amount) {
        const uint32_t sum = *counter + amount;
        *counter = sum < amount ? 0xFFFFFFFFu : sum;
    }

    // Byte `b` of state word `word` (each byte has one writer).
    inline void put_byte(
        volatile tt_l1_ptr uint32_t* state, uint32_t word, uint32_t byte, uint32_t value
    ) {
        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(state + word)[byte] = value;
    }

    // True if `pages` pages from `ptr` (this side's fifo_rd_ptr or fifo_wr_ptr, in its
    // units) end at or before the limit of buffer cb: an item is one contiguous region.
    inline bool item_fits(uint32_t cb, uint32_t pointer, uint32_t pages) {
        const LocalCBInterface& circular_buffer = get_local_cb_interface(cb);
        return (
            pointer + pages * circular_buffer.fifo_page_size <=
            circular_buffer.fifo_limit
        );
    }

    // ready() until it holds (kGaveNone); kGaveLimit after `limit` 1024-cycle units;
    // kGaveStop `grace` units after *stop; kGaveDeadline once *abort is `sequence`.
    template <typename Ready>
    inline uint32_t poll_stop(
        volatile tt_l1_ptr uint32_t* stop,
        Ready ready,
        uint32_t limit,
        uint32_t grace,
        volatile tt_l1_ptr uint32_t* abort = nullptr,
        uint32_t sequence = 0
    ) {
        uint64_t end = (
            limit == 0 ? ~0ull : wall64() + (static_cast<uint64_t>(limit) << 10)
        );
        bool stopped = false;
        for (uint32_t polls = 0;; polls++) {
            invalidate_l1_cache();
            if (ready()) {
                return kGaveNone;
            }
            if (abort != nullptr && *abort == sequence) {
                return kGaveDeadline;
            }
            if (!stopped && *stop != 0) {
                if (grace == 0) {
                    return kGaveStop;
                }
                stopped = true;
                const uint64_t grace_end = (
                    wall64() + (static_cast<uint64_t>(grace) << 10)
                );
                end = grace_end < end ? grace_end : end;
            }
            if ((polls & 15) == 15 && wall64() > end) {
                return stopped ? kGaveStop : kGaveLimit;
            }
        }
    }

    // poll_stop for kernel `role` in the runtime wait `where` (0: not shown live), on
    // its command's abort word when `abortable`; its live where word shows the wait.
    template <typename Ready>
    inline uint32_t poll_live(
        uint32_t control,
        uint32_t role,
        uint32_t where,
        bool abortable,
        Ready ready,
        uint32_t limit,
        uint32_t grace
    ) {
        volatile tt_l1_ptr uint32_t* live = l1(control + KM_CONTROL_LIVE);
        const uint32_t sequence = live[kLvSequence + role];
        volatile tt_l1_ptr uint32_t* abort = (
            abortable && sequence != 0
                ? l1(control + KM_CONTROL_ABORT) + live[kLvSlot + role] % KM_RING_SLOTS
                : nullptr
        );
        const uint32_t before = live[kLvWhere + role];
        if (where != 0) {
            live[kLvWhere + role] = where;
        }
        const uint32_t code = (
            poll_stop(
                l1(control + KM_CONTROL_STOP), ready, limit, grace, abort, sequence
            )
        );
        if (where != 0) {
            live[kLvWhere + role] = before;
        }
        return code;
    }

    // A give-up into state word pair `word`: a deadline's per command, the run goes on;
    // any other unless one came first, setting the tile's stop flag (all kernels end).
    inline void command_gave_up(
        uint32_t control,
        uint32_t word,
        uint32_t code,
        uint32_t where,
        uint32_t slot,
        uint32_t sequence
    ) {
        volatile tt_l1_ptr uint32_t* state = l1(control + KM_CONTROL_STATE);
        const bool after_deadline = (state[word] & 0xFF) == kGaveDeadline;
        if (code == kGaveDeadline) {
            if (state[word] == 0 || (after_deadline && state[word + 1] != sequence)) {
                state[word] = code | where << 8 | slot << 16;
                state[word + 1] = sequence;
            }
            return;
        }
        if (state[word] == 0 || after_deadline) {
            state[word] = code | where << 8 | slot << 16;
            state[word + 1] = sequence;
        }
        *l1(control + KM_CONTROL_STOP) = 1;
    }

    // Kernel `role`'s own live words, at its start: no wait, no command, none left.
    inline void clear_live(uint32_t control, uint32_t role) {
        volatile tt_l1_ptr uint32_t* live = l1(control + KM_CONTROL_LIVE);
        live[kLvWhere + role] = 0;
        live[kLvSequence + role] = 0;
        live[kLvSlot + role] = 0;
        live[kLvFinished + role] = 0;
    }

    // Whether the dispatcher gave up `sequence` in `slot` (its abort word holds it).
    inline bool given_up(uint32_t control, uint32_t slot, uint32_t sequence) {
        return (
            sequence != 0 &&
            l1(control + KM_CONTROL_ABORT)[slot % KM_RING_SLOTS] == sequence
        );
    }

    // Producer p's page counts (16 bits a buffer) at the end of its command number
    // `index`: two tables, by the count's parity (a producer runs one command ahead).
    inline volatile tt_l1_ptr uint16_t* ends(
        uint32_t control, uint32_t producer, uint32_t index
    ) {
        return (
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                l1(control + KM_CONTROL_LIVE) + kLvEnds +
                (producer * 2 + (index & 1)) * 16
            )
        );
    }

    // The role that fills buffer cb (published by the producers' timed waits), or
    // kRoles if none of them does.
    inline uint32_t producer_of(uint32_t control, uint32_t cb, uint32_t& producer) {
        volatile tt_l1_ptr uint32_t* live = l1(control + KM_CONTROL_LIVE);
        for (producer = 0; producer < kProducers; producer++) {
            if ((live[kLvProduces + producer] >> cb) & 1) {
                return (
                    producer == kProducerReader
                        ? kRoleReader
                        : producer == kProducerPack
                            ? kRolePack
                            : kRoleWriter
                );
            }
        }
        return kRoles;
    }

    // A command's end on kernel `role`, command number `index`: a producer's counts,
    // then the number of commands it has left; the command is no longer its own.
    template <typename Count>
    inline void end_command(
        uint32_t control,
        uint32_t role,
        uint32_t index,
        uint32_t produces,
        uint32_t producer,
        Count count
    ) {
        if (produces != 0) {
            volatile tt_l1_ptr uint16_t* table = ends(control, producer, index);
            for (uint32_t cb = 0; cb < kClearBuffers; cb++) {
                if ((produces >> cb) & 1) {
                    table[cb] = static_cast<uint16_t>(count(cb));
                }
            }
        }
        volatile tt_l1_ptr uint32_t* live = l1(control + KM_CONTROL_LIVE);
        live[kLvFinished + role] = index;
        live[kLvSequence + role] = 0;
    }

    // After command `index` was given up, per buffer this consumer drains: the producer
    // ends it, then its pages are popped. kGaveNone, or why it gave up.
    template <typename Acked, typename Pop>
    inline uint32_t clear_buffers(
        uint32_t control,
        uint32_t role,
        uint32_t index,
        uint32_t consumes,
        uint32_t limit,
        Acked acked,
        Pop pop
    ) {
        volatile tt_l1_ptr uint32_t* live = l1(control + KM_CONTROL_LIVE);
        for (uint32_t cb = 0; cb < kClearBuffers; cb++) {
            uint32_t producer = 0;
            const uint32_t from = (
                (consumes >> cb) & 1 ? producer_of(control, cb, producer) : kRoles
            );
            if (from == kRoles) {
                continue;
            }
            const uint32_t code = poll_live(
                control,
                role,
                kWhereClear,
                false,
                [&] {
                    return static_cast<int32_t>(live[kLvFinished + from] - index) >= 0;
                },
                limit,
                0
            );
            if (code != kGaveNone) {
                return code;
            }
            const uint16_t left = (
                static_cast<uint16_t>(ends(control, producer, index)[cb] - acked(cb))
            );
            const LocalCBInterface& buffer = get_local_cb_interface(cb);
            if (left > buffer.fifo_size / buffer.fifo_page_size) {
                return kGaveBadItem;
            }
            for (uint32_t page = 0; page < left; page++) {
                pop(cb);
            }
        }
        return kGaveNone;
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // Word 0 of slot `slot` of KM_CONTROL_OUT in the control block at `control`.
    inline volatile tt_l1_ptr uint32_t* out_slot(uint32_t control, uint32_t slot) {
        return l1(control + KM_CONTROL_OUT + 16 * slot);
    }

    // A small NoC write: Bytes from slot From of KM_CONTROL_OUT to slot `target_slot`
    // above the 16-byte aligned `base`, so both ends share their low four address bits.
    template <uint32_t From, uint32_t Bytes>
    inline void small_write(uint32_t control, uint64_t base, uint32_t target_slot) {
        static_assert(From < kOutSlots && Bytes > 0 && Bytes <= 16, "one slot");
        noc_async_write(
            control + KM_CONTROL_OUT + 16 * From, base + 16 * target_slot, Bytes
        );
    }

    // A command's block as read into L1: its words and its bytes (0: none).
    struct Block {
        const volatile tt_l1_ptr uint32_t* words;
        uint32_t bytes;
    };

    // A data-movement kernel's view of the control block and of its start-up arguments,
    // and of the command it runs (its role's live words, its buffers for a clear).
    class Core {
    public:
        explicit Core(uint32_t role) : role_(role) {
            uint32_t* words = &args_.control;
            for (uint32_t index = 0; index < kArgs; index++) {
                words[index] = get_arg_val<uint32_t>(index);
            }
            args_.tag &= 0xFFFFu;
            if (args_.depth == 0 || args_.depth > KM_RING_SLOTS) {
                args_.depth = KM_RING_SLOTS;
            }
            invalidate_l1_cache();
            island_first_ = islands_at(args_.control).first();
        }

        const Args& args() const {
            return args_;
        }

        // This core's flat rank: its island's first rank plus its index (args().me).
        uint32_t me() const {
            return island_first_ + args_.me;
        }

        // This core's name in `command`'s group (km::name_in): its rank in the group
        // is this less the group's first, in a group of link cores too.
        uint32_t name(const km_command_t& command) const {
            return name_in(islands_at(args_.control), args_.me, command);
        }

        // This island's core index of flat rank `rank` (for the core table).
        uint32_t core_of(uint32_t rank) const {
            return rank - island_first_;
        }

        // The program's argument `index` (after the start-up arguments).
        uint32_t arg(uint32_t index) const {
            return get_arg_val<uint32_t>(kArgs + index);
        }

        volatile tt_l1_ptr uint32_t* control(uint32_t offset) const {
            return l1(args_.control + offset);
        }

        volatile tt_l1_ptr uint32_t* state() const {
            return control(KM_CONTROL_STATE);
        }

        // Nonzero stop flag; the caller has invalidated the L1 cache.
        bool stop() const {
            return *control(KM_CONTROL_STOP) != 0;
        }

        // NoC address of `address` on core `index` of the core table.
        uint64_t core_noc(uint32_t index, uint32_t address) const {
            const uint32_t coordinates = control(KM_CONTROL_CORES)[index];
            return get_noc_addr(coordinates & 0xFFFFu, coordinates >> 16, address);
        }

        uint64_t disp_noc(uint32_t address) const {
            return (
                get_noc_addr(
                    args_.dispatcher_coordinates & 0xFFFFu,
                    args_.dispatcher_coordinates >> 16,
                    address
                )
            );
        }

        uint32_t role() const {
            return role_;
        }

        // poll_stop on this tile's stop flag and on the abort word of the command this
        // kernel runs; poll_at also shows the runtime wait `where` in the live words.
        template <typename Ready>
        uint32_t poll(Ready ready, uint32_t limit, uint32_t grace = 0) const {
            return poll_live(args_.control, role_, 0, true, ready, limit, grace);
        }

        template <typename Ready>
        uint32_t poll_at(
            uint32_t where, Ready ready, uint32_t limit, uint32_t grace = 0
        ) const {
            return poll_live(args_.control, role_, where, true, ready, limit, grace);
        }

        // poll_at on the stop flag alone: a wait the dispatcher's abort does not end.
        template <typename Ready>
        uint32_t poll_stop_at(uint32_t where, Ready ready, uint32_t limit) const {
            return poll_live(args_.control, role_, where, false, ready, limit, 0);
        }

        // The command this kernel runs (its slot and sequence in the live words).
        void begin(const km_command_t& command) const {
            control(KM_CONTROL_LIVE)[kLvSlot + role_] = slot(command);
            control(KM_CONTROL_LIVE)[kLvSequence + role_] = command.sequence;
        }

        uint32_t running_slot() const {
            return control(KM_CONTROL_LIVE)[kLvSlot + role_];
        }

        uint32_t running_sequence() const {
            return control(KM_CONTROL_LIVE)[kLvSequence + role_];
        }

        // Whether the dispatcher gave the command up (its deadline passed).
        bool given_up(const km_command_t& command) const {
            invalidate_l1_cache();
            return km::given_up(args_.control, slot(command), command.sequence);
        }

        // Timed NoC barrier of this kernel (kFlush*), polled with the stop flag; a stop
        // gives it one more wait limit. False: gave up in kWhereNocFlush (recorded).
        bool flushed(uint32_t what, uint32_t word) const {
            const uint32_t code = poll_live(
                args_.control,
                role_,
                kWhereNocFlush,
                false,
                [&] {
                    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
                        return (
                            what == kFlushReads
                                ? ncrisc_dynamic_noc_reads_flushed(noc_index)
                                : what == kFlushSent
                                    ? ncrisc_dynamic_noc_nonposted_writes_sent(
                                        noc_index
                                    )
                                    : what == kFlushAtomics
                                        ? ncrisc_dynamic_noc_nonposted_atomics_flushed(
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
                                    : what == kFlushAtomics
                                        ? ncrisc_noc_nonposted_atomics_flushed(
                                            noc_index
                                        )
                                        : ncrisc_noc_nonposted_writes_flushed(noc_index)
                        );
                    }
                },
                args_.wait_limit,
                args_.wait_limit
            );
            if (code != kGaveNone) {
                gave_up(word, code, kWhereNocFlush, running_slot(), running_sequence());
                return false;
            }
            invalidate_l1_cache();
            return true;
        }

        // A buffer this kernel fills (a producer: the reader) or drains (a consumer)
        // through the timed waits: what a clear after a give-up looks at.
        void produce(uint32_t cb) const {
            if (cb < kClearBuffers && !((produces_ >> cb) & 1)) {
                produces_ |= 1u << cb;
                control(KM_CONTROL_LIVE)[kLvProduces + producer()] = produces_;
            }
        }

        // This kernel's producer number: the reader's, or the writer's (a broadcast).
        uint32_t producer() const {
            return role_ == kRoleWriter ? kProducerWriter : kProducerReader;
        }

        void consume(uint32_t cb) const {
            consumes_ |= cb < kClearBuffers ? 1u << cb : 0;
        }

        // The command's end: the producer's counts, the commands left; after a give-up,
        // the buffers cleared and rewound. False: the clear gave up (the tile stops).
        bool finish(const km_command_t& command, uint32_t word) {
            index_++;
            end_command(
                args_.control, role_, index_, produces_, producer(), [](uint32_t cb) {
                    return *get_cb_tiles_received_ptr(cb);
                }
            );
            if (!given_up(command)) {
                return true;
            }
            if (consumes_ == 0) {
                return true;
            }
            const uint32_t code = clear_buffers(
                args_.control,
                role_,
                index_,
                consumes_,
                args_.wait_limit,
                [](uint32_t cb) {
                    return static_cast<uint16_t>(*get_cb_tiles_acked_ptr(cb));
                },
                [](uint32_t cb) {
                    cb_pop_front(cb, 1);
                }
            );
            if (code != kGaveNone) {
                gave_up(word, code, kWhereClear, slot(command), command.sequence);
                return false;
            }
            return true;
        }

        // The command's block in copy area `area` (0 reader, 1 writer), read once per
        // command; 0 bytes for none, one over KM_BLOCK_BYTES, a bad address or bank.
        Block block_in(const km_command_t& command, uint32_t area) const {
            const uint32_t bytes = command.a8 & KM_BLOCK_BYTES_MASK;
            const uint32_t field = command.a8 >> KM_BLOCK_BANK_SHIFT;
            if (
                bytes == 0 || bytes > KM_BLOCK_BYTES || command.a7 % 64 != 0 ||
                field > KM_BLOCK_BANKS
            ) {
                return {nullptr, 0};
            }
            const uint32_t at = (
                args_.control + KM_CONTROL_BLOCK + area * KM_BLOCK_AREA_BYTES
            );
            volatile tt_l1_ptr uint32_t* line = l1(at);
            if (line[0] != command.sequence) {
                const uint32_t bank = (
                    field != 0 ? field - 1 : args_.me % KM_BLOCK_BANKS
                );
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, command.a7),
                    at + 64,
                    (bytes + 63) / 64 * 64
                );
                if (!flushed(kFlushReads, area == 0 ? kStReader : kStWriter)) {
                    return {nullptr, 0};
                }
                line[0] = command.sequence;
            }
            return {line + 16, bytes};
        }

        // command_gave_up for this tile (kStReader or kStWriter).
        void gave_up(
            uint32_t word,
            uint32_t code,
            uint32_t where,
            uint32_t slot,
            uint32_t sequence
        ) const {
            command_gave_up(args_.control, word, code, where, slot, sequence);
        }

    private:
        Args args_;
        uint32_t island_first_;
        uint32_t role_;
        uint32_t index_ = 0;             // commands this kernel took (EXIT aside)
        mutable uint32_t produces_ = 0;  // bit cb: filled through a timed wait
        mutable uint32_t consumes_ = 0;  // bit cb: drained through a timed wait
    };

}  // namespace km
#endif
