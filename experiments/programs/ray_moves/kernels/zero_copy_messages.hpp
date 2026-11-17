// The page-sized messages with pages no core copies: forward_page() hands the writer a
// page where it landed, sent on with one NoC write; the reader checks each on arrival.
#pragma once

#include <stdint.h>

#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/core/km_command.hpp"

namespace km {

    // Arguments of a dynamic program: reader and writer program arguments 0 to
    // kMessagesArgs - 1 (the program's own follow), compute program arguments 0 and 1.
    struct MessagesArgs {
        uint32_t handler;  // the dynamic handler
        uint32_t
            queue;  // DRAM address of the queues, kMessagesPageMessages messages a page
        uint32_t
            cap;  // messages per queue: a power of two, at least kMessagesPageMessages
        uint32_t message_bytes;  // a multiple of 64, at most kMessagesMessageMax
        uint32_t lines;     // status lines at the submitter: core me at lines + 16 me
        uint32_t idle;      // cycles the idle condition holds before a line says so
        uint32_t bins;      // writer bins, 1 to kMessagesBinsMax
        uint32_t counters;  // K: the group's last K cores are counter tiles
    };

    constexpr uint32_t kMessagesArgs = sizeof(MessagesArgs) / 4;
    constexpr uint32_t kMessagesCArgMessage = 0;  // compute: message bytes
    constexpr uint32_t kMessagesCArgMaxLive = 1;  // compute: max_live, the run's ranks'
                                                  // sends (quota: an equal share)
    constexpr uint32_t kMessagesCArgCounters = 2;  // compute: K, as MessagesArgs::counters
    constexpr uint32_t kMessagesCArgs = 3;

    // Buffers: packets reader -> compute, outgoing messages compute -> writer, bins.
    constexpr uint32_t kCbMessagesIn = 26;
    constexpr uint32_t kCbMessagesOut = 27;
    constexpr uint32_t kCbMessagesBins = 28;
    constexpr uint32_t kMessagesPacket = 4;  // messages of a packet and of a bin
    constexpr uint32_t kMessagesPageMessages = 32;           // messages of a DRAM page of a queue
    constexpr uint32_t kMessagesHeader = 64;  // packet header (n, first position): DRAM
                                              // reads land 64-byte aligned
    constexpr uint32_t kMessagesMessageMax = 4096;  // bytes
    constexpr uint32_t kMessagesBinsMax = 16;
    constexpr uint32_t kMessagesEnd = 0xFFFFFFFF;  // packet n / outgoing dest: the run ended
    constexpr uint32_t kMessagesWaiting = 1u << 31;  // idle word bit (compute, writer)
    constexpr uint32_t kMessagesInPlace = 1u
        << 30;  // outgoing destination: word 1 is the page

    // Idle words (KM_MESSAGES_FLAGS): word 0 the run, word 1 the state.
    constexpr uint32_t kMessagesFlagReader = 0x00;
    constexpr uint32_t kMessagesFlagCompute = 0x10;  // waiting in next() | messages pushed
    constexpr uint32_t kMessagesFlagWriter = 0x20;  // every bin sent | messages taken

    // Positions per core and session. Word 0 of the message at `position` once valid; 0
    // is empty. Queues start zero-filled and positions never restart: no stale mark.
    constexpr uint32_t kMessagesPositionMax = 1u << 30;

    inline uint32_t messages_mark(uint32_t position) {
        return position + 1;
    }

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

    // The counter tile (a core index) that holds the tail of core `core`; `offset`:
    // the flat rank of the island's core 0 (the command's first is a flat rank).
    inline uint32_t messages_counter(
        const km_command_t& command,
        uint32_t counter_tiles,
        uint32_t core,
        uint32_t offset
    ) {
        const uint32_t low = first(command) - offset;
        return (
            low + messages_ranks(command, counter_tiles) + (core - low) % counter_tiles
        );
    }

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
                get_noc_addr(reader.args().me * pages + page, queue_addresses),
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

    // Generic reader: this core's queue in order, to compute in packets; idle and busy
    // lines to the submitter; leaves on the run's end, the stop flag or an error.
    template <typename ReaderType, typename Check>
    inline void messages_reader(
        ReaderType& reader, const km_command_t& command, bool filled, Check check
    ) {
        const Args& arguments = reader.args();
        const MessagesArgs messages_arguments = messages_args(reader);
        const uint32_t offset = islands_at(arguments.control).offset();  // flat - core
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
        const uint32_t in_pages = get_local_cb_interface(kCbMessagesIn).fifo_num_pages;
        const uint64_t line = (
            reader.disp_noc(messages_arguments.lines + 16 * arguments.me)
        );
        const uint64_t wait = static_cast<uint64_t>(arguments.wait_limit) << 10;
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
            where = kWhereMessagesFull;  // marks stay unique below 2^31 positions
        }
        // A counter tile sets its cores' tails to their heads, then shows READY; a
        // core waits for its counter tile's READY before it reads its tail there.
        const uint32_t counter_tiles = messages_arguments.counters;
        const uint32_t ranks = messages_ranks(command, counter_tiles);
        const bool counter = arguments.me - first(command) >= ranks;
        if (counter_tiles == 0 || counter_tiles > KM_COUNTERS_MAX || ranks == 0) {
            where = where != 0 ? where : kWhereMessagesCounter;
        } else if (where == 0 && counter) {
            *tail = head;  // nobody sends here: the loop sees an empty queue
            uint32_t served = 0;
            for (
                uint32_t rank = arguments.me - first(command) - ranks; rank < ranks;
                rank += counter_tiles
            ) {
                const uint32_t core = first(command) + rank;
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
                reader,
                messages_counter(command, counter_tiles, arguments.me, offset),
                KM_TAILS_READER,
                run
            )
        ) {
            where = kWhereMessagesCounter;
        }
        const uint64_t tail_line = (
            counter
                ? 0
                : reader.core_noc(
                    messages_counter(
                        command,
                        counter_tiles == 0 ? 1 : counter_tiles,
                        arguments.me,
                        offset
                    ),
                    arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE +
                        16 * arguments.me
                )
        );
        // A core's tail, read from its counter tile into IN_TAIL (with a timeout).
        auto refresh = [&] {
            if (counter) {
                return true;  // nobody sends to a counter tile: its tail stays its head
            }
            noc_async_read(
                tail_line,
                arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_IN_TAIL,
                16
            );
            const bool ok = noc_timeout(reader, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            });
            invalidate_l1_cache();
            return ok;
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
                tail_value == head && compute_flags[0] == run &&
                writer_flags[0] == run && (words[1] & kMessagesWaiting) &&
                (words[2] & kMessagesWaiting) && words[1] == words[2] &&
                cb_pages_reservable_at_back(kCbMessagesIn, in_pages)
            );
        };
        // True when the next command of this core in the ring is this run's end.
        auto end_command = [&] {
            const volatile km_command_t*
                ring = (
                    reinterpret_cast<const volatile km_command_t*>(
                        reader.control(KM_CONTROL_RING)
                    )
                );
            volatile tt_l1_ptr uint32_t* last = reader.state() + kStLast;
            for (uint32_t slot_index = 0; slot_index < arguments.depth; slot_index++) {
                const uint32_t sequence = ring[slot_index].sequence;
                if (!newer(sequence, last[slot_index])) {
                    continue;
                }
                km_command_t candidate;
                copy_command(&ring[slot_index], &candidate);
                if (
                    candidate.sequence == sequence && KM_LANDED(&candidate) &&
                    handler(candidate) == KM_HANDLER_MESSAGES_END &&
                    candidate.a0 == run && in_group(candidate, arguments.me)
                ) {
                    return true;
                }
            }
            return false;
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
            if (tail_value == head) {
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
            if (reader.poll(
                [&] {
                    return cb_pages_reservable_at_back(kCbMessagesIn, 1);
                },
                arguments.wait_limit
            ) != kGaveNone) {
                where = kWhereMessagesInPage;
                break;
            }
            cb_reserve_back(kCbMessagesIn, 1);
            const uint32_t page = get_write_ptr(kCbMessagesIn);
            uint32_t batch = (
                tail_value - head < kMessagesPacket
                    ? tail_value - head
                    : kMessagesPacket
            );
            for (uint32_t i = 0; i < batch;) {
                const uint32_t position = (head + i) & (messages_arguments.cap - 1);
                const uint32_t in_page = position % kMessagesPageMessages;
                uint32_t chunk = kMessagesPageMessages - in_page;
                chunk = chunk < batch - i ? chunk : batch - i;
                noc_async_read(
                    get_noc_addr(
                        messages_page(arguments.me, position, messages_arguments.cap),
                        queue_addresses,
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
            uint32_t valid = 0;
            // The prefix of valid marks: a message is valid only after its packet's
            // first message, whose mark is written once the whole packet has landed.
            while (
                valid < batch &&
                l1(
                    page + kMessagesHeader + valid * messages_arguments.message_bytes
                )[0] ==
                    messages_mark(head + valid)
            ) {
                valid++;
            }
            if (valid == 0) {  // the message at head has not landed yet
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
            check(page + kMessagesHeader, valid);
            l1(page)[0] = valid;
            l1(page)[1] = head;
            cb_push_back(kCbMessagesIn, 1);
            head += valid;
            *head_word = head;
        }
        if (where != 0) {
            // a gap: word 0 of the message found at the head, for the report
            messages_gave_up(
                arguments.control,
                where,
                head,
                run,
                l1(get_write_ptr(kCbMessagesIn))[kMessagesHeader / 4]
            );
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

    // Generic writer: outgoing messages in bins by destination; one atomic add per bin
    // reserves positions, then the messages go to that queue. True: the run ended.
    template <typename WriterType>
    inline bool messages_writer(WriterType& writer, const km_command_t& command) {
        const Args& arguments = writer.args();
        const MessagesArgs messages_arguments = messages_args(writer);
        const uint32_t offset = islands_at(arguments.control).offset();  // flat - core
        const uint32_t run = command.sequence;
        const uint32_t words = messages_arguments.message_bytes / 4;
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
        const uint32_t store = get_write_ptr(kCbMessagesBins);
        uint32_t bin_destination[kMessagesBinsMax], bin_fill[kMessagesBinsMax];
        for (uint32_t bin = 0; bin < kMessagesBinsMax; bin++) {
            bin_destination[bin] = kMessagesEnd;
            bin_fill[bin] = 0;
        }
        uint32_t taken = 0, sent = 0, held = 0, where = 0, flag = 0, lost_to = 0;
        messages[KM_MESSAGES_SENT / 4] = 0;
        writer_flags[1] = 0;
        writer_flags[0] = run;
        // A core reserves on the counter tiles: it waits until each shows READY.
        const uint32_t counter_tiles = messages_arguments.counters;
        const uint32_t ranks = messages_ranks(command, counter_tiles);
        if (counter_tiles == 0 || counter_tiles > KM_COUNTERS_MAX || ranks == 0) {
            where = kWhereMessagesCounter;
        } else if (arguments.me - first(command) < ranks) {
            for (
                uint32_t counter_tile = 0; counter_tile < counter_tiles && where == 0;
                counter_tile++
            ) {
                if (!messages_ready(
                    writer,
                    first(command) + ranks + counter_tile,
                    KM_TAILS_WRITER,
                    run
                )) {
                    where = kWhereMessagesCounter;
                }
            }
        }

        // `count` messages at `base` to `destination`: reserve, mark, write, wait.
        auto send = [&](uint32_t destination, uint32_t base, uint32_t count) {
            const uint32_t bin_messages = count;
            returned[0] = kMessagesEnd;
            invalidate_l1_cache();  // a fence: the store lands before the answer can
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index,
                write_at_cmd_buf,
                writer.core_noc(
                    messages_counter(command, counter_tiles, destination, offset),
                    arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE +
                        16 * destination
                ),
                NOC_UNICAST_WRITE_VC,
                bin_messages,
                31,
                false,
                false,
                returned_address
            );
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
                lost_to = destination;
                return false;
            }
            // Commit: the messages go out with the first one's mark 0, then after their
            // barrier one 16-byte write sets that mark.
            const uint32_t first_position = returned[0];
            l1(base)[0] = 0;
            for (uint32_t i = 1; i < bin_messages; i++) {
                l1(base + i * messages_arguments.message_bytes)[0] = messages_mark(
                    first_position + i
                );
            }
            invalidate_l1_cache();  // a fence: the NoC reads the stored marks
            for (uint32_t i = 0; i < bin_messages;) {
                const uint32_t position = (
                    (first_position + i) & (messages_arguments.cap - 1)
                );
                const uint32_t in_page = position % kMessagesPageMessages;
                uint32_t chunk = kMessagesPageMessages - in_page;
                chunk = chunk < bin_messages - i ? chunk : bin_messages - i;
                noc_async_write(
                    base + i * messages_arguments.message_bytes,
                    get_noc_addr(
                        messages_page(destination, position, messages_arguments.cap),
                        queue_addresses,
                        in_page * messages_arguments.message_bytes
                    ),
                    chunk * messages_arguments.message_bytes
                );
                i += chunk;
            }
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                where = kWhereMessagesFlush;
                return false;
            }
            l1(base)[0] = messages_mark(first_position);
            invalidate_l1_cache();
            noc_async_write(
                base,
                get_noc_addr(
                    messages_page(destination, first_position, messages_arguments.cap),
                    queue_addresses,
                    (first_position & (messages_arguments.cap - 1)) %
                        kMessagesPageMessages * messages_arguments.message_bytes
                ),
                16
            );
            if (!noc_timeout(writer, [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            })) {
                where = kWhereMessagesFlush;
                return false;
            }
            sent += bin_messages;
            messages[KM_MESSAGES_SENT / 4] = sent;
            return true;
        };
        // Bin `bin` to its destination.
        auto flush = [&](uint32_t bin) {
            const uint32_t base = (
                store + bin * kMessagesPacket * messages_arguments.message_bytes
            );
            if (!send(bin_destination[bin], base, bin_fill[bin])) {
                return false;
            }
            bin_fill[bin] = 0;
            bin_destination[bin] = kMessagesEnd;
            held--;
            return true;
        };
        auto flush_all = [&] {
            for (uint32_t bin = 0; bin < bins; bin++) {
                if (bin_fill[bin] != 0 && !flush(bin)) {
                    return false;
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
        for (uint32_t loop = 0; where == 0 && !ended; loop++) {
            invalidate_l1_cache();
            if (writer.stop()) {
                where = kWhereMessagesStopped;
                break;
            }
            if (!cb_pages_available_at_front(kCbMessagesOut, 1)) {
                if (
                    held != 0 && compute_flags[0] == run &&
                    (compute_flags[1] & kMessagesWaiting)
                ) {
                    flush_all();  // compute waits: nothing stays in a bin
                    continue;
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
            cb_wait_front(kCbMessagesOut, 1);
            volatile tt_l1_ptr uint32_t* outgoing = l1(get_read_ptr(kCbMessagesOut));
            const uint32_t destination = outgoing[0];
            if (destination == kMessagesEnd) {
                cb_pop_front(kCbMessagesOut, 1);
                ended = flush_all();
                break;
            }
            if (destination & kMessagesInPlace) {  // a page sent from where it lies
                if (!send(destination & ~kMessagesInPlace, outgoing[1], 1)) {
                    break;
                }
                cb_pop_front(kCbMessagesOut, 1);
                taken++;
                flag = taken;
                writer_flags[1] = taken;
                continue;
            }
            const uint32_t bin = destination % bins;
            if (
                bin_destination[bin] != destination && bin_fill[bin] != 0 && !flush(bin)
            ) {
                break;
            }
            if (bin_fill[bin] == 0) {
                bin_destination[bin] = destination;
                held++;
            }
            volatile tt_l1_ptr uint32_t* slot_words = (
                l1(
                    store +
                    (bin * kMessagesPacket + bin_fill[bin]) *
                        messages_arguments.message_bytes
                )
            );
            for (uint32_t word = 1; word < words; word++) {
                slot_words[word] = outgoing[word];
            }
            bin_fill[bin]++;
            cb_pop_front(kCbMessagesOut, 1);
            taken++;
            flag = taken;
            writer_flags[1] = taken;
            if (bin_fill[bin] == kMessagesPacket && !flush(bin)) {
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

    // A packet of messages in L1 (the input buffer's page), valid until done().
    struct Packet {
        uint32_t count;
        uint32_t base;  // byte address of message 0
        uint32_t message_bytes;

        explicit operator bool() const {
            return count != 0;
        }

        const volatile tt_l1_ptr uint32_t* operator[](uint32_t index) const {
            return l1(base + index * message_bytes);
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
            first_(km::first(command)),
            count_(messages_ranks(command, compute.arg(kMessagesCArgCounters))),
            rank_(compute.me() - km::first(command)),
            message_bytes_(compute.arg(kMessagesCArgMessage)),
            quota_(compute.arg(kMessagesCArgMaxLive) / (count_ != 0 ? count_ : 1u)),
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
            return {packet_count, address + kMessagesHeader, message_bytes_};
        }

        // The message goes on at rank `destination` of the run's group. message:
        // message_bytes / 4 words; word 0 is the runtime's mark.
        void forward(uint32_t destination, const uint32_t* message) {
            if (destination >= count_) {
                failed_ = true;
                PACK((
                    messages_gave_up(control_, kWhereMessagesBadDest, destination, run_)
                ));
                PACK((command_gave_up(
                    control_, kStCompute, kGaveStop, kWhereCOutput, 0, run_
                )));
                return;
            }
            PACK(({
                if (!failed_) {
                    push_out(first_ + destination, message);
                }
            }));
        }

        // The page at L1 address `address` (a message of the current packet) goes on at
        // rank `destination` as it is: the writer sends it from there; done() waits.
        void forward_page(uint32_t destination, uint32_t address) {
            if (destination >= count_) {
                failed_ = true;
                PACK((
                    messages_gave_up(control_, kWhereMessagesBadDest, destination, run_)
                ));
                PACK((command_gave_up(
                    control_, kStCompute, kGaveStop, kWhereCOutput, 0, run_
                )));
                return;
            }
            PACK(({
                if (!failed_ && reserve()) {
                    volatile tt_l1_ptr uint32_t* outgoing = (
                        l1(
                            get_local_cb_interface(kCbMessagesOut).fifo_wr_ptr
                            << cb_addr_shift
                        )
                    );
                    outgoing[1] = address;
                    outgoing[0] = (first_ + destination) | kMessagesInPlace;
                    __sync_synchronize();  // the words are in L1 before the push
                    cb_push_back(kCbMessagesOut, 1);
                    pushed_++;
                }
            }));
        }

        // The message ends here: one send returns to this core's quota.
        void consume() {
            quota_++;
        }

        // A new message at rank `destination`; false when this core has no quota left.
        bool send(uint32_t destination, const uint32_t* message) {
            if (quota_ == 0) {
                return false;
            }
            quota_--;
            forward(destination, message);
            return true;
        }

        // The packet's page is free; CONSUMED counts its messages.
        void done(const Packet& packet) {
            MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1)));
            PACK((wait_taken()));
            PACK((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1)));
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
        // Pack: waits until this run's writer has taken every pushed message, so the
        // pages it sends from where they lie are out (with a timeout, the stop flag).
        void wait_taken() {
            volatile tt_l1_ptr uint32_t* writer_flags = (
                l1(
                    control_ + KM_CONTROL_MESSAGES + KM_MESSAGES_FLAGS + kMessagesFlagWriter
                )
            );
            const uint32_t code = poll_stop(
                l1(control_ + KM_CONTROL_STOP),
                [&] {
                    const uint32_t taken = writer_flags[1] & ~kMessagesWaiting;
                    return (
                        writer_flags[0] == run_ &&
                        static_cast<int32_t>(taken - pushed_) >= 0
                    );
                },
                wait_,
                0
            );
            if (code != kGaveNone) {
                failed_ = true;
                messages_gave_up(control_, kWhereMessagesOutPage, pushed_, run_);
                command_gave_up(control_, kStCompute, code, kWhereCOutput, 0, run_);
            }
        }

        // Pack: one outgoing message, word 0 the destination core.
        void push_out(uint32_t destination, const uint32_t* message) {
            if (!reserve()) {
                return;
            }
            volatile tt_l1_ptr uint32_t* outgoing = (
                l1(get_local_cb_interface(kCbMessagesOut).fifo_wr_ptr << cb_addr_shift)
            );
            for (uint32_t word = 1; word < message_bytes_ / 4; word++) {
                outgoing[word] = message[word];
            }
            outgoing[0] = destination;
            __sync_synchronize();  // the message is in L1 before the push
            cb_push_back(kCbMessagesOut, 1);
            pushed_++;
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
        uint32_t first_;
        uint32_t count_;
        uint32_t rank_;
        uint32_t message_bytes_;
        uint32_t quota_;
        bool seeded_;
        uint32_t pushed_ = 0;
        uint32_t consumed_ = 0;
        bool failed_ = false;
    };

}  // namespace km
#endif
