// The chain (komorebi/abi/km_chain.h): each kernel runs a chain command's records in
// turn; a record's input waits for the marks of the record before's roots, not a host.
#pragma once

#include <stdint.h>

#include "komorebi/abi/km_chain.h"
#include "km_group.hpp"
#include "km_worker.hpp"

namespace km {

    // The chain's words in buffer kChainBuffer (bytes): marks and credits (one a record
    // parity) other cores count up (16-byte slots), the ring, the counts waited for.
    constexpr uint32_t kChainMarks = 0;     // even records'; odd at kChainMarksOdd
    constexpr uint32_t kChainCredits = 16;  // even records'; odd at kChainCreditsOdd
    constexpr uint32_t kChainReady = 32;    // the reader: the last record read's tag
    constexpr uint32_t kChainWriterTaken = 48;   // the writer: the last record it took
    constexpr uint32_t kChainComputeTaken = 64;  // unpack: the last record compute took
    constexpr uint32_t kChainReaderWanted = 80;  // marks the reader waited for (even)
    constexpr uint32_t kChainReaderWantedOdd = 84;
    constexpr uint32_t kChainWriterWantedOdd = 88;
    constexpr uint32_t kChainWriterWanted = 96;    // marks the writer waited for (even)
    constexpr uint32_t kChainCreditsWanted = 100;  // credits the writer waited for
    constexpr uint32_t kChainCreditDue = 104;  // 1: this record's emit waits for a credit
    constexpr uint32_t kChainCreditsWantedOdd = 108;
    constexpr uint32_t kChainCreditsOdd = 112;
    constexpr uint32_t kChainSlots = 128;  // record r in slot r % 2
    constexpr uint32_t kChainMarksOdd = kChainStamps + kChainStampBytes;  // past them
    constexpr uint32_t kChainAreaBytes = kChainMarksOdd + 16;  // the buffer's page

    // A record's tag in the ring: the chain command's sequence and its index.
    inline uint32_t chain_tag(uint32_t sequence, uint32_t record) {
        return sequence * kChainRecordsMax + record;
    }

    // The command a record names: the chain command's header and sequence, its words
    // (the one the info word names ORed with the chain's a0, the per-submit word).
    inline km_command_t chain_command(
        const km_command_t& chain, const uint32_t* record
    ) {
        km_command_t command = chain;
        command.group = record[kChainGroup];
        command.a0 = record[kChainArguments + 0];
        command.a1 = record[kChainArguments + 1];
        command.a2 = record[kChainArguments + 2];
        command.a3 = record[kChainArguments + 3];
        command.a4 = record[kChainArguments + 4];
        command.a5 = record[kChainArguments + 5];
        command.a6 = record[kChainArguments + 6];
        command.a7 = record[kChainArguments + 7];
        command.a8 = record[kChainArguments + 8];
        command.a9 = record[kChainArguments + 9];
        const uint32_t word = (
            (record[kChainInfo] >> kChainTokenShift) & kChainTokenMask
        );
        reinterpret_cast<uint32_t*>(&command)[word] |= word != 0 ? chain.a0 : 0u;
        return command;
    }

    // Whether core `me` is one of the record's cores.
    inline bool chain_member(const uint32_t* record, uint32_t me) {
        const uint32_t first = record[kChainGroup] & 0xFFFFu;
        return me >= first && me - first < (record[kChainGroup] >> 16);
    }

}  // namespace km

#if defined(KM_CHAIN_STAMPS) && !defined(KM_OVERLAY_PART)
#define KM_CHAIN_STAMP(statement) statement
#else
#define KM_CHAIN_STAMP(statement)
#endif

#if defined(KM_CHAIN_STAMPS) && !defined(KM_OVERLAY_PART)
namespace km {

    // This kernel's row of the record it runs (0: none), its chain's start, the words
    // stamped once a record (bits).
    inline uint32_t chain_stamp_row = 0;
    inline uint32_t chain_stamp_start = 0;
    inline uint32_t chain_stamp_once_bits = 0;

    inline void chain_stamp_add(uint32_t word, uint32_t cycles) {
        if (chain_stamp_row == 0) {
            return;
        }
#if !defined(COMPILE_FOR_TRISC)
        invalidate_l1_cache();  // the writer zeroes the rows after a read-out
#endif
        volatile tt_l1_ptr uint32_t* row = l1(chain_stamp_row);
        row[word] += cycles;
    }

    inline void chain_stamp(uint32_t word) {
        chain_stamp_add(word, wall() - chain_stamp_start);
    }

    inline void chain_stamp_once(uint32_t word) {
        if ((chain_stamp_once_bits >> word & 1u) == 0) {
            chain_stamp_once_bits |= 1u << word;
            chain_stamp(word);
        }
    }

    // Kernel `kernel` takes a record of stamp row `record` (info's top byte; the
    // chain's buffer at `area`).
    inline void chain_stamp_take(
        uint32_t area, uint32_t kernel, const uint32_t* record
    ) {
        chain_stamp_row = 0;
        const uint32_t index = record[kChainInfo] >> kChainRowShift;
        if (index < kChainStampRecords) {
            const uint32_t row = kernel * kChainStampRecords + index;
            chain_stamp_row = area + kChainStamps + row * kChainStampWords * 4;
        }
        chain_stamp_once_bits = 0;
        chain_stamp_add(0, 1);
        chain_stamp(kChainStampTaken);
    }

}  // namespace km
#endif

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#include "km_movement.hpp"

namespace km {

    inline volatile tt_l1_ptr uint32_t* chain_word(uint32_t offset) {
        const LocalCBInterface& buffer = get_local_cb_interface(kChainBuffer);
        return l1(buffer.fifo_limit - buffer.fifo_size + offset);
    }

    inline uint32_t chain_address(uint32_t offset) {
        const LocalCBInterface& buffer = get_local_cb_interface(kChainBuffer);
        return buffer.fifo_limit - buffer.fifo_size + offset;
    }

    // The credit words of a record parity: a credit for the record after cannot meet
    // this one's wait.
    inline uint32_t chain_credits_at(bool odd) {
        return odd ? kChainCreditsOdd : kChainCredits;
    }

    inline volatile tt_l1_ptr uint32_t* chain_credits_wanted() {
        const bool odd = (*chain_word(kChainWriterTaken) & 1u) != 0;
        return chain_word(odd ? kChainCreditsWantedOdd : kChainCreditsWanted);
    }

    // Writer, before a tree's emit in a chain's record: its receiver's credit (none
    // due: at once). False: gave up (recorded).
    template <typename Writer>
    bool chain_credit(const Writer& writer) {
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* due = chain_word(kChainCreditDue);
        if (*due == 0) {
            return true;
        }
        *due = 0;
        const uint32_t credits = (
            chain_credits_at((*chain_word(kChainWriterTaken) & 1u) != 0)
        );
        volatile tt_l1_ptr uint32_t* wanted = chain_credits_wanted();
        const uint32_t target = *wanted + 1;
        *wanted = target;
        KM_CHAIN_STAMP(const uint32_t at = wall());
        const uint32_t code = writer.poll_at(
            kWhereEmit,
            [&] {
                invalidate_l1_cache();
                return static_cast<int32_t>(*chain_word(credits) - target) >= 0;
            },
            writer.args().wait_limit
        );
        KM_CHAIN_STAMP(chain_stamp_add(kChainStampCredit, wall() - at));
        if (code != kGaveNone) {
            writer.gave_up(kStWriter, code, kWhereEmit, 0xC3u, 0);
        }
        return code == kGaveNone;
    }

}  // namespace km

#if !defined(KM_OVERLAY_PART)
namespace km {

    // Tiles first .. first + count - 1 of a row written in blocks of `width` tiles by
    // `groups` groups: the blocks they meet.
    inline uint32_t chain_blocks(
        uint32_t first, uint32_t count, uint32_t width, uint32_t groups
    ) {
        if (count == 0 || width == 0 || first / width >= groups) {
            return 0;
        }
        const uint32_t last = (first + count - 1) / width;
        return (last < groups ? last : groups - 1) - first / width + 1;
    }

    struct ChainRange {
        uint32_t first;
        uint32_t count;
    };

    // A record's group size, the output tiles a group writes, its K range and its
    // tree's form, as the record states them; a program may give its own.
    struct ChainShape {
        template <typename Kernel>
        uint32_t size(const Kernel&, const uint32_t* record) const {
            return (record[kChainInfo] >> kChainSizeShift) & kChainSizeMask;
        }

        template <typename Kernel>
        uint32_t width(const Kernel&, const uint32_t* record) const {
            return (record[kChainInfo] >> kChainWidthShift) & kChainWidthMask;
        }

        template <typename Kernel>
        ChainRange k_range(const Kernel&, const uint32_t* record) const {
            const uint32_t word = record[kChainRanges];
            return {word & 0xFFFu, (word >> 20) & 0x7FFu};
        }

        uint32_t form(const km_command_t&) const {
            return kReduceTree;
        }
    };

    // The running chain on this kernel: on, whether a record is before this one, this
    // record's index and tag base, its words and the one before's (its input's writer).
    struct ChainNow {
        bool on;
        bool has_before;
        uint32_t record;
        uint32_t tag_base;
        uint32_t now[kChainWords];
        uint32_t before[kChainWords];
    };

    inline ChainNow chain_now = {};

    // Range i of what rank `rank` of group `index` of `record` reads of the record
    // before's output (count 0: none).
    template <typename Kernel, typename Shape>
    ChainRange chain_need(
        const Shape& shape,
        const Kernel& core,
        const uint32_t* record,
        uint32_t i,
        uint32_t rank,
        uint32_t index
    ) {
        if (((record[kChainInfo] >> kChainNeedsShift) & 1u) == kChainNeedsK) {
            if (i != 0) {
                return {0, 0};
            }
            const ChainRange k = shape.k_range(core, record);
            const Partition part = partition(k.count, rank, shape.size(core, record));
            return {k.first + part.start, part.len};
        }
        const uint32_t word = record[kChainRanges + i];
        if (word == 0 || ((word >> 31) != 0 && rank != 0)) {
            return {0, 0};
        }
        return {
            (word & 0xFFFu) + ((word >> 12) & 0xFFu) * index, (word >> 20) & 0x7FFu
        };
    }

    // The marks this core's record waits for: the blocks of the record before's output
    // its ranges meet, one mark a block and range.
    template <typename Kernel, typename Shape>
    uint32_t chain_wanted(const Shape& shape, const Kernel& core) {
        const uint32_t* now = chain_now.now;
        const uint32_t* before = chain_now.before;
        const uint32_t size = shape.size(core, now);
        const uint32_t index = core.me() - (now[kChainGroup] & 0xFFFFu);
        const uint32_t groups = (before[kChainGroup] >> 16) / shape.size(core, before);
        const uint32_t width = shape.width(core, before);
        uint32_t wanted = 0;
        for (uint32_t i = 0; i < kChainRangeCount; i++) {
            const ChainRange range = (
                chain_need(shape, core, now, i, index % size, index / size)
            );
            wanted += chain_blocks(range.first, range.count, width, groups);
        }
        return wanted;
    }

    // The marks word of record `record` (its parity): a mark for the record after
    // cannot meet this one's wait.
    inline uint32_t chain_marks_at(uint32_t record) {
        return (record & 1u) != 0 ? kChainMarksOdd : kChainMarks;
    }

    // This record's marks: false when it waits for none; else `target`, the count the
    // marks word reaches once its input is written (this mover's total, kept in L1).
    template <typename Kernel, typename Shape>
    bool chain_target(const Shape& shape, const Kernel& core, uint32_t& target) {
        if (
            !chain_now.on || !chain_now.has_before ||
            (chain_now.now[kChainInfo] & kChainMarked) == 0
        ) {
            return false;
        }
        const bool odd = (chain_now.record & 1u) != 0;
        volatile tt_l1_ptr uint32_t* wanted = (
            chain_word(
                kMover == kMoverA
                    ? (odd ? kChainReaderWantedOdd : kChainReaderWanted)
                    : (odd ? kChainWriterWantedOdd : kChainWriterWanted)
            )
        );
        target = *wanted + chain_wanted(shape, core);
        *wanted = target;
        return true;
    }

    // Whether this record's marks word has reached `target`.
    inline bool chain_marked(uint32_t target) {
        invalidate_l1_cache();
        return (
            static_cast<int32_t>(
                *chain_word(chain_marks_at(chain_now.record)) - target
            ) >= 0
        );
    }

    // The record's input: waits until it is marked (a record without marks: at once).
    // False: gave up (recorded).
    template <typename Kernel, typename Shape>
    bool chain_input(const Shape& shape, const Kernel& core) {
        uint32_t target = 0;
        if (!chain_target(shape, core, target)) {
            KM_CHAIN_STAMP(chain_stamp(kChainStampMarked));
            return true;
        }
        const uint32_t code = core.poll_at(
            kWherePort,
            [&] {
                return chain_marked(target);
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(kStMover, code, kWherePort, 0xC0u, chain_now.record);
        }
        KM_CHAIN_STAMP(chain_stamp(kChainStampMarked));
        return code == kGaveNone;
    }

    // What a body run while the input is awaited did: nothing, some work (the wait's
    // limit starts again), or gave up (recorded).
    enum ChainBody : uint32_t { kChainIdle = 0, kChainWorked = 1, kChainStopped = 2 };

    // Until marked `target` (chain_target): `body()` again and again, its idle passes
    // bounded by the wait limit and the stop flag. False: gave up (recorded).
    template <typename Kernel, typename Body>
    bool chain_wait(const Kernel& core, uint32_t target, Body body) {
        const uint64_t limit = static_cast<uint64_t>(core.args().wait_limit) << 10;
        uint64_t deadline = wall64() + limit;
        for (uint32_t pass = 0; !chain_marked(target); pass++) {
            const uint32_t did = body();
            if (did == kChainStopped) {
                return false;
            }
            if (did == kChainWorked) {
                deadline = wall64() + limit;
                continue;
            }
            if ((pass & 15) == 15) {
                if (core.stop()) {
                    core.gave_up(kStMover, kGaveStop, kWherePort, 0xC0u, 0);
                    return false;
                }
                if (wall64() > deadline) {
                    core.gave_up(kStMover, kGaveLimit, kWherePort, 0xC0u, 0);
                    return false;
                }
            }
        }
        return true;
    }

    // Reader: record r's 64 bytes from DRAM into its slot, then its tag. False: gave
    // up.
    inline bool chain_fetch(
        const Reader& reader, uint32_t bank, uint32_t address, uint32_t tag
    ) {
        const uint32_t slot = (
            chain_address(kChainSlots + (tag % 2) * kChainRecordBytes)
        );
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank, address), slot, kChainRecordBytes
        );
        if (!reader.flushed(kFlushReads, kStReader)) {
            return false;
        }
        *chain_word(kChainReady) = tag;
        return true;
    }

    // Record `tag`'s words from its slot.
    inline void chain_copy(uint32_t tag, uint32_t* record) {
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* slot = (
            chain_word(kChainSlots + (tag % 2) * kChainRecordBytes)
        );
        for (uint32_t word = 0; word < kChainWords; word++) {
            record[word] = slot[word];
        }
    }

    // The record after this one into `words` when `wanted` and the reader has it in
    // the ring already. False: none (`words` this record's).
    inline bool chain_peek(uint32_t* words, bool wanted) {
        const uint32_t tag = chain_now.tag_base + chain_now.record + 1;
        for (uint32_t word = 0; word < kChainWords; word++) {  // none: this record's
            words[word] = chain_now.now[word];
        }
        invalidate_l1_cache();
        if (
            !wanted || !chain_now.on || chain_now.now[kChainNext] == 0 ||
            static_cast<int32_t>(*chain_word(kChainReady) - tag) < 0
        ) {
            return false;
        }
        chain_copy(tag, words);
        return true;
    }

    // Waits until the word at `offset` has reached `tag`. False: gave up (recorded).
    template <typename Kernel>
    bool chain_reached(
        const Kernel& core, uint32_t offset, uint32_t tag, uint32_t word
    ) {
        const uint32_t code = core.poll_at(
            kWhereData,
            [&] {
                invalidate_l1_cache();
                return static_cast<int32_t>(*chain_word(offset) - tag) >= 0;
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(word, code, kWhereData, 0xC1u, tag);
        }
        return code == kGaveNone;
    }

    // Before a record (both movers): this record, the one before it.
    inline void chain_enter(uint32_t record, uint32_t tag_base, const uint32_t* words) {
        chain_now.has_before = record != 0;
        if (record != 0) {
            for (uint32_t word = 0; word < kChainWords; word++) {
                chain_now.before[word] = chain_now.now[word];
            }
        }
        for (uint32_t word = 0; word < kChainWords; word++) {
            chain_now.now[word] = words[word];
        }
        chain_now.record = record;
        chain_now.tag_base = tag_base;
        chain_now.on = true;
    }

    // Reader: the chain from its first record (a7, a8's bank), `product` for each
    // record this core is in; record r + 1 read once compute and the writer took r - 1.
    template <typename Product>
    void chain_reader(Reader& reader, const km_command_t& command, Product product) {
        const uint32_t bank = (command.a8 >> KM_BLOCK_BANK_SHIFT) - 1u;
        const uint32_t base = chain_tag(command.sequence, 0);
        uint32_t address = command.a7;
        if (
            bank >= KM_BLOCK_BANKS || address == 0 ||
            address % kChainRecordBytes != 0 ||
            !chain_fetch(reader, bank, address, base)
        ) {
            reader.gave_up(kStReader, kGaveBadItem, kWhereData, 0xC2u, 0);
            return;
        }
        KM_CHAIN_STAMP(chain_stamp_start = wall());
        for (uint32_t record = 0; record < kChainRecordsMax; record++) {
            uint32_t words[kChainWords];
            chain_copy(base + record, words);
            const uint32_t next = words[kChainNext];
            KM_CHAIN_STAMP(const uint32_t fetch_at = wall());
            if (next != 0) {
                if (
                    record >= 1 &&
                    (!chain_reached(
                        reader, kChainWriterTaken, base + record - 1, kStReader
                    ) ||
                     !chain_reached(
                         reader, kChainComputeTaken, base + record - 1, kStReader
                     ))
                ) {
                    break;
                }
                if (
                    record + 1 == kChainRecordsMax || next % kChainRecordBytes != 0 ||
                    !chain_fetch(reader, bank, next, base + record + 1)
                ) {
                    reader.gave_up(kStReader, kGaveBadItem, kWhereData, 0xC2u, 0);
                    break;
                }
            }
            KM_CHAIN_STAMP(const uint32_t fetched = wall() - fetch_at);
            chain_enter(record, base, words);
            if (chain_member(words, reader.me())) {
                KM_CHAIN_STAMP(chain_stamp_take(chain_address(0), 0, words));
                KM_CHAIN_STAMP(chain_stamp_add(kChainStampLast, fetched));
                product(chain_command(command, words));
                KM_CHAIN_STAMP(chain_stamp(kChainStampDone));
                KM_CHAIN_STAMP(chain_stamp_row = 0);
            }
            if (next == 0 || reader.given_up(command)) {
                break;
            }
        }
        chain_now.on = false;
    }

    // Writer: the credits of a record whose parts go up its tree (kChainSends): one to
    // each sender of this core; and whether this core's own emit waits for one.
    template <typename Writer, typename Shape>
    void chain_credits(
        const Shape& shape, const Writer& writer, const km_command_t& one
    ) {
        if ((chain_now.now[kChainInfo] & kChainSends) == 0) {
            return;
        }
        const Group group = (
            digit_group(one, writer.me(), 1, shape.size(writer, chain_now.now))
        );
        const uint32_t form = shape.form(one);
        for (uint32_t rank = 0; rank < group.size; rank++) {
            Group sender = group;
            sender.rank = rank;
            const ReduceStep step = reduce_send(sender, form);
            if (rank != group.rank && !step.root && step.to == writer.me()) {
                noc_semaphore_inc(
                    writer.core_noc(
                        writer.core_of(core_of(group, rank)),
                        chain_address(chain_credits_at((chain_now.record & 1u) != 0))
                    ),
                    1
                );
            }
        }
        *chain_word(kChainCreditDue) = group.rank != 0 ? 1u : 0u;
    }

    // Where a direct input lands and comes from: the L1 of its K range on every reading
    // core (`tiles` tiles of `tile_bytes`) and the root's block in its own L1.
    struct ChainDirect {
        uint32_t source;
        uint32_t area;
        uint32_t tiles;
        uint32_t tile_bytes;
    };

    // A root, a kChainDataMark next: tiles low .. high - 1 of its output (at
    // direct.source) into each next core's K range, its trid's acks waited.
    template <typename Writer, typename Shape>
    bool chain_send(
        const Shape& shape,
        const Writer& writer,
        const uint32_t* next,
        uint32_t low,
        uint32_t high,
        const ChainDirect& direct
    ) {
        const uint32_t first = next[kChainGroup] & 0xFFFFu;
        const uint32_t count = next[kChainGroup] >> 16;
        const uint32_t size = shape.size(writer, next), groups = count / size;
        const uint32_t bytes = direct.tile_bytes;
        for (uint32_t rank = 0; rank < size; rank++) {
            const ChainRange range = chain_need(shape, writer, next, 0, rank, 0);
            if (
                ((next[kChainInfo] >> kChainNeedsShift) & 1u) != kChainNeedsK ||
                range.count > direct.tiles
            ) {
                writer.gave_up(kStWriter, kGaveBadItem, kWhereData, 0xC4u, 0);
                return false;
            }
            const uint32_t lo = range.first > low ? range.first : low;
            const uint32_t end = range.first + range.count;
            const uint32_t hi = end < high ? end : high;
            noc_async_write_set_trid(kChainInputTrid);
            for (uint32_t group = 0; lo < hi && group < groups; group++) {
                noc_async_write(
                    direct.source + (lo - low) * bytes,
                    writer.core_noc(
                        writer.core_of(first + group * size + rank),
                        direct.area + (lo - range.first) * bytes
                    ),
                    (hi - lo) * bytes
                );
            }
            noc_async_write_set_trid(0);
        }
        const uint32_t code = writer.poll_at(
            kWhereData,
            [] {
                return (
                    ncrisc_noc_nonposted_write_with_transaction_id_flushed(
                        noc_index, kChainInputTrid
                    )
                );
            },
            writer.args().wait_limit
        );
        if (code != kGaveNone) {
            writer.gave_up(kStWriter, code, kWhereData, 0xC4u, 0);
            return false;
        }
        return true;
    }

    // A group's root, its output written: its writes acked (a direct input sent; with
    // `dram_too` the DRAM copy too), a mark to each next core a range meets this block.
    template <typename Writer, typename Shape>
    bool chain_mark(
        const Shape& shape,
        const Writer& writer,
        const km_command_t& command,
        const ChainDirect& direct,
        bool dram_too
    ) {
        const uint32_t* now = chain_now.now;
        if (!chain_now.on || now[kChainNext] == 0) {
            return true;
        }
        const uint32_t tag = chain_now.tag_base + chain_now.record + 1;
        if (!chain_reached(writer, kChainReady, tag, kStWriter)) {
            return false;
        }
        uint32_t next[kChainWords];
        chain_copy(tag, next);
        if ((next[kChainInfo] & kChainMarked) == 0) {
            return true;
        }
        const uint32_t width = shape.width(writer, now);
        const uint32_t low = (
            (writer.me() - km::first(command)) / shape.size(writer, now) * width
        );
        const uint32_t high = low + width;
        const bool data = (next[kChainInfo] & kChainDataMark) != 0;
        if (
            (data && !chain_send(shape, writer, next, low, high, direct)) ||
            ((!data || dram_too) && !writer.flushed(kFlushWrites, kStWriter))
        ) {
            return false;
        }
        KM_CHAIN_STAMP(chain_stamp(kChainStampLast));
        const uint32_t first = next[kChainGroup] & 0xFFFFu;
        const uint32_t count = next[kChainGroup] >> 16;
        const uint32_t size = shape.size(writer, next), groups = count / size;
        const uint32_t marks = chain_address(chain_marks_at(chain_now.record + 1));
        const auto meets = [&](const ChainRange& range) {
            return (
                range.count != 0 && range.first < high &&
                low < range.first + range.count
            );
        };
        const auto mark = [&](uint32_t rank) {
            noc_semaphore_inc(writer.core_noc(writer.core_of(rank), marks), 1);
        };
        const bool by_k = ((next[kChainInfo] >> kChainNeedsShift) & 1u) == kChainNeedsK;
        for (uint32_t rank = 0; by_k && rank < size; rank++) {
            if (meets(chain_need(shape, writer, next, 0, rank, 0))) {
                for (uint32_t group = 0; group < groups; group++) {
                    mark(first + group * size + rank);
                }
            }
        }
        for (uint32_t group = 0; !by_k && group < groups; group++) {
            for (uint32_t i = 0; i < kChainRangeCount; i++) {
                if (meets(chain_need(shape, writer, next, i, 0, group))) {
                    mark(first + group * size);
                }
                if (meets(chain_need(shape, writer, next, i, 1, group))) {
                    for (uint32_t rank = 1; rank < size; rank++) {
                        mark(first + group * size + rank);
                    }
                }
            }
        }
        KM_CHAIN_STAMP(chain_stamp(kChainStampMarks));
        return true;
    }

    // Writer: the chain, record by record (each taken in turn, `product` where this
    // core is in it, the credits first). True: every record reported.
    template <typename Product, typename Shape>
    bool chain_writer(
        const Shape& shape, Writer& writer, const km_command_t& command, Product product
    ) {
        const uint32_t base = chain_tag(command.sequence, 0);
        bool ok = true;
        KM_CHAIN_STAMP(chain_stamp_start = wall());
        for (uint32_t record = 0; ok && record < kChainRecordsMax; record++) {
            if (!chain_reached(writer, kChainReady, base + record, kStWriter)) {
                ok = false;
                break;
            }
            uint32_t words[kChainWords];
            chain_copy(base + record, words);
            *chain_word(kChainWriterTaken) = base + record;
            chain_enter(record, base, words);
            if (chain_member(words, writer.me())) {
                const km_command_t one = chain_command(command, words);
                KM_CHAIN_STAMP(chain_stamp_take(chain_address(0), 1, words));
                chain_credits(shape, writer, one);
                ok = product(one);
                KM_CHAIN_STAMP(chain_stamp_row = 0);
                invalidate_l1_cache();
                if (*chain_word(kChainCreditDue) != 0) {  // no emit came: credit kept
                    *chain_word(kChainCreditDue) = 0;
                    *chain_credits_wanted() += 1;
                }
            }
            if (words[kChainNext] == 0) {
                break;
            }
        }
        chain_now.on = false;
        const uint32_t code = writer.poll_at(
            kWhereEmit,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            writer.args().wait_limit
        );
        return ok && code == kGaveNone;
    }

#if defined(KM_CHAIN_STAMPS)
    inline void chain_stamps_zero() {
        for (uint32_t offset = 0; offset < kChainStampBytes; offset += 4) {
            *chain_word(kChainStamps + offset) = 0;
        }
    }

    // Writer: this core's stamps to page `me` of the interleaved tensor at `address`
    // (pages of `page_bytes`), then zero.
    template <typename Writer>
    bool chain_stamps_out(const Writer& writer, uint32_t address, uint32_t page_bytes) {
        const InterleavedAddrGen<true> pages = {
            .bank_base_address = address, .page_size = page_bytes
        };
        noc_async_write(
            chain_address(kChainStamps),
            get_noc_addr(writer.me(), pages),
            kChainStampBytes
        );
        if (!writer.flushed(kFlushWrites, kStWriter)) {
            return false;
        }
        chain_stamps_zero();
        return true;
    }
#endif

    // Reader, before its first command: the chain's words zero (nothing counts them
    // yet).
    inline void chain_start() {
        for (uint32_t offset = 0; offset < kChainSlots; offset += 4) {
            *chain_word(offset) = 0;
        }
        *chain_word(kChainMarksOdd) = 0;
        KM_CHAIN_STAMP(chain_stamps_zero());
    }

}  // namespace km
#endif
#else
namespace km {

    // Compute (unpack reads, math and pack get each word by mailbox): the chain, each
    // record taken in turn, `product` where this core is in it.
    template <typename Product>
    void chain_compute(Compute& compute, const km_command_t& command, Product product) {
        const uint32_t base = chain_tag(command.sequence, 0);
        KM_CHAIN_STAMP(uint32_t stamps = 0);  // pack's: the chain's buffer, from unpack
        KM_CHAIN_STAMP(UNPACK((compute.hand(
            ckernel::ThreadId::PackThreadId,
            ckernel::get_tile_l1_byte_address(kChainBuffer, 0)
        ))));
        KM_CHAIN_STAMP(
            PACK((stamps = compute.handed(ckernel::ThreadId::UnpackThreadId)))
        );
        KM_CHAIN_STAMP(PACK((chain_stamp_start = wall())));
        for (uint32_t record = 0; record < kChainRecordsMax; record++) {
            uint32_t words[kChainWords] = {};
            uint32_t ok = 0;
            UNPACK(({
                const uint32_t at = ckernel::get_tile_l1_byte_address(kChainBuffer, 0);
                volatile tt_l1_ptr uint32_t* ready = l1(at + kChainReady);
                const uint32_t code = poll_live(
                    compute.control(),
                    kRoleUnpack,
                    kWhereData,
                    true,
                    [&] {
                        return static_cast<int32_t>(*ready - (base + record)) >= 0;
                    },
                    compute.wait_limit(),
                    0
                );
                ok = code == kGaveNone;
                compute.hand(ckernel::ThreadId::MathThreadId, ok);
                compute.hand(ckernel::ThreadId::PackThreadId, ok);
                volatile tt_l1_ptr uint32_t* slot = (
                    l1(at + kChainSlots + ((base + record) % 2) * kChainRecordBytes)
                );
                for (uint32_t word = 0; ok && word < kChainWords; word++) {
                    words[word] = slot[word];
                    compute.hand(ckernel::ThreadId::MathThreadId, words[word]);
                    compute.hand(ckernel::ThreadId::PackThreadId, words[word]);
                }
                if (ok) {
                    *l1(at + kChainComputeTaken) = base + record;
                } else {
                    command_gave_up(
                        compute.control(), kStCompute, code, kWhereData, 0xC1u, record
                    );
                }
            }));
            MATH(({
                ok = compute.handed(ckernel::ThreadId::UnpackThreadId);
                for (uint32_t word = 0; ok && word < kChainWords; word++) {
                    words[word] = compute.handed(ckernel::ThreadId::UnpackThreadId);
                }
            }));
            PACK(({
                ok = compute.handed(ckernel::ThreadId::UnpackThreadId);
                for (uint32_t word = 0; ok && word < kChainWords; word++) {
                    words[word] = compute.handed(ckernel::ThreadId::UnpackThreadId);
                }
            }));
            if (ok == 0) {
                return;
            }
            if (chain_member(words, compute.me())) {
                KM_CHAIN_STAMP(PACK((chain_stamp_take(stamps, 2, words))));
                product(chain_command(command, words));
                KM_CHAIN_STAMP(PACK((chain_stamp_row = 0)));
            }
            if (words[kChainNext] == 0) {
                break;
            }
        }
    }

}  // namespace km
#endif
