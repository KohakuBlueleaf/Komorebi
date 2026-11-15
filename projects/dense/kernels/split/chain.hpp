// SPLIT's chain (SPLIT_CHAIN): each kernel runs the records of one command in turn;
// a record's input waits for the marks of the record before's roots, not the thread.
#pragma once

#include <stdint.h>

#include "chain.h"
#include "komorebi/device/km_group.hpp"
#include "split.h"

// The chain's words in buffer kChainBuffer (bytes): marks and credits (one a record
// parity) that other cores count up (16-byte slots), the ring, the counts waited for.
constexpr uint32_t kChainMarks = 0;
constexpr uint32_t kChainCredits = 16;         // even records'; odd records' at 112
constexpr uint32_t kChainReady = 32;           // the reader: the last record read's tag
constexpr uint32_t kChainWriterTaken = 48;     // the writer: the last record it took
constexpr uint32_t kChainComputeTaken = 64;    // unpack: the last record compute took
constexpr uint32_t kChainReaderWanted = 80;    // marks the reader waited for, in all
constexpr uint32_t kChainWriterWanted = 96;    // marks the writer waited for
constexpr uint32_t kChainCreditsWanted = 100;  // credits the writer waited for (even)
constexpr uint32_t kChainCreditDue = 104;  // 1: this record's emit waits for a credit
constexpr uint32_t kChainCreditsWantedOdd = 108;
constexpr uint32_t kChainCreditsOdd = 112;
constexpr uint32_t kChainSlots = 128;  // record r in slot r % 2
constexpr uint32_t kChainBytes = kChainSlots + 2 * kChainRecordBytes;

// A record's tag in the ring: the chain command's sequence and its index.
inline uint32_t chain_tag(uint32_t sequence, uint32_t record) {
    return sequence * kChainRecordsMax + record;
}

// The command a record names: the chain command's header and sequence, its words
// (one ORed with the chain's a0 where the info word names it).
inline km_command_t chain_command(const km_command_t& chain, const uint32_t* record) {
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
    const uint32_t token = (record[kChainInfo] >> kChainTokenShift) & kChainTokenMask;
    reinterpret_cast<uint32_t*>(&command)[token] |= token != 0 ? chain.a0 : 0u;
    return command;
}

inline bool chain_member(const uint32_t* record, uint32_t me) {
    const uint32_t first = record[kChainGroup] & 0xFFFFu;
    return me >= first && me - first < (record[kChainGroup] >> 16);
}

#if defined(SPLIT_CHAIN_STAMPS) && !defined(KM_OVERLAY_PART)
#define CHAIN_STAMP(statement) statement
#else
#define CHAIN_STAMP(statement)
#endif

#if defined(SPLIT_CHAIN_STAMPS) && !defined(KM_OVERLAY_PART)
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
    volatile tt_l1_ptr uint32_t* row = km::l1(chain_stamp_row);
    row[word] += cycles;
}

inline void chain_stamp(uint32_t word) {
    chain_stamp_add(word, km::wall() - chain_stamp_start);
}

inline void chain_stamp_once(uint32_t word) {
    if ((chain_stamp_once_bits >> word & 1u) == 0) {
        chain_stamp_once_bits |= 1u << word;
        chain_stamp(word);
    }
}

// Kernel `kernel` takes a record of stamp row `record` (info's top byte; the chain's
// buffer at `area`).
inline void chain_stamp_take(uint32_t area, uint32_t kernel, const uint32_t* record) {
    chain_stamp_row = 0;
    const uint32_t index = record[kChainInfo] >> 24;
    if (index < kChainStampRecords) {
        const uint32_t row = kernel * kChainStampRecords + index;
        chain_stamp_row = area + kChainStamps + row * kChainStampWords * 4;
    }
    chain_stamp_once_bits = 0;
    chain_stamp_add(0, 1);
    chain_stamp(kChainStampTaken);
}
#endif

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_movement.hpp"

inline volatile tt_l1_ptr uint32_t* chain_word(uint32_t offset) {
    const LocalCBInterface& buffer = get_local_cb_interface(kChainBuffer);
    return km::l1(buffer.fifo_limit - buffer.fifo_size + offset);
}

inline uint32_t chain_address(uint32_t offset) {
    const LocalCBInterface& buffer = get_local_cb_interface(kChainBuffer);
    return buffer.fifo_limit - buffer.fifo_size + offset;
}

// The credit words of the record the writer took last (its tag's parity): a credit
// for the record after cannot meet this one's wait.
inline uint32_t chain_credits_at(bool odd) {
    return odd ? kChainCreditsOdd : kChainCredits;
}

inline volatile tt_l1_ptr uint32_t* chain_credits_wanted() {
    const bool odd = (*chain_word(kChainWriterTaken) & 1u) != 0;
    return chain_word(odd ? kChainCreditsWantedOdd : kChainCreditsWanted);
}

// Writer, before an emit of a chain's record: its receiver's credit (none due: at
// once). False: gave up (recorded).
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
    const uint32_t code = writer.poll_at(
        km::kWhereEmit,
        [&] {
            invalidate_l1_cache();
            return static_cast<int32_t>(*chain_word(credits) - target) >= 0;
        },
        writer.args().wait_limit
    );
    if (code != km::kGaveNone) {
        writer.gave_up(km::kStWriter, code, km::kWhereEmit, 0xC3u, 0);
    }
    return code == km::kGaveNone;
}

#if !defined(KM_OVERLAY_PART)
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

// The running chain on this mover: the record, the one before (its input's writer), its
// index, its tag's base (off: no chain); the next W early: allowed, its tag and block.
struct ChainNow {
    bool on;
    bool has_before;
    bool ahead_on;
    bool data_dram;
    uint32_t record;
    uint32_t tag_base;
    uint32_t ahead_tag;
    uint32_t ahead_block;
    uint32_t now[kChainWords];
    uint32_t before[kChainWords];
};

inline ChainNow chain_now = {};

// A record's plan word.
template <typename Kernel>
uint32_t chain_plan(const Kernel& core, const uint32_t* record, uint32_t word) {
    return core.arg(record[kChainArguments + 5] * kSplitWords + word);
}

// Range i of what rank `rank` of group `index` of `record` reads of the record
// before's output (count 0: none).
template <typename Kernel>
ChainRange chain_need(
    const Kernel& core,
    const uint32_t* record,
    uint32_t i,
    uint32_t rank,
    uint32_t index
) {
    if (((record[kChainInfo] >> 8) & 0xFFu) == kChainNeedsK) {
        if (i != 0) {
            return {0, 0};
        }
        const km::Partition part = (
            km::partition(
                chain_plan(core, record, kSpKt), rank, chain_plan(core, record, kSpS)
            )
        );
        return {part.start, part.len};
    }
    const uint32_t word = record[kChainRanges + i];
    if (word == 0 || ((word >> 31) != 0 && rank != 0)) {
        return {0, 0};
    }
    return {(word & 0xFFFu) + ((word >> 12) & 0xFFu) * index, (word >> 20) & 0x7FFu};
}

// Output tiles a group of `record` writes.
template <typename Kernel>
uint32_t chain_width(const Kernel& core, const uint32_t* record) {
    const uint32_t width = (record[kChainInfo] >> 16) & 0xFFu;
    return width != 0 ? width : chain_plan(core, record, kSpOw);
}

// The marks this core's record waits for: the blocks of the record before's output
// its ranges meet, one mark a block and range.
template <typename Kernel>
uint32_t chain_wanted(const Kernel& core) {
    const uint32_t* now = chain_now.now;
    const uint32_t* before = chain_now.before;
    const uint32_t size = chain_plan(core, now, kSpS);
    const uint32_t index = core.me() - (now[kChainGroup] & 0xFFFFu);
    const uint32_t before_size = chain_plan(core, before, kSpS);
    const uint32_t groups = (before[kChainGroup] >> 16) / before_size;
    const uint32_t width = chain_width(core, before);
    uint32_t wanted = 0;
    for (uint32_t i = 0; i < kChainRangeCount; i++) {
        const ChainRange range = chain_need(core, now, i, index % size, index / size);
        wanted += chain_blocks(range.first, range.count, width, groups);
    }
    return wanted;
}

// This record's marks: false when it waits for none; else `target`, the count the
// marks word reaches once its input is written (this mover's total, kept in L1).
template <typename Kernel>
bool chain_target(const Kernel& core, uint32_t& target) {
    if (
        !chain_now.on || !chain_now.has_before ||
        (chain_now.now[kChainInfo] & kChainMarked) == 0
    ) {
        return false;
    }
    volatile tt_l1_ptr uint32_t* wanted = (
        chain_word(km::kMover == km::kMoverA ? kChainReaderWanted : kChainWriterWanted)
    );
    target = *wanted + chain_wanted(core);
    *wanted = target;
    return true;
}

inline bool chain_marked(uint32_t target) {
    invalidate_l1_cache();
    return static_cast<int32_t>(*chain_word(kChainMarks) - target) >= 0;
}

// The record's input: waits until it is marked (a record without marks: at once).
// False: gave up (recorded).
template <typename Kernel>
bool chain_input(const Kernel& core) {
    uint32_t target = 0;
    if (!chain_target(core, target)) {
        CHAIN_STAMP(chain_stamp(kChainStampMarked));
        return true;
    }
    const uint32_t code = core.poll_at(
        km::kWherePort,
        [&] {
            return chain_marked(target);
        },
        core.args().wait_limit
    );
    if (code != km::kGaveNone) {
        core.gave_up(km::kStMover, code, km::kWherePort, 0xC0u, chain_now.record);
    }
    CHAIN_STAMP(chain_stamp(kChainStampMarked));
    return code == km::kGaveNone;
}

// Until the input is marked: W's blocks once A freed the half (`held`: some are in),
// then the next record's (`after`); `skip`, `after_skip` the blocks. False: gave up.
template <typename Kernel, typename Accessor>
bool chain_weight_first(
    const Kernel& core,
    const km::Stream<Accessor>& weights,
    const km::Stream<Accessor>& after,
    bool held,
    uint32_t& skip,
    uint32_t& after_skip
) {
    skip = 0;
    after_skip = 0;
    uint32_t target = 0;
    if (!chain_target(core, target)) {
        CHAIN_STAMP(chain_stamp(kChainStampMarked));
        return true;
    }
    CHAIN_STAMP(uint32_t full_at = 0);
    km::MoveCursor cursor, later;
    uint32_t filled = 0, drained = 0;
    if (
        !km::move_begin(weights, cursor, filled, drained) ||
        !km::move_begin(after, later, filled, drained)
    ) {
        core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
        return false;
    }
    const uint32_t half = weights.pair.half(km::kMover);
    const uint64_t limit = static_cast<uint64_t>(core.args().wait_limit) << 10;
    uint64_t deadline = km::wall64() + limit;
    bool freed = held, any_after = false;
    for (uint32_t pass = 0; !chain_marked(target); pass++) {
        freed = (
            freed ||
            cb_pages_reservable_at_back(
                half, get_local_cb_interface(half).fifo_num_pages
            )
        );
        bool issued = false, open = false, reads = false, writes = false;
        bool later_issued = false;
        if (
            freed && !km::move_offer(core, weights, cursor, issued, open, reads, writes)
        ) {
            core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
            return false;
        }
        if (
            freed && !open &&
            !km::move_offer(core, after, later, later_issued, open, reads, writes)
        ) {
            core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
            return false;
        }
        if (issued || later_issued) {
            CHAIN_STAMP(chain_stamp_once(kChainStampWeight));
            if (!km::move_settle(core, true, false)) {
                return false;
            }
            km::Moved moved = {0, 0};
            km::move_hand_on(weights, cursor, issued, moved);
            any_after = any_after || later_issued;
            km::move_hand_on(after, later, later_issued, moved);
            deadline = km::wall64() + limit;
            continue;
        }
        CHAIN_STAMP(full_at = freed && full_at == 0 ? km::wall() | 1u : full_at);
        if ((pass & 15) == 15) {
            if (core.stop()) {
                core.gave_up(km::kStMover, km::kGaveStop, km::kWherePort, 0xC0u, 0);
                return false;
            }
            if (km::wall64() > deadline) {
                core.gave_up(km::kStMover, km::kGaveLimit, km::kWherePort, 0xC0u, 0);
                return false;
            }
        }
    }
    CHAIN_STAMP(
        chain_stamp_add(kChainStampFull, full_at != 0 ? km::wall() - full_at : 0)
    );
    CHAIN_STAMP(chain_stamp(kChainStampMarked));
    skip = cursor.block < cursor.blocks ? cursor.block : cursor.blocks;
    after_skip = (
        any_after ? (later.block < later.blocks ? later.block : later.blocks) : 0
    );
    return true;
}

// Reader: record r's 64 bytes from DRAM into its slot, then its tag. False: gave up.
inline bool chain_fetch(
    const km::Reader& reader, uint32_t bank, uint32_t address, uint32_t tag
) {
    const uint32_t slot = chain_address(kChainSlots + (tag % 2) * kChainRecordBytes);
    noc_async_read(
        get_noc_addr_from_bank_id<true>(bank, address), slot, kChainRecordBytes
    );
    if (!reader.flushed(km::kFlushReads, km::kStReader)) {
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

// The record after this one when this mover may read its W early: a product this core
// is in, already in the ring (`words`). False: none (`words` this record's).
template <typename Kernel>
bool chain_next(const Kernel& core, uint32_t* words) {
    const uint32_t tag = chain_now.tag_base + chain_now.record + 1;
    for (uint32_t word = 0; word < kChainWords; word++) {  // none: this record's
        words[word] = chain_now.now[word];
    }
    invalidate_l1_cache();
    if (
        !chain_now.on || !chain_now.ahead_on || chain_now.now[kChainNext] == 0 ||
        static_cast<int32_t>(*chain_word(kChainReady) - tag) < 0
    ) {
        return false;
    }
    chain_copy(tag, words);
    return (
        chain_member(words, core.me()) && words[kChainArguments + 4] != kSplitOp &&
        words[kChainArguments + 5] < kSplitPlans
    );
}

// This record's W blocks this mover read in the record before (0: none).
inline uint32_t chain_ahead_taken() {
    const uint32_t tag = chain_now.tag_base + chain_now.record;
    const uint32_t block = chain_now.ahead_tag == tag ? chain_now.ahead_block : 0;
    chain_now.ahead_tag = 0;
    return block;
}

// The next record's W read early up to block `block` (0: none).
inline void chain_ahead_keep(uint32_t block) {
    chain_now.ahead_tag = block != 0 ? chain_now.tag_base + chain_now.record + 1 : 0;
    chain_now.ahead_block = block;
}

// Waits until the word at `offset` has reached `tag`. False: gave up (recorded).
template <typename Kernel>
bool chain_reached(const Kernel& core, uint32_t offset, uint32_t tag, uint32_t word) {
    const uint32_t code = core.poll_at(
        km::kWhereData,
        [&] {
            invalidate_l1_cache();
            return static_cast<int32_t>(*chain_word(offset) - tag) >= 0;
        },
        core.args().wait_limit
    );
    if (code != km::kGaveNone) {
        core.gave_up(word, code, km::kWhereData, 0xC1u, tag);
    }
    return code == km::kGaveNone;
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

// Reader: the chain from the command's first record, `product` for each record this
// core is in; record r + 1 read at r's start once compute and the writer took r - 1.
template <typename Product>
void chain_records(km::Reader& reader, const km_command_t& command, Product product) {
    const uint32_t bank = (command.a8 >> KM_BLOCK_BANK_SHIFT) - 1u;
    const uint32_t base = chain_tag(command.sequence, 0);
    uint32_t address = command.a7;
    if (
        bank >= KM_BLOCK_BANKS || address == 0 || address % kChainRecordBytes != 0 ||
        !chain_fetch(reader, bank, address, base)
    ) {
        reader.gave_up(km::kStReader, km::kGaveBadItem, km::kWhereData, 0xC2u, 0);
        return;
    }
    CHAIN_STAMP(chain_stamp_start = km::wall());
    chain_now.ahead_on = (command.a1 & kChainAheadOff) == 0;
    chain_now.ahead_tag = 0;
    for (uint32_t record = 0; record < kChainRecordsMax; record++) {
        uint32_t words[kChainWords];
        chain_copy(base + record, words);
        const uint32_t next = words[kChainNext];
        CHAIN_STAMP(const uint32_t fetch_at = km::wall());
        if (next != 0) {
            if (
                record >= 1 &&
                (!chain_reached(
                    reader, kChainWriterTaken, base + record - 1, km::kStReader
                ) ||
                 !chain_reached(
                     reader, kChainComputeTaken, base + record - 1, km::kStReader
                 ))
            ) {
                return;
            }
            if (
                record + 1 == kChainRecordsMax || next % kChainRecordBytes != 0 ||
                !chain_fetch(reader, bank, next, base + record + 1)
            ) {
                reader.gave_up(
                    km::kStReader, km::kGaveBadItem, km::kWhereData, 0xC2u, 0
                );
                return;
            }
        }
        CHAIN_STAMP(const uint32_t fetched = km::wall() - fetch_at);
        chain_enter(record, base, words);
        if (chain_member(words, reader.me())) {
            CHAIN_STAMP(chain_stamp_take(chain_address(0), 0, words));
            CHAIN_STAMP(chain_stamp_add(kChainStampLast, fetched));
            product(chain_command(command, words));
            CHAIN_STAMP(chain_stamp(kChainStampDone));
            CHAIN_STAMP(chain_stamp_row = 0);
        }
        if (next == 0 || reader.given_up(command)) {
            break;
        }
    }
}

// Reader: chain_records, then no chain runs.
template <typename Product>
void chain_reader(km::Reader& reader, const km_command_t& command, Product product) {
    chain_records(reader, command, product);
    chain_now.on = false;
}

// Writer: the credits of a record whose parts go up its tree: one to each sender of
// this core; and whether this core's own emit waits for one.
template <typename Writer>
void chain_credits(const Writer& writer, const km_command_t& command) {
    if ((chain_now.now[kChainInfo] & kChainSends) == 0) {
        return;
    }
    const km::Group group = (
        km::digit_group(
            command, writer.me(), 1, chain_plan(writer, chain_now.now, kSpS)
        )
    );
    const uint32_t form = split_form(command);
    for (uint32_t rank = 0; rank < group.size; rank++) {
        km::Group sender = group;
        sender.rank = rank;
        const km::ReduceStep step = km::reduce_send(sender, form);
        if (rank != group.rank && !step.root && step.to == writer.me()) {
            noc_semaphore_inc(
                writer.core_noc(
                    writer.core_of(km::core_of(group, rank)),
                    chain_address(chain_credits_at((chain_now.record & 1u) != 0))
                ),
                1
            );
        }
    }
    *chain_word(kChainCreditDue) = group.rank != 0 ? 1u : 0u;
}

// The L1 a core's K range of a kChainDataMark record's input lands in: its arrival
// buffer from page kChainInputPage on; and the tiles it holds.
inline uint32_t chain_input_area() {
    const LocalCBInterface& arrival = get_local_cb_interface(kCbArrive);
    return (
        arrival.fifo_limit - arrival.fifo_size +
        kChainInputPage * arrival.fifo_page_size
    );
}

inline uint32_t chain_input_tiles() {
    const LocalCBInterface& arrival = get_local_cb_interface(kCbArrive);
    return (arrival.fifo_size - kChainInputPage * arrival.fifo_page_size) / kTileBytes;
}

// A root, a kChainDataMark next: block low .. high - 1 (kCbSplitOut's base, kept until
// its next credits) into each next core's K range, its trid's acks waited (not DRAM's).
template <typename Writer>
bool chain_send_input(
    const Writer& writer, const uint32_t* next, uint32_t low, uint32_t high
) {
    const uint32_t first = next[kChainGroup] & 0xFFFFu, count = next[kChainGroup] >> 16;
    const uint32_t size = chain_plan(writer, next, kSpS), groups = count / size;
    const LocalCBInterface& out = get_local_cb_interface(kCbSplitOut);
    const uint32_t source = out.fifo_limit - out.fifo_size, area = chain_input_area();
    for (uint32_t rank = 0; rank < size; rank++) {
        const ChainRange range = chain_need(writer, next, 0, rank, 0);
        if (
            ((next[kChainInfo] >> 8) & 0xFFu) != kChainNeedsK ||
            range.count > chain_input_tiles()
        ) {
            writer.gave_up(km::kStWriter, km::kGaveBadItem, km::kWhereData, 0xC4u, 0);
            return false;
        }
        const uint32_t lo = range.first > low ? range.first : low;
        const uint32_t end = range.first + range.count;
        const uint32_t hi = end < high ? end : high;
        noc_async_write_set_trid(kChainInputTrid);
        for (uint32_t group = 0; lo < hi && group < groups; group++) {
            noc_async_write(
                source + (lo - low) * kTileBytes,
                writer.core_noc(
                    writer.core_of(first + group * size + rank),
                    area + (lo - range.first) * kTileBytes
                ),
                (hi - lo) * kTileBytes
            );
        }
        noc_async_write_set_trid(0);
    }
    const uint32_t code = writer.poll_at(
        km::kWhereData,
        [] {
            return (
                ncrisc_noc_nonposted_write_with_transaction_id_flushed(
                    noc_index, kChainInputTrid
                )
            );
        },
        writer.args().wait_limit
    );
    if (code != km::kGaveNone) {
        writer.gave_up(km::kStWriter, code, km::kWhereData, 0xC4u, 0);
        return false;
    }
    return true;
}

// A group's root writer, its output written: the writes acked (or its input sent), then
// a mark on each core of the next record for each of its ranges that meets this block.
template <typename Writer>
bool chain_mark(const Writer& writer, const km_command_t& command) {
    const uint32_t* now = chain_now.now;
    if (!chain_now.on || now[kChainNext] == 0) {
        return true;
    }
    const uint32_t tag = chain_now.tag_base + chain_now.record + 1;
    if (!chain_reached(writer, kChainReady, tag, km::kStWriter)) {
        return false;
    }
    uint32_t next[kChainWords];
    chain_copy(tag, next);
    if ((next[kChainInfo] & kChainMarked) == 0) {
        return true;
    }
    const uint32_t width = chain_width(writer, now);
    const uint32_t low = (
        (writer.me() - km::first(command)) / chain_plan(writer, now, kSpS) * width
    );
    const uint32_t high = low + width;
    const bool data = (next[kChainInfo] & kChainDataMark) != 0;
    if (
        (data && !chain_send_input(writer, next, low, high)) ||
        (
            (!data || chain_now.data_dram) &&
            !writer.flushed(km::kFlushWrites, km::kStWriter)
        )
    ) {
        return false;
    }
    CHAIN_STAMP(chain_stamp(kChainStampLast));
    const uint32_t first = next[kChainGroup] & 0xFFFFu, count = next[kChainGroup] >> 16;
    const uint32_t size = chain_plan(writer, next, kSpS), groups = count / size;
    const uint32_t marks = chain_address(kChainMarks);
    const auto meets = [&](const ChainRange& range) {
        return (
            range.count != 0 && range.first < high && low < range.first + range.count
        );
    };
    const auto mark = [&](uint32_t rank) {
        noc_semaphore_inc(writer.core_noc(writer.core_of(rank), marks), 1);
    };
    const bool by_k = ((next[kChainInfo] >> 8) & 0xFFu) == kChainNeedsK;
    for (uint32_t rank = 0; by_k && rank < size; rank++) {
        if (meets(chain_need(writer, next, 0, rank, 0))) {
            for (uint32_t group = 0; group < groups; group++) {
                mark(first + group * size + rank);
            }
        }
    }
    for (uint32_t group = 0; !by_k && group < groups; group++) {
        for (uint32_t i = 0; i < kChainRangeCount; i++) {
            if (meets(chain_need(writer, next, i, 0, group))) {
                mark(first + group * size);
            }
            if (meets(chain_need(writer, next, i, 1, group))) {
                for (uint32_t rank = 1; rank < size; rank++) {
                    mark(first + group * size + rank);
                }
            }
        }
    }
    CHAIN_STAMP(chain_stamp(kChainStampMarks));
    return true;
}

// Writer: the chain, record by record (each taken in turn, `product` where this core
// is in it, the credits first). True: every record reported.
template <typename Product>
bool chain_writer(km::Writer& writer, const km_command_t& command, Product product) {
    const uint32_t base = chain_tag(command.sequence, 0);
    bool ok = true;
    CHAIN_STAMP(chain_stamp_start = km::wall());
    chain_now.ahead_on = (command.a1 & kChainAheadOff) == 0;
    chain_now.data_dram = (command.a1 & kChainDataDram) != 0;
    chain_now.ahead_tag = 0;
    for (uint32_t record = 0; ok && record < kChainRecordsMax; record++) {
        if (!chain_reached(writer, kChainReady, base + record, km::kStWriter)) {
            ok = false;
            break;
        }
        uint32_t words[kChainWords];
        chain_copy(base + record, words);
        *chain_word(kChainWriterTaken) = base + record;
        chain_enter(record, base, words);
        if (chain_member(words, writer.me())) {
            const km_command_t one = chain_command(command, words);
            CHAIN_STAMP(chain_stamp_take(chain_address(0), 1, words));
            chain_credits(writer, one);
            ok = product(one);
            CHAIN_STAMP(chain_stamp_row = 0);
            invalidate_l1_cache();
            if (*chain_word(kChainCreditDue) != 0) {  // no emit came: its credit kept
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
        km::kWhereEmit,
        [] {
            return ncrisc_noc_nonposted_atomics_flushed(noc_index);
        },
        writer.args().wait_limit
    );
    return ok && code == km::kGaveNone;
}

#if defined(SPLIT_CHAIN_STAMPS)
inline void chain_stamps_zero() {
    for (uint32_t offset = 0; offset < kChainStampBytes; offset += 4) {
        *chain_word(kChainStamps + offset) = 0;
    }
}

// Writer, SPLIT's kChainStampsMode: this core's stamps to page `me` at a0, then zero.
template <typename Writer>
bool chain_stamps_out(const Writer& writer, const km_command_t& command) {
    const InterleavedAddrGen<true> pages = {
        .bank_base_address = command.a0, .page_size = kChainStampPage
    };
    noc_async_write(
        chain_address(kChainStamps), get_noc_addr(writer.me(), pages), kChainStampBytes
    );
    if (!writer.flushed(km::kFlushWrites, km::kStWriter)) {
        return false;
    }
    chain_stamps_zero();
    return true;
}
#endif

// Reader, before its first command: the chain's words zero (nothing counts them yet).
inline void chain_start() {
    for (uint32_t offset = 0; offset < kChainSlots; offset += 4) {
        *chain_word(offset) = 0;
    }
    CHAIN_STAMP(chain_stamps_zero());
}
#endif
#else
// Compute (unpack reads, math and pack get each word by mailbox): the chain, each
// record taken in turn, `product` where this core is in it.
template <typename Product>
void chain_compute(km::Compute& compute, const km_command_t& command, Product product) {
    const uint32_t base = chain_tag(command.sequence, 0);
    CHAIN_STAMP(uint32_t stamps = 0);  // pack's: the chain's buffer, from unpack
    CHAIN_STAMP(UNPACK((compute.hand(
        ckernel::ThreadId::PackThreadId,
        ckernel::get_tile_l1_byte_address(kChainBuffer, 0)
    ))));
    CHAIN_STAMP(PACK((stamps = compute.handed(ckernel::ThreadId::UnpackThreadId))));
    CHAIN_STAMP(PACK((chain_stamp_start = km::wall())));
    for (uint32_t record = 0; record < kChainRecordsMax; record++) {
        uint32_t words[kChainWords] = {};
        uint32_t ok = 0;
        UNPACK(({
            const uint32_t at = ckernel::get_tile_l1_byte_address(kChainBuffer, 0);
            volatile tt_l1_ptr uint32_t* ready = km::l1(at + kChainReady);
            const uint32_t code = km::poll_live(
                compute.control(),
                km::kRoleUnpack,
                km::kWhereData,
                true,
                [&] {
                    return static_cast<int32_t>(*ready - (base + record)) >= 0;
                },
                compute.wait_limit(),
                0
            );
            ok = code == km::kGaveNone;
            compute.hand(ckernel::ThreadId::MathThreadId, ok);
            compute.hand(ckernel::ThreadId::PackThreadId, ok);
            volatile tt_l1_ptr uint32_t* slot = (
                km::l1(at + kChainSlots + ((base + record) % 2) * kChainRecordBytes)
            );
            for (uint32_t word = 0; ok && word < kChainWords; word++) {
                words[word] = slot[word];
                compute.hand(ckernel::ThreadId::MathThreadId, words[word]);
                compute.hand(ckernel::ThreadId::PackThreadId, words[word]);
            }
            if (ok) {
                *km::l1(at + kChainComputeTaken) = base + record;
            } else {
                km::command_gave_up(
                    compute.control(),
                    km::kStCompute,
                    code,
                    km::kWhereData,
                    0xC1u,
                    record
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
            CHAIN_STAMP(PACK((chain_stamp_take(stamps, 2, words))));
            product(chain_command(command, words));
            CHAIN_STAMP(PACK((chain_stamp_row = 0)));
        }
        if (words[kChainNext] == 0) {
            break;
        }
    }
}
#endif
