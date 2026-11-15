// SPLIT's chain (SPLIT_CHAIN): each kernel runs the records of one command in turn;
// a record's input waits for the marks of the record before's roots, not the thread.
#pragma once

#include <stdint.h>

#include "chain.h"
#include "komorebi/device/km_group.hpp"
#include "split.h"

// The chain's words in buffer kChainBuffer (bytes): marks and credits that other cores
// count up (16-byte slots), the record ring, the counts each kernel has waited for.
constexpr uint32_t kChainMarks = 0;
constexpr uint32_t kChainCredits = 16;
constexpr uint32_t kChainReady = 32;           // the reader: the last record read's tag
constexpr uint32_t kChainWriterTaken = 48;     // the writer: the last record it took
constexpr uint32_t kChainComputeTaken = 64;    // unpack: the last record compute took
constexpr uint32_t kChainReaderWanted = 80;    // marks the reader waited for, in all
constexpr uint32_t kChainWriterWanted = 96;    // marks the writer waited for
constexpr uint32_t kChainCreditsWanted = 100;  // credits the writer waited for
constexpr uint32_t kChainCreditDue = 104;  // 1: this record's emit waits for a credit
constexpr uint32_t kChainSlots = 128;      // record r in slot r % 2
constexpr uint32_t kChainBytes = kChainSlots + 2 * kChainRecordBytes;

// A record's tag in the ring: the chain command's sequence and its index.
inline uint32_t chain_tag(uint32_t sequence, uint32_t record) {
    return sequence * kChainRecordsMax + record;
}

// The command a record names: the chain command's header and sequence, its words.
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
    return command;
}

inline bool chain_member(const uint32_t* record, uint32_t me) {
    const uint32_t first = record[kChainGroup] & 0xFFFFu;
    return me >= first && me - first < (record[kChainGroup] >> 16);
}

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
    volatile tt_l1_ptr uint32_t* wanted = chain_word(kChainCreditsWanted);
    const uint32_t target = *wanted + 1;
    *wanted = target;
    const uint32_t code = writer.poll_at(
        km::kWhereEmit,
        [&] {
            invalidate_l1_cache();
            return static_cast<int32_t>(*chain_word(kChainCredits) - target) >= 0;
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

// The running chain on this mover: the record, the one before it (its input's
// writer), the record's index and its tag's base (not in a chain: off).
struct ChainNow {
    bool on;
    bool has_before;
    uint32_t record;
    uint32_t tag_base;
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
    return code == km::kGaveNone;
}

// Until the record's input is marked: its W blocks into this mover's half once A
// has freed it (the half empty), one block a pass. `skip`: the chunks W's stream
// starts after (this mover's next block). False: gave up (recorded).
template <typename Kernel, typename Accessor>
bool chain_weight_first(
    const Kernel& core, const km::Stream<Accessor>& weights, uint32_t& skip
) {
    skip = 0;
    uint32_t target = 0;
    if (!chain_target(core, target)) {
        return true;
    }
    km::MoveCursor cursor;
    uint32_t filled = 0, drained = 0;
    if (!km::move_begin(weights, cursor, filled, drained)) {
        core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
        return false;
    }
    const uint32_t half = weights.pair.half(km::kMover);
    const uint64_t limit = static_cast<uint64_t>(core.args().wait_limit) << 10;
    uint64_t deadline = km::wall64() + limit;
    bool freed = false;
    for (uint32_t pass = 0; !chain_marked(target); pass++) {
        freed = (
            freed ||
            cb_pages_reservable_at_back(
                half, get_local_cb_interface(half).fifo_num_pages
            )
        );
        bool issued = false, open = false, reads = false, writes = false;
        if (
            freed && !km::move_offer(core, weights, cursor, issued, open, reads, writes)
        ) {
            core.gave_up(km::kStMover, km::kGaveBadItem, km::kWhereData, 0, 0);
            return false;
        }
        if (issued) {
            if (!km::move_settle(core, true, false)) {
                return false;
            }
            km::Moved moved = {0, 0};
            km::move_hand_on(weights, cursor, issued, moved);
            deadline = km::wall64() + limit;
            continue;
        }
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
    skip = cursor.block < cursor.blocks ? cursor.block : cursor.blocks;
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
    for (uint32_t record = 0; record < kChainRecordsMax; record++) {
        uint32_t words[kChainWords];
        chain_copy(base + record, words);
        const uint32_t next = words[kChainNext];
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
        chain_enter(record, base, words);
        if (chain_member(words, reader.me())) {
            product(chain_command(command, words));
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
                    chain_address(kChainCredits)
                ),
                1
            );
        }
    }
    *chain_word(kChainCreditDue) = group.rank != 0 ? 1u : 0u;
}

// A group's root writer, once its output is written: the writes acknowledged, then a
// mark on each core of the next record for each of its ranges that meets this block.
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
    if (!writer.flushed(km::kFlushWrites, km::kStWriter)) {
        return false;
    }
    const uint32_t width = chain_width(writer, now);
    const uint32_t low = (
        (writer.me() - km::first(command)) / chain_plan(writer, now, kSpS) * width
    );
    const uint32_t high = low + width;
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
    return true;
}

// Writer: the chain, record by record (each taken in turn, `product` where this core
// is in it, the credits first). True: every record reported.
template <typename Product>
bool chain_writer(km::Writer& writer, const km_command_t& command, Product product) {
    const uint32_t base = chain_tag(command.sequence, 0);
    bool ok = true;
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
            chain_credits(writer, one);
            ok = product(one);
            invalidate_l1_cache();
            if (*chain_word(kChainCreditDue) != 0) {  // no emit came: its credit kept
                *chain_word(kChainCreditDue) = 0;
                *chain_word(kChainCreditsWanted) += 1;
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

// Reader, before its first command: the chain's words zero (nothing counts them yet).
inline void chain_start() {
    for (uint32_t offset = 0; offset < kChainSlots; offset += 4) {
        *chain_word(offset) = 0;
    }
}
#endif
#else
// Compute (unpack reads, math and pack get each word by mailbox): the chain, each
// record taken in turn, `product` where this core is in it.
template <typename Product>
void chain_compute(km::Compute& compute, const km_command_t& command, Product product) {
    const uint32_t base = chain_tag(command.sequence, 0);
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
            product(chain_command(command, words));
        }
        if (words[kChainNext] == 0) {
            break;
        }
    }
}
#endif
