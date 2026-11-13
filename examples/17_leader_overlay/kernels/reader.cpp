// The reader: each round the leader picks an overlay and casts its number; every core
// loads and calls it. LEADER_COMPILED: the operations compiled in, the word picks one.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"
#if !defined(LEADER_COMPILED)
#include "komorebi/device/km_overlay.hpp"
#endif
#include "leader_overlay.h"
#include "ops.h"

// A round's cycles and what it loaded (kLoaded* bits).
struct Round {
    uint32_t load = 0;
    uint32_t call = 0;
    uint32_t loaded = 0;
};

constexpr uint32_t kLoadedReader = 1;   // the reader part was read into a slot
constexpr uint32_t kLoadedCompute = 2;  // compute's parts were read: the meeting

// Timed wait until this kernel's reads (or writes) have landed; gives up otherwise.
inline bool landed(const km::Reader& reader, const km_command_t& command, bool writes) {
    const uint32_t code = reader.poll(
        [&] {
            return (
                writes
                    ? ncrisc_noc_nonposted_writes_flushed(noc_index)
                    : ncrisc_noc_reads_flushed(noc_index)
            );
        },
        reader.args().wait_limit,
        reader.args().wait_limit
    );
    if (code != km::kGaveNone) {
        reader.gave_up(
            km::kStReader, code, km::kWhereData, km::slot(command), command.sequence
        );
    }
    return code == km::kGaveNone;
}

// Hands the cast word to this core's compute; 0 ends compute's rounds.
inline bool hand(km::Reader& reader, uint32_t word) {
    if (!reader.reserve(kCbChoice, 1)) {
        return false;
    }
    km::l1(get_write_ptr(kCbChoice))[0] = word;
    cb_push_back(kCbChoice, 1);
    return true;
}

#if defined(LEADER_COMPILED)
// The operation the word names, compiled in.
inline bool take(km::Reader& reader, uint32_t word, uint32_t data, Round& round) {
    const uint64_t started = km::wall64();
    uint32_t done = 0;
    switch (word & 3u) {
        case 0:
            done = op_mix(km::l1(data), index_of(word));
            break;
        case 1:
            done = op_turn(km::l1(data), index_of(word));
            break;
        default: done = op_tile(reader, km::l1(data));
    }
    round.call = static_cast<uint32_t>(km::wall64() - started);
    return done != 0;
}
#else
// --8<-- [start:take]
// The overlay the word names: loaded unless the tile holds it, then called.
inline bool take(km::Reader& reader, uint32_t word, uint32_t data, Round& round) {
    const uint32_t number = overlay_of(word);
    km::Overlay overlay(reader, number);
    const uint32_t control = reader.args().control;
    round.loaded = (
        (overlay.held() ? 0 : kLoadedReader) |
        (
            has_compute(word) && !km::overlay_members_held(control, number)
                ? kLoadedCompute
                : 0
        )
    );
    const uint64_t started = km::wall64();
    if (
        !overlay.load()
    ) {  // reads what the tile lacks; meets compute if it reads theirs
        return false;
    }
    const uint64_t loaded = km::wall64();
    const uint32_t done = overlay.call(DataArgument{data, index_of(word)});
    round.load = static_cast<uint32_t>(loaded - started);
    round.call = static_cast<uint32_t>(km::wall64() - loaded);
    return done != 0;
}

// --8<-- [end:take]
#endif

inline uint32_t saturate(uint32_t value, uint32_t limit) {
    return value < limit ? value : limit;
}

void kernel_main() {
    constexpr auto values_args = TensorAccessorArgs<0>();
    constexpr uint32_t results_offset = values_args.next_compile_time_args_offset();
    constexpr auto results_args = TensorAccessorArgs<results_offset>();
    constexpr uint32_t records_offset = results_args.next_compile_time_args_offset();
    constexpr auto records_args = TensorAccessorArgs<records_offset>();
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLead) {
            return;
        }
        km::Broadcast broadcast(reader, command, kArgWords);
        if (!broadcast.ok() || !broadcast.included()) {
            return;
        }
        const auto values = TensorAccessor(values_args, command.a0);
        const uint32_t rounds = command.a1;
        const uint32_t first = command.a3;
        const bool timing = command.a5 != 0;
        const uint32_t stride = command.a7 == 0 ? 1 : command.a7;
        const auto page = km::l1(get_write_ptr(kCbValues));
        const uint32_t data = get_write_ptr(kCbData);
        volatile tt_l1_ptr uint32_t* row = km::l1(get_write_ptr(kCbResult));
        volatile tt_l1_ptr uint32_t* times = km::l1(get_write_ptr(kCbRecords));
        for (uint32_t k = 0; k < kDataWords; k++) {
            km::l1(data)[k] = data_start(reader.me(), first, k);
        }
        for (uint32_t word = kDataWords; word < kResultWords; word++) {
            row[word] = 0;
        }
        row[kResultWrong + 1] = kNone;
        uint32_t timed = 0;
        // --8<-- [start:round]
        for (uint32_t cast = 0; cast < rounds; cast++) {
            const uint32_t index = first + cast;
            uint64_t begun = 0;
            if (broadcast.sender()) {
                if (cast == 0 || index % kValuesPage == 0) {
                    noc_async_read_page(
                        index / kValuesPage, values, get_write_ptr(kCbValues)
                    );
                    if (!landed(reader, command, false)) {
                        return;
                    }
                    invalidate_l1_cache();
                }
                const uint32_t choice = pick(page[index % kValuesPage]);
                begun = km::wall64();
                if (!broadcast.cast(cast, choice_word(choice, index))) {
                    return;
                }
            } else {
                begun = km::wall64();
                if (!broadcast.arrive(cast)) {
                    return;
                }
            }
            const uint64_t arrived = km::wall64();
            const uint32_t word = broadcast.word();  // the leader's choice, everywhere
            if (index_of(word) != index) {
                if (row[kResultWrong]++ == 0) {
                    row[kResultWrong + 1] = cast;
                }
                continue;
            }
            if (has_compute(word) && !hand(reader, word)) {
                return;
            }
            Round round;
            if (!take(reader, word, data, round)) {
                return;
            }
            // --8<-- [end:round]
            row[kResultWrong + 3] += round.loaded & kLoadedReader;
            row[kResultWrong + 4] += (round.loaded & kLoadedCompute) >> 1;
            if (timing && cast % stride == 0 && timed < kRecordRounds) {
                volatile tt_l1_ptr uint32_t* record = times + kRecordWords * timed++;
                record[0] = static_cast<uint32_t>(begun);
                record[1] = (
                    saturate(static_cast<uint32_t>(arrived - begun), 0xFFFF) |
                    saturate(round.call, 0xFFFF) << 16
                );
                record[2] = (
                    saturate(round.load, 0xFFFFFF) | overlay_of(word) << 24 |
                    round.loaded << 28
                );
                record[3] = static_cast<uint32_t>(km::wall64());
            }
        }
        if (!hand(reader, 0) || !broadcast.finish()) {
            return;
        }
        for (uint32_t k = 0; k < kDataWords; k++) {
            row[k] = km::l1(data)[k];
        }
        row[kResultWrong + 2] = broadcast.blocks();
        const uint32_t rank = reader.me() - km::first(command);
        const auto results = TensorAccessor(results_args, command.a2);
        noc_async_write_page(command.a4 + rank, results, get_write_ptr(kCbResult));
        if (timing) {
            for (
                uint32_t word = kRecordWords * timed;
                word < kRecordWords * kRecordRounds;
                word++
            ) {
                times[word] = 0;
            }
            const auto records = TensorAccessor(records_args, command.a5);
            noc_async_write_page(command.a6 + rank, records, get_write_ptr(kCbRecords));
        }
        landed(reader, command, true);
    });
}
