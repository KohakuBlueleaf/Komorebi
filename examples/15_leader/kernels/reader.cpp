// Leader reader: the broadcast, on NoC0. The leader of each broadcast reads a value,
// picks what its group does next and casts the choice; every core of it does that.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"
#include "leader.h"

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

void kernel_main() {
    constexpr auto values_args = TensorAccessorArgs<0>();
    constexpr uint32_t tiles_offset = values_args.next_compile_time_args_offset();
    constexpr auto tiles_args = TensorAccessorArgs<tiles_offset>();
    constexpr uint32_t results_offset = tiles_args.next_compile_time_args_offset();
    constexpr auto results_args = TensorAccessorArgs<results_offset>();
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerLead && handler != kHandlerCarry) {
            return;
        }
        const bool carry = handler == kHandlerCarry;
        km::Broadcast broadcast(reader, command, carry ? kArgTiles : kArgWords);
        if (!broadcast.ok() || !broadcast.included()) {
            return;
        }
        const auto values = TensorAccessor(values_args, command.a0);
        const auto tiles = TensorAccessor(tiles_args, command.a5);
        const uint32_t casts = command.a1;
        const uint32_t first = command.a3;
        const auto page = (
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kCbValues))
        );
        uint32_t state = first;
        uint32_t wrong = 0;
        uint32_t first_wrong = kNone;
        // --8<-- [start:lead]
        for (uint32_t cast = 0; cast < casts; cast++) {
            const uint32_t index = first + cast;
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
                if (carry) {
                    const uint32_t address = broadcast.reserve(cast);
                    if (address == 0) {
                        return;
                    }
                    noc_async_read_page(index % command.a6, tiles, address);
                }
                if (!broadcast.cast(cast, choice_word(choice, index))) {
                    return;
                }
            } else if (!broadcast.arrive(cast)) {
                return;
            }
            const uint32_t word = broadcast.word();  // the leader's choice, everywhere
            if (word >> 2 != index + 1 && wrong++ == 0) {
                first_wrong = cast;
            }
            state = step(word, state, index);
        }
        if (!broadcast.finish()) {
            return;
        }
        // --8<-- [end:lead]
        const auto row = (
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kCbResult))
        );
        row[0] = state;
        row[1] = wrong;
        row[2] = first_wrong;
        row[3] = broadcast.blocks();
        const auto results = TensorAccessor(results_args, command.a2);
        const uint32_t rank = reader.me() - km::first(command);
        noc_async_write_page(command.a4 + rank, results, get_write_ptr(kCbResult));
        landed(reader, command, true);
    });
}
