// Chain example, reader: a step's tiles read once its input is marked, worked on in
// L1 by this mover (int32), handed to the writer; SUM adds its tree's parts first.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_chain.hpp"
#include "komorebi/device/km_port.hpp"
#include "steps.h"

namespace {

    using Tiles = km::Tensor<0>;

    // Page `page` of the tensor at `address` into `to`. False: gave up.
    bool read_tile(
        const km::Reader& reader,
        const km_command_t& command,
        uint32_t address,
        uint32_t page,
        uint32_t to
    ) {
        if (!km::read_pages(reader, command, Tiles(address), page, 1, to)) {
            return false;
        }
        noc_async_read_barrier();
        return true;
    }

    void add_words(uint32_t to, uint32_t from) {
        volatile tt_l1_ptr int32_t*
            sum = (reinterpret_cast<volatile tt_l1_ptr int32_t*>(to));
        volatile tt_l1_ptr int32_t*
            part = (reinterpret_cast<volatile tt_l1_ptr int32_t*>(from));
        for (uint32_t word = 0; word < kTileWords; word++) {
            sum[word] += part[word];
        }
    }

    // SCALE or ADD: this core's `width` tiles from tile index x width, one at a time.
    void map_tiles(km::Reader& reader, const km_command_t& command) {
        const uint32_t width = command.a6;
        const uint32_t first = (reader.me() - km::first(command)) * width;
        for (uint32_t tile = first; tile < first + width; tile++) {
            if (!reader.reserve(kCbOut, 1)) {
                return;
            }
            const uint32_t out = get_write_ptr(kCbOut);
            if (!read_tile(reader, command, command.a1, tile, out)) {
                return;
            }
            if (km::handler(command) == kHandlerAdd) {
                const uint32_t other = get_write_ptr(kCbOther);
                if (!read_tile(reader, command, command.a3, tile, other)) {
                    return;
                }
                add_words(out, other);
            } else {
                volatile tt_l1_ptr int32_t*
                    words = (reinterpret_cast<volatile tt_l1_ptr int32_t*>(out));
                for (uint32_t word = 0; word < kTileWords; word++) {
                    words[word] *= static_cast<int32_t>(command.a5);
                }
            }
            cb_push_back(kCbOut, 1);
        }
    }

    // SUM: this rank's share of the a3 input tiles summed, then the part of each round
    // whose sender this rank receives from, in round order.
    void sum_tiles(km::Reader& reader, const km_command_t& command) {
        const km::Group group = km::group_of(command, reader.me());
        const km::Partition share = km::partition(command.a3, group.rank, group.size);
        if (!reader.reserve(kCbOut, 1)) {
            return;
        }
        const uint32_t out = get_write_ptr(kCbOut);
        volatile tt_l1_ptr int32_t*
            sum = (reinterpret_cast<volatile tt_l1_ptr int32_t*>(out));
        for (uint32_t word = 0; word < kTileWords; word++) {
            sum[word] = 0;
        }
        const uint32_t scratch = get_write_ptr(kCbIn);
        for (uint32_t tile = share.start; tile < share.start + share.len; tile++) {
            if (!read_tile(reader, command, command.a1, tile, scratch)) {
                return;
            }
            add_words(out, scratch);
        }
        const km::Arrival arrival = {kCbArrive, kArrivePages, 1};
        for (uint32_t round = 0; round < km::kTreeRounds; round++) {
            if (km::tree_receives(group.rank, group.size, round)) {
                if (!km::port_wait(reader, command, round, 1)) {
                    return;
                }
                add_words(out, km::slot_addr(arrival, round));
            }
        }
        cb_push_back(kCbOut, 1);
    }

    // A step: in a chain's record, once its input is marked.
    void step(km::Reader& reader, const km_command_t& command) {
        if (!step_handler(km::handler(command))) {
            return;
        }
        if (!km::chain_input(km::ChainShape{}, reader)) {  // a chain's record: marked
            return;
        }
        if (km::handler(command) == kHandlerSum) {
            sum_tiles(reader, command);
        } else {
            map_tiles(reader, command);
        }
    }

    // A command: a chain command's records in turn, or one step.
    void command_of(km::Reader& reader, const km_command_t& command) {
        const auto product = [&](const km_command_t& record) {
            step(reader, record);
        };
        if (step_handler(km::handler(command)) && command.a4 == kChainMode) {
            km::chain_reader(reader, command, product);
        } else {
            product(command);
        }
    }

}  // namespace

#if !defined(KM_OVERLAY_PART)  // the runtime's form calls command_of (runtime/)
void kernel_main() {
    km::Reader reader;
    km::chain_start();
    reader.run([&](const km_command_t& command) {
        command_of(reader, command);
    });
}
#endif
