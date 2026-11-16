// Casters reader (NoC0): casters cast their parts by the counted VALID; every core
// checks every word and the count beside it, then writes its result row.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "casters.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "counted.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    using Results = km::Tensor<0>;
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCast) {
            return;
        }
        km::counted::Broadcast broadcast(reader, command, kArgBroadcast);
        if (!broadcast.ok() || !broadcast.included()) {
            return;
        }
        const uint32_t blocks = command.a1, first = command.a2;
        const auto arg = [&](uint32_t offset) {
            return reader.arg(kArgBroadcast + offset / 4);
        };
        const uint32_t casters = arg(offsetof(km::counted::BroadcastArgs, casters));
        const uint32_t area = (
            reader.args().control + KM_CONTROL_BROADCAST +
            arg(offsetof(km::counted::BroadcastArgs, broadcast)) % KM_BROADCASTS *
                KM_BROADCAST_BYTES
        );
        const LocalCBInterface& buffer = get_local_cb_interface(kCbBlocks);
        const uint32_t block_bytes = broadcast.pages() * buffer.fifo_page_size;
        const uint32_t words = block_bytes / 4, window = broadcast.window();
        // this core's words of a block: its part (counted), all (the sender), none
        const km::Partition mine = broadcast.counted()
            ? km::partition(words, broadcast.rank(), casters)
            : km::Partition{0, broadcast.sender() ? words : 0};
        uint32_t wrong = 0, first_block = kNone, first_place = kNone;
        uint32_t counts_out = 0, count_high = 0;
        for (uint32_t block = 0; block < blocks; block++) {
            const uint32_t index = first + block;
            const uint32_t slot = (
                buffer.fifo_limit - buffer.fifo_size + block % window * block_bytes
            );
            if (broadcast.casts() || broadcast.sender()) {
                if (broadcast.reserve(block) != slot) {
                    return;
                }
                volatile tt_l1_ptr uint32_t* words_at = km::l1(slot);
                for (
                    uint32_t place = mine.start; place < mine.start + mine.len; place++
                ) {
                    words_at[place] = expected(index, place);
                }
                if (broadcast.counted()) {
                    if (!broadcast.part(
                        block, slot + 4 * mine.start, 4 * mine.start, 4 * mine.len
                    )) {
                        return;
                    }
                }
                if (!broadcast.cast(block)) {
                    return;
                }
            } else if (!broadcast.arrive(block)) {
                return;
            }
            // the block stays: its slot is cast again only after this core's DONE-BY
            // for it, which this kernel sends in its next broadcast call
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* words_at = km::l1(slot);
            for (uint32_t place = 0; place < words; place++) {
                if (words_at[place] != expected(index, place) && wrong++ == 0) {
                    first_block = index;
                    first_place = place;
                }
            }
            if (broadcast.counted()) {  // blocks 0 to block here; at most a window on
                const uint32_t count = km::l1(area + KM_BROADCAST_COUNT)[0];
                if (
                    count < casters * (block + 1) || count > casters * (block + window)
                ) {
                    counts_out++;
                }
                count_high = count > count_high ? count : count_high;
            }
        }
        if (!broadcast.finish()) {
            return;
        }
        const auto row = (
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kCbResult))
        );
        row[0] = wrong;
        row[1] = first_block;
        row[2] = first_place;
        row[3] = broadcast.blocks();
        row[4] = counts_out;
        row[5] = count_high;
        row[6] = km::l1(area + KM_BROADCAST_COUNT)[0];  // cleared by finish()
        row[7] = broadcast.rank();
        const Results results(command.a0);
        const uint32_t rank = reader.me() - km::first(command);
        if (km::write_pages(
            reader, command, results, command.a3 + rank, 1, get_write_ptr(kCbResult)
        )) {
            reader.poll(
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                reader.args().wait_limit,
                reader.args().wait_limit
            );
        }
    });
}
