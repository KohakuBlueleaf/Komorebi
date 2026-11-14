// Merge-tree reader: reads the core's partition of the T input tiles in blocks, then
// waits for the parts this core combines (by owner or by tree).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_port.hpp"
#include "merge.h"

void kernel_main() {
    using Input = km::Tensor<0>;
    km::Reader reader;
    uint32_t write_page = 0;  // write position in kCbIn, in pages: a block never wraps
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMerge) {
            return;
        }
        const uint32_t input_address = command.a0;
        const uint32_t tiles = command.a1;
        const uint32_t form = command.a4;
        const uint32_t first_tile = command.a6;
        const uint32_t part_tiles = command.a7;
        const km::Group group = km::group_of(command, reader.me());
        const km::Partition partition = km::partition(tiles, group.rank, group.size);
        const Input input(input_address);  // its reference word
        for (uint32_t done = 0; done < partition.len;) {
            uint32_t block_tiles = (
                partition.len - done < kBlock ? partition.len - done : kBlock
            );
            block_tiles = (
                block_tiles < kInPages - write_page
                    ? block_tiles
                    : kInPages - write_page
            );
            if (!reader.reserve(kCbIn, block_tiles)) {
                return;
            }
            const uint32_t destination = get_write_ptr(kCbIn);
            // the tiles another island holds are asked for, then waited for
            if (!km::read_pages(
                reader,
                command,
                input,
                first_tile + partition.start + done,
                block_tiles,
                destination
            )) {
                return;
            }
            noc_async_read_barrier();
            cb_push_back(kCbIn, block_tiles);
            done += block_tiles;
            write_page = (write_page + block_tiles) % kInPages;
        }
        km::reduce_arrive(
            reader, command, {kCbArrive, kArrivePages, part_tiles}, group, form
        );
    });
}
