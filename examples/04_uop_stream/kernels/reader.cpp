// Reader: MAP reads the core's partition of the T input tiles in blocks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "uop_stream.h"

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<0>();
    km::Reader reader;
    uint32_t write_page = 0;  // write position in kCbIn, in pages: a block never wraps
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMap) {
            return;
        }
        const uint32_t input_address = command.a0;
        const uint32_t tiles = command.a1;
        const uint32_t first_tile = command.a4;
        const km::Partition partition = (
            km::partition(tiles, reader.me() - km::first(command), km::count(command))
        );
        const auto input = TensorAccessor(input_args, input_address);
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
            for (uint32_t tile = 0; tile < block_tiles; tile++) {
                noc_async_read_page(
                    first_tile + partition.start + done + tile,
                    input,
                    destination + tile * kTileBytes
                );
            }
            noc_async_read_barrier();
            cb_push_back(kCbIn, block_tiles);
            done += block_tiles;
            write_page = (write_page + block_tiles) % kInPages;
        }
    });
}
