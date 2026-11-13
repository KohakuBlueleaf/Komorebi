// Map/collect reader: MAP reads the core's partition of the T input tiles in blocks;
// STALL makes rank a0 wait for the stop flag.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "mapcollect.h"

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<0>();
    km::Reader reader;
    uint32_t write_page = 0;  // write position in kCbIn, in pages: a block never wraps
    reader.run([&](const km_command_t& command) {
        const uint32_t index = reader.me() - km::first(command);
        switch (km::handler(command)) {
            case kHandlerMap: {
                const uint32_t input_address = command.a0;
                const uint32_t tiles = command.a1;
                const uint32_t first_tile = command.a6;
                const km::Partition partition = (
                    km::partition(tiles, index, km::count(command))
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
                break;
            }
            case kHandlerBadItem: {  // from page `write_page`: past the end unless 0
                const uint32_t faulty_rank = command.a0;
                const uint32_t faulty_role = command.a1;
                if (
                    index == faulty_rank && faulty_role == 1 &&
                    !reader.reserve(kCbIn, kInPages)
                ) {
                    return;
                }
                break;
            }
            case kHandlerStall: {
                const uint32_t stalled_rank = command.a0;
                if (index == stalled_rank) {
                    reader.wait_stop(command);
                }
                break;
            }
            default:
                break;
        }
    });
}
