// The writer kernel: waits for the tile in L1 and writes tile_count tiles to another
// tensor, one after the other.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto destination_arguments = TensorAccessorArgs<0>();
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t destination_address = command.a2;
            const uint32_t first_tile = command.a3;
            const uint32_t tile_count = command.a4;
            const auto destination = (
                TensorAccessor(destination_arguments, destination_address)
            );
            for (uint32_t index = 0; index < tile_count; index++) {
                if (!writer.wait_front(kTileBuffer, 1)) {
                    return false;
                }
                noc_async_write_page(
                    first_tile + index, destination, get_read_ptr(kTileBuffer)
                );
                noc_async_write_barrier();
                cb_pop_front(kTileBuffer, 1);
            }
        }
        return true;
    });
    // --8<-- [end:run]
}
