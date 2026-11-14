// The reader kernel: copies tile_count tiles from DRAM, one after the other, through
// the buffer in L1.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t source_address = command.a0;
            const uint32_t first_tile = command.a1;
            const uint32_t tile_count = command.a4;
            const Source source(source_address);
            for (uint32_t index = 0; index < tile_count; index++) {
                if (!reader.reserve(kTileBuffer, 1)) {
                    return;
                }
                if (!km::read_pages(
                    reader,
                    command,
                    source,
                    first_tile + index,
                    1,
                    get_write_ptr(kTileBuffer)
                )) {
                    return;
                }
                noc_async_read_barrier();
                cb_push_back(kTileBuffer, 1);
            }
        }
    });
    // --8<-- [end:run]
}
