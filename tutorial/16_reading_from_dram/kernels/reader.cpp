// The reader kernel: copies one tile from DRAM into the buffer in L1.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerFetch) {
            const uint32_t source_address = command.a0;
            const uint32_t tile_number = command.a1;
            const Source source(source_address);
            if (!reader.reserve(kTileBuffer, 1)) {
                return;
            }
            const uint32_t where = get_write_ptr(kTileBuffer);
            if (!km::read_pages(reader, command, source, tile_number, 1, where)) {
                return;
            }
            noc_async_read_barrier();
            cb_push_back(kTileBuffer, 1);
        }
    });
    // --8<-- [end:run]
}
