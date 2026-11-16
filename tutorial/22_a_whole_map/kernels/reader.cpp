// The reader kernel: for each tile of this core's share, copies tile i of the first
// half of the source and tile i of the second half into the two input buffers.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerMap) {
            const uint32_t source_address = command.a0;
            const uint32_t tile_count = command.a1;
            const uint32_t rank = reader.me() - km::first(command);
            const km::Partition share = (
                km::partition(tile_count, rank, km::count(command))
            );
            const Source source(source_address);
            for (uint32_t index = 0; index < share.len; index++) {
                const uint32_t tile = share.start + index;
                if (
                    !reader.reserve(kFirstInput, 1) || !reader.reserve(kSecondInput, 1)
                ) {
                    return;
                }
                if (!km::read_pages(
                    reader, command, source, tile, 1, get_write_ptr(kFirstInput)
                )) {
                    return;
                }
                if (!km::read_pages(
                    reader,
                    command,
                    source,
                    tile_count + tile,
                    1,
                    get_write_ptr(kSecondInput)
                )) {
                    return;
                }
                noc_async_read_barrier();
                cb_push_back(kFirstInput, 1);
                cb_push_back(kSecondInput, 1);
            }
        }
    });
    // --8<-- [end:run]
}
