// The writer kernel: waits for the tile in L1 and writes it to a tile of another
// tensor.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Destination = km::Tensor<0>;
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t destination_address = command.a2;
            const uint32_t destination_tile = command.a3;
            const Destination destination(destination_address);
            if (!writer.wait_front(kTileBuffer, 1)) {
                return false;
            }
            if (!km::write_pages(
                writer,
                command,
                destination,
                destination_tile,
                1,
                get_read_ptr(kTileBuffer)
            )) {
                return false;
            }
            noc_async_write_barrier();
            cb_pop_front(kTileBuffer, 1);
        }
        return true;
    });
    // --8<-- [end:run]
}
