// The writer kernel: waits for the tile in L1 and sends one number of it back.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerFetch) {
            const uint32_t position = command.a2;
            if (!writer.wait_front(kTileBuffer, 1)) {
                return false;
            }
            const auto tile = (
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_read_ptr(kTileBuffer)
                )
            );
            writer.result(tile[position]);
            cb_pop_front(kTileBuffer, 1);
        }
        return true;
    });
    // --8<-- [end:run]
}
