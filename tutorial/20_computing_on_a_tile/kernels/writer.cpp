// The writer kernel: writes the tile of the output buffer to the destination tensor.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Destination = km::Tensor<0>;
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerAdd) {
            const uint32_t destination_address = command.a3;
            const uint32_t destination_tile = command.a4;
            const Destination destination(destination_address);
            if (!writer.wait_front(kOutput, 1)) {
                return false;
            }
            if (!km::write_pages(
                writer,
                command,
                destination,
                destination_tile,
                1,
                get_read_ptr(kOutput)
            )) {
                return false;
            }
            noc_async_write_barrier();
            cb_pop_front(kOutput, 1);
        }
        return true;
    });
}
