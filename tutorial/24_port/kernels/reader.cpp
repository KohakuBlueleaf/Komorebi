// The reader kernel: the sending core (rank 1) reads a tile from DRAM; the receiving
// core (rank 0) waits for the tile to arrive.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto source_arguments = TensorAccessorArgs<0>();
    km::Reader reader;
    const km::Arrival arrival = {kArrival, 1, 1};
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSend) {
            const uint32_t rank = reader.me() - km::first(command);
            if (rank == 1) {
                const auto source = TensorAccessor(source_arguments, command.a0);
                if (!reader.reserve(kOutgoing, 1)) {
                    return;
                }
                noc_async_read_page(command.a1, source, get_write_ptr(kOutgoing));
                noc_async_read_barrier();
                cb_push_back(kOutgoing, 1);
            } else {
                km::arrive(reader, command, arrival, 0, 1);
            }
        }
    });
    // --8<-- [end:run]
}
