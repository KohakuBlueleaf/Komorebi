// The reader kernel: the sending core (rank 1) reads a tile from DRAM; the receiving
// core (rank 0) waits for the tile to arrive.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    const km::Arrival arrival = {kArrival, 1, 1};
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSend) {
            const uint32_t rank = reader.me() - km::first(command);
            if (rank == 1) {
                const Source source(command.a0);
                if (!reader.reserve(kOutgoing, 1)) {
                    return;
                }
                if (!km::read_pages(
                    reader, command, source, command.a1, 1, get_write_ptr(kOutgoing)
                )) {
                    return;
                }
                noc_async_read_barrier();
                cb_push_back(kOutgoing, 1);
            } else {
                km::arrive(reader, command, arrival, 0, 1);
            }
        }
    });
    // --8<-- [end:run]
}
