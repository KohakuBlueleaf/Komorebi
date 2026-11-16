// The writer kernel: the sending core (rank 1) sends the tile to the receiving core
// (rank 0), which returns a number of it as the result.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/movement/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    const km::Arrival arrival = {kArrival, 1, 1};
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSend) {
            const uint32_t rank = writer.me() - km::first(command);
            if (rank == 1) {
                const uint32_t receiver = km::first(command);
                return km::emit(writer, command, arrival, kOutgoing, 1, receiver, 0, 0);
            }
            if (!writer.wait_front(kArrival, 1, km::kBehindPort)) {
                return false;
            }
            const auto tile = km::l1(get_read_ptr(kArrival));
            writer.result(tile[command.a2]);
            cb_pop_front(kArrival, 1);
        }
        return true;
    });
    // --8<-- [end:run]
}
