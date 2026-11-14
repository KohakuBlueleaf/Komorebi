// The reader kernel: reads the core's own tile (tile number rank), then waits for the
// tiles that the other cores send to it.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    const km::Arrival arrival = {kArrival, kArrivalPages, 1};
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerReduce) {
            const km::Group group = km::group_of(command, reader.me());
            const uint32_t form = command.a4;
            const Source source(command.a0);
            if (!reader.reserve(kOwnTile, 1)) {
                return;
            }
            if (!km::read_pages(
                reader,
                command,
                source,
                command.a1 + group.rank,
                1,
                get_write_ptr(kOwnTile)
            )) {
                return;
            }
            noc_async_read_barrier();
            cb_push_back(kOwnTile, 1);
            km::reduce_arrive(reader, command, arrival, group, form);
        }
    });
    // --8<-- [end:run]
}
