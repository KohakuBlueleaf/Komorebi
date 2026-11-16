// The reader kernel: reads the tiles of the core's share of the source, one after the
// other, then takes part in the reduction.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    const km::Arrival arrival = {kArrival, kArrivalPages, 1};
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSumTiles) {
            const km::Group group = km::group_of(command, reader.me());
            const uint32_t tile_count = command.a5;
            const km::Partition share = (
                km::partition(tile_count, group.rank, group.size)
            );
            const Source source(command.a0);
            for (uint32_t index = 0; index < share.len; index++) {
                if (!reader.reserve(kOwnTiles, 1)) {
                    return;
                }
                if (!km::read_pages(
                    reader,
                    command,
                    source,
                    command.a1 + share.start + index,
                    1,
                    get_write_ptr(kOwnTiles)
                )) {
                    return;
                }
                noc_async_read_barrier();
                cb_push_back(kOwnTiles, 1);
            }
            km::reduce_arrive(reader, command, arrival, group, 0);
        }
    });
    // --8<-- [end:run]
}
