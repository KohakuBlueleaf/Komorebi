// The writer kernel: a core other than rank 0 sends its sum to the core it reports to;
// rank 0 writes the total to DRAM.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto destination_arguments = TensorAccessorArgs<0>();
    km::Writer writer;
    const km::Arrival arrival = {kArrival, kArrivalPages, 1};
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSumSquares) {
            const km::Group group = km::group_of(command, writer.me());
            const uint32_t form = command.a4;
            if (group.rank != 0) {
                return (
                    km::reduce_emit(writer, command, arrival, group, form, kOutput, 1)
                );
            }
            if (!km::reduce_result(writer, kOutput, 1)) {
                return false;
            }
            const auto destination = TensorAccessor(destination_arguments, command.a2);
            noc_async_write_page(command.a3, destination, get_read_ptr(kOutput));
            noc_async_write_barrier();
            cb_pop_front(kOutput, 1);
        }
        return true;
    });
    // --8<-- [end:run]
}
