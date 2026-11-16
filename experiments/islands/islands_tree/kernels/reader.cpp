// Tree experiment, reader: the rank's P input tiles, then the reduce's arrivals and
// (an all-reduce) the sum cast back to it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_tree.hpp"
#include "tree.h"

void kernel_main() {
    using Input = km::Tensor<0>;
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerAllReduce) {
            return;
        }
        const uint32_t form = command.a2;
        const uint32_t part_tiles = command.a3;
        const km::Group group = km::group_of(command, reader.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, part_tiles};
        if (!reader.reserve(kCbIn, part_tiles)) {
            return;
        }
        if (!km::read_pages(
            reader,
            command,
            Input(command.a0),
            reader.me() * part_tiles,
            part_tiles,
            get_write_ptr(kCbIn)
        )) {
            return;
        }
        noc_async_read_barrier();
        cb_push_back(kCbIn, part_tiles);
        if (command.a4 != 0) {
            km::all_reduce_arrive(reader, command, arrival, group, form);
        } else {
            km::reduce_arrive(reader, command, arrival, group, form);
        }
    });
}
