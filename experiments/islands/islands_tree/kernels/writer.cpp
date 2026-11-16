// Tree experiment, writer: sends its part; in an all-reduce every rank then writes the
// sum it holds, else rank 0 alone.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_tree.hpp"
#include "tree.h"

void kernel_main() {
    using Output = km::Tensor<0>;
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerAllReduce) {
            return true;
        }
        const uint32_t form = command.a2;
        const uint32_t part_tiles = command.a3;
        const bool cast = command.a4 != 0;
        const km::Group group = km::group_of(command, writer.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, part_tiles};
        if (cast) {
            if (!km::all_reduce_emit(
                writer, command, arrival, group, form, kCbOut, part_tiles
            )) {
                return false;
            }
        } else if (group.rank != 0) {
            return (
                km::reduce_emit(
                    writer, command, arrival, group, form, kCbOut, part_tiles
                )
            );
        }
        if (!km::reduce_result(writer, kCbOut, part_tiles)) {
            return false;
        }
        if (!km::write_pages(
            writer,
            command,
            Output(command.a1),
            writer.me() * part_tiles,
            part_tiles,
            get_read_ptr(kCbOut)
        )) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, part_tiles);
        return true;
    });
}
