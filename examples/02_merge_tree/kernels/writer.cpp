// Merge-tree writer: rank 0 writes the total from page first_page on (its checksum the
// result word if asked); the other ranks send their parts, but stalled_rank stalls.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "merge.h"

void kernel_main() {
    using Output = km::Tensor<0>;
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMerge) {
            return true;
        }
        const uint32_t output_address = command.a2;
        const uint32_t first_page = command.a3;
        const uint32_t form = command.a4;
        const uint32_t stalled_rank = command.a5;
        const uint32_t part_tiles = command.a7;
        const km::Group group = km::group_of(command, writer.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, part_tiles};
        if (
            group.rank == stalled_rank
        ) {  // a test: waits for the stop flag, sends nothing
            const uint32_t code = writer.poll(
                [] {
                    return false;
                },
                writer.args().idle_limit
            );
            writer.gave_up(
                km::kStWriter,
                code,
                km::kWhereStall,
                km::slot(command),
                command.sequence
            );
            return false;
        }
        if (group.rank != 0) {
            return (
                km::reduce_emit(
                    writer, command, arrival, group, form, kCbOut, part_tiles
                )
            );
        }
        if (!km::reduce_result(writer, kCbOut, part_tiles)) {
            return false;
        }
        // The command's result word, only when it asks for one: the total's 32-bit
        // words summed (a checksum: one more pass over the total, so only if asked).
        if (km::flags(command) & KM_FLAG_RESULT) {
            const volatile tt_l1_ptr uint32_t*
                words = (
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kCbOut))
                );
            uint32_t checksum = 0;
            for (uint32_t word = 0; word < part_tiles * kTileBytes / 4; word++) {
                checksum += words[word];
            }
            writer.result(checksum);
        }
        const Output output(output_address);  // its reference word
        // the pages another island holds are put there and waited for
        if (!km::write_pages(
            writer, command, output, first_page, part_tiles, get_read_ptr(kCbOut)
        )) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, part_tiles);
        return true;
    });
}
