// Island-aware sum, writer: a core sends its part inside its island (km::local), a
// leader of another island to the first leader, which writes the total.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_port.hpp"
#include "sum.h"

void kernel_main() {
    using Output = km::Tensor<0>;
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerSum) {
            return true;
        }
        const uint32_t output_address = command.a2;
        const uint32_t first_page = command.a3;
        const uint32_t form = command.a4;
        const uint32_t part_tiles = command.a7;
        const km::Group group = km::group_of(command, writer.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, part_tiles};
        const km::Group leaders = group.leaders();
        if (group.local().rank != 0) {  // inside my island
            return (
                km::reduce_emit(
                    writer, command, arrival, group, form, kCbOut, part_tiles, km::local
                )
            );
        }
        if (leaders.rank != 0) {  // my island's part, to the first leader's island
            const uint32_t root = group.leader(0);
            return (
                km::to_island(km::island_of(root))
                    .put(
                        writer,
                        command,
                        arrival,
                        kCbOut,
                        part_tiles,
                        root,
                        kCrossPort,
                        cross_slot(part_tiles, leaders.rank)
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
            writer,
            command,
            output,
            first_page,
            part_tiles,
            get_read_ptr(kCbOut),
            kPutPort
        )) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, part_tiles);
        return true;
    });
}
