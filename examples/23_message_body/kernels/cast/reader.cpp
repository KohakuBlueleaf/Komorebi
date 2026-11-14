// Cast reader: the broadcast, on NoC0. Its sender reads each body from DRAM into the
// slot and casts it; every other core of the broadcast arrives.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"
#include "cast.h"

void kernel_main() {
    using Bodies = km::Tensor<0>;
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCast) {
            return;
        }
        km::Broadcast broadcast(reader, command, 0);
        if (!broadcast.ok() || !broadcast.included()) {
            return;
        }
        const Bodies bodies(command.a1);
        const uint32_t casts = command.a0;
        for (uint32_t cast = 0; cast < casts; cast++) {
            if (broadcast.sender()) {
                const uint32_t address = broadcast.reserve(cast);
                if (address == 0) {
                    return;
                }
                if (!km::read_pages(
                    reader, command, bodies, cast % command.a2, 1, address
                )) {
                    return;
                }
                const uint32_t code = reader.poll(
                    [] {
                        return ncrisc_noc_reads_flushed(noc_index);
                    },
                    reader.args().wait_limit,
                    reader.args().wait_limit
                );
                if (code != km::kGaveNone) {
                    reader.gave_up(
                        km::kStReader,
                        code,
                        km::kWhereData,
                        km::slot(command),
                        command.sequence
                    );
                    return;
                }
                if (!broadcast.cast(cast)) {
                    return;
                }
            } else if (!broadcast.arrive(cast)) {
                return;
            }
        }
        broadcast.finish();
    });
}
