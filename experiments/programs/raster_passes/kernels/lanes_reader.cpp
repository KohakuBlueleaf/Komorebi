// Lanes probe reader: streams a command's input pages from DRAM into the input buffer.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "lanes.h"
#include "raster.h"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerTile) {
            return;
        }
        const InterleavedAddrGen<true> input = {
            .bank_base_address = command.a0, .page_size = kLanesPageBytes
        };
        const uint32_t first = command.a1, pages = command.a2;
        for (uint32_t page = 0; page < pages; ++page) {
            if (!reader.reserve(kLanesCbIn, 1)) {
                return;
            }
            noc_async_read(
                get_noc_addr(first + page, input),
                get_write_ptr(kLanesCbIn),
                kLanesPageBytes
            );
            const uint32_t code = reader.poll(
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                reader.args().wait_limit
            );
            if (code != km::kGaveNone) {
                reader.gave_up(
                    km::kStReader, code, kWhereReadDone, 0, command.sequence
                );
                return;
            }
            cb_push_back(kLanesCbIn, 1);
        }
    });
}
