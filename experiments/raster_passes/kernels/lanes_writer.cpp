// Lanes probe writer: a command's output pages (the planes, then the profile) to DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "lanes.h"
#include "raster.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerTile) {
            return true;
        }
        const InterleavedAddrGen<true> output = {
            .bank_base_address = command.a3, .page_size = kLanesPageBytes
        };
        if (!writer.wait_front(kLanesCbOut, kLanesOutPages)) {
            return false;
        }
        const uint32_t source = get_read_ptr(kLanesCbOut);
        for (uint32_t page = 0; page < kLanesOutPages; ++page) {
            noc_async_write(
                source + page * kLanesPageBytes,
                get_noc_addr(command.a4 + page, output),
                kLanesPageBytes
            );
        }
        noc_async_write_barrier();
        cb_pop_front(kLanesCbOut, kLanesOutPages);
        return true;
    });
}
