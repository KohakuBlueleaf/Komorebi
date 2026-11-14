// READ on one mover: chunks `parity`, `parity` + 2, .. of this rank's partition.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ceilings.h"
#include "komorebi/device/km_worker.hpp"

template <typename Core, typename Args>
void read_chunks(
    const Core& core,
    const km_command_t& command,
    const Args& args,
    uint32_t parity,
    uint32_t to
) {
    const auto tensor = TensorAccessor(args, command.a0);
    const uint32_t chunk = command.a2 < kChunkPages ? command.a2 : kChunkPages;
    const km::Partition part = (
        km::partition(command.a1, core.me() - km::first(command), km::count(command))
    );
    for (uint32_t done = parity * chunk; done < part.len; done += 2 * chunk) {
        const uint32_t pages = part.len - done < chunk ? part.len - done : chunk;
        for (uint32_t page = 0; page < pages; ++page) {
            noc_async_read_page(
                part.start + done + page, tensor, to + page * kTileBytes
            );
        }
        noc_async_read_barrier();
    }
}
