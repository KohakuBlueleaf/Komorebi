// The passes' DRAM addresses for the reader and writer: an interleaved tensor, or a
// page of the X280 thread's regions in one DRAM bank (see setup.h, a8).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace su {

    // Byte `offset` of page `page` (of `bytes`) at `base`: with bank 0 an interleaved
    // tensor's page, otherwise bank (bank - 1)'s address base + page x bytes + offset.
    inline uint64_t addr(
        uint32_t bank, uint32_t base, uint32_t page, uint32_t bytes, uint32_t offset = 0
    ) {
        if (bank != 0) {
            return (
                get_noc_addr_from_bank_id<true>(bank - 1, base + page * bytes + offset)
            );
        }
        const InterleavedAddrGen<true> g = {
            .bank_base_address = base, .page_size = bytes
        };
        return get_noc_addr(page, g, offset);
    }

}  // namespace su
