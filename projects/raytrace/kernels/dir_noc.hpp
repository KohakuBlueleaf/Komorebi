// A core's directory entry for the reader and writer: a page of the directory tensor,
// or for a TREELETS command whose a9 names a bank, of the X280 thread's plan there.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "raytrace.h"

template <typename Command>
inline uint64_t dir_addr(const Command& r, uint32_t handler, uint32_t core) {
    const uint32_t bank = handler == kHandlerTreelets ? r.a9 >> kBankShift : 0;
    if (bank != 0) {
        return get_noc_addr_from_bank_id<true>(bank - 1, r.a0 + core * kDirBytes);
    }
    const InterleavedAddrGen<true> dirs = {
        .bank_base_address = r.a0, .page_size = kDirBytes
    };
    return get_noc_addr(core, dirs);
}
