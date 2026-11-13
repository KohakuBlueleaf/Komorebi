// A core's directory entry for the reader and writer: a page of the directory tensor,
// or, when a TREELETS `policy_word` (a9) names a bank, of the X280 thread's plan.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "raytrace.h"

template <typename Command>
inline uint64_t directory_noc_address(
    const Command& command, uint32_t handler, uint32_t core
) {
    const uint32_t directory_address = command.a0, policy_word = command.a9;
    const uint32_t bank = handler == kHandlerTreelets ? policy_word >> kBankShift : 0;
    if (bank != 0) {
        return (
            get_noc_addr_from_bank_id<true>(
                bank - 1, directory_address + core * kDirBytes
            )
        );
    }
    const InterleavedAddrGen<true> directory_pages = {
        .bank_base_address = directory_address, .page_size = kDirBytes
    };
    return get_noc_addr(core, directory_pages);
}
