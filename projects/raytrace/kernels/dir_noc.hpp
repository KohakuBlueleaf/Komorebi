// A core's directory entry for the reader and writer: a page of the directory tensor,
// or, when a TREELETS `policy_word` (a9) names a bank, of the X280 thread's plan.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/km_ask.hpp"
#include "raytrace.h"

// Starts the read of this core's entry into L1 `destination`: from the thread's plan by
// bank address, or the directory tensor's page (km::read_pages, `Directories` its
// km::Tensor). False: the page call gave up (recorded).
template <typename Directories, typename Kernel>
inline bool read_directory(
    const Kernel& core,
    const km_command_t& command,
    uint32_t handler,
    uint32_t destination
) {
    const uint32_t directory_address = command.a0, policy_word = command.a9;
    const uint32_t bank = handler == kHandlerTreelets ? policy_word >> kBankShift : 0;
    if (bank != 0) {
        noc_async_read(
            get_noc_addr_from_bank_id<true>(
                bank - 1, directory_address + core.me() * kDirBytes
            ),
            destination,
            kDirBytes
        );
        return true;
    }
    return km::read_pages(
        core, command, Directories(directory_address), core.me(), 1, destination
    );
}
