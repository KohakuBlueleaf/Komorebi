// The wide-node probe's reader: for the unpacker, the node tiles from L1 into kCbNodes.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "probe.h"

void kernel_main() {
    km::Reader reader;
    const uint32_t tiles_address = reader.arg(0);
    reader.run([&](const km_command_t& command) {
        const uint32_t form = command.a0;
        const uint32_t groups = command.a1;
        const uint32_t ops = command.a2;
        const uint32_t method = command.a4;
        if (
            method != kMethodUnpack || form == kFormRay || groups != 1 ||
            ops > kTilesMax
        ) {
            return;
        }
        if (!reader.reserve(kCbNodes, ops)) {
            return;
        }
        volatile tt_l1_ptr uint32_t* destination = km::l1(get_write_ptr(kCbNodes));
        volatile tt_l1_ptr uint32_t* source = km::l1(tiles_address);
        for (uint32_t index = 0; index < ops * kTileWords; ++index) {
            destination[index] = source[index];
        }
        cb_push_back(kCbNodes, ops);
    });
}
