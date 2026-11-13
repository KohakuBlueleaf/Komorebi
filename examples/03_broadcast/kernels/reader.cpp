// Broadcast reader: the row broadcast, on NoC0. Each row's sender reads the row's A
// blocks from DRAM and multicasts them along the row (READ: every rank reads its own).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto row_args = TensorAccessorArgs<0>();
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerBroadcast && handler != kHandlerRead) {
            return;
        }
        const uint32_t row_address = command.a0;
        const uint32_t blocks = command.a3;
        const uint32_t stalled_rank = command.a4;
        const uint32_t stall_block = command.a5;
        const uint32_t data_blocks = command.a7;
        km::Broadcast broadcast(reader, command, kArgBroadcast);
        if (!broadcast.ok() || !broadcast.included()) {
            return;
        }
        const auto row_source = TensorAccessor(row_args, row_address);
        const uint32_t first_tile = (
            reader.arg(kArgIndex) * data_blocks * broadcast.pages()
        );
        if (handler == kHandlerRead) {
            read_blocks(
                reader,
                km::kStReader,
                command,
                broadcast,
                kCbRow,
                row_source,
                first_tile,
                blocks,
                data_blocks
            );
            return;
        }
        const bool stalls = reader.me() - km::first(command) == stalled_rank;
        broadcast_blocks(
            broadcast,
            row_source,
            first_tile,
            blocks,
            data_blocks,
            [&](uint32_t block) {
                if (!stalls || block != stall_block) {
                    return false;
                }
                reader.wait_stop(command);  // a test: this receiver stops taking blocks
                return true;
            }
        );
    });
}
