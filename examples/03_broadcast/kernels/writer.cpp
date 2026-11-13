// Broadcast writer: the column broadcast, on NoC1 (W blocks down each column), then the
// core's checksum tile at output page a6 first_page + rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto column_args = TensorAccessorArgs<0>();
    constexpr uint32_t output_offset = column_args.next_compile_time_args_offset();
    constexpr auto output_args = TensorAccessorArgs<output_offset>();
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerBroadcast && handler != kHandlerRead) {
            return true;
        }
        const uint32_t column_address = command.a1;
        const uint32_t output_address = command.a2;
        const uint32_t blocks = command.a3;
        const uint32_t first_page = command.a6;
        const uint32_t data_blocks = command.a7;
        km::Broadcast broadcast(writer, command, kArgBroadcast);
        if (!broadcast.ok()) {
            return false;
        }
        if (!broadcast.included()) {
            return true;
        }
        const auto column_source = TensorAccessor(column_args, column_address);
        const uint32_t first_tile = (
            writer.arg(kArgIndex) * data_blocks * broadcast.pages()
        );
        bool got_blocks = false;
        if (handler == kHandlerRead) {
            got_blocks = (
                read_blocks(
                    writer,
                    km::kStWriter,
                    command,
                    broadcast,
                    kCbCol,
                    column_source,
                    first_tile,
                    blocks,
                    data_blocks
                )
            );
        } else {
            got_blocks = broadcast_blocks(
                broadcast,
                column_source,
                first_tile,
                blocks,
                data_blocks,
                [](uint32_t) {
                    return false;
                }
            );
        }
        if (!got_blocks || !writer.wait_front(kCbOut, 1)) {
            return false;
        }
        const auto output = TensorAccessor(output_args, output_address);
        noc_async_write_page(
            first_page + writer.me() - km::first(command), output, get_read_ptr(kCbOut)
        );
        const bool sent = writer.poll(
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            writer.args().wait_limit,
            writer.args().wait_limit
        ) == km::kGaveNone;
        cb_pop_front(kCbOut, 1);
        return sent;
    });
}
