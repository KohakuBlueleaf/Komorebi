// Pages across islands, reader: MOVE reads a tensor's pages with km::read_pages and
// writes them into another tensor with km::write_pages, wherever either lives.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // both tensors' (one shape)
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMove) {
            return;
        }
        const uint32_t from = command.a0, to = command.a1, step = command.a3;
        const uint32_t buffer = get_write_ptr(kCbMove);
        const km::Partition share = km::partition(
            command.a2, reader.me() - km::first(command), km::count(command)
        );
        for (uint32_t done = 0; done < share.len;) {
            const uint32_t n = share.len - done < step ? share.len - done : step;
            const uint32_t first = share.start + done;
            if (
                !km::read_pages(
                    reader, command, args, from, first, n, buffer, kAskPort
                )
            ) {
                return;
            }
            noc_async_read_barrier();
            if (
                !km::write_pages(
                    reader, command, args, to, first, n, buffer, kPutPort
                )
            ) {
                return;
            }
            noc_async_write_barrier();
            done += n;
        }
    });
}
