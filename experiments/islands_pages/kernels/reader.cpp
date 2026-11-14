// Pages across islands, reader: ASK reads pages another island holds with
// km::read_pages and writes them into a tensor of its own island.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // every tensor's (one shape)
    using Far = km::Placement<args.next_compile_time_args_offset()>;
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerAsk) {
            return;
        }
        const uint32_t from = command.a0, to = command.a1, step = command.a3;
        const auto source = TensorAccessor(args, from);
        const auto target = TensorAccessor(args, to);
        const uint32_t page = source.get_aligned_page_size();
        const uint32_t buffer = get_write_ptr(kCbAsk);
        const km::Partition share = km::partition(
            command.a2, reader.me() - km::first(command), km::count(command)
        );
        for (uint32_t done = 0; done < share.len;) {
            const uint32_t n = share.len - done < step ? share.len - done : step;
            const uint32_t first = share.start + done;
            if (
                !km::read_pages(
                    reader, command, Far(), source, from, first, n, buffer, kAskPort
                )
            ) {
                return;
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < n; k++) {
                noc_async_write_page(first + k, target, buffer + k * page);
            }
            noc_async_write_barrier();
            done += n;
        }
    });
}
