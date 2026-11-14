// Pages across islands, writer: WHERE answers which island holds a page of one of
// the four tensors whose placement the session wrote into its compile arguments; PUT
// writes pages of its own island into a tensor another island holds with
// km::write_pages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_placement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // every tensor's (one shape)
    constexpr auto t0 = km::Placement<args.next_compile_time_args_offset()>();
    constexpr auto t1 = km::Placement<t0.next_compile_time_args_offset()>();
    constexpr auto t2 = km::Placement<t1.next_compile_time_args_offset()>();
    constexpr auto t3 = km::Placement<t2.next_compile_time_args_offset()>();
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerWhere) {
            const uint32_t tensor = command.a0;
            const uint32_t page = command.a1;
            writer.result(
                tensor == 0
                    ? km::island_of(t0, page)
                    : tensor == 1
                        ? km::island_of(t1, page)
                        : tensor == 2
                            ? km::island_of(t2, page)
                            : km::island_of(t3, page)
            );
        } else if (km::handler(command) == kHandlerPut) {
            const uint32_t from = command.a0, to = command.a1, step = command.a3;
            const auto source = TensorAccessor(args, from);
            const auto target = TensorAccessor(args, to);
            const uint32_t page = source.get_aligned_page_size();
            const uint32_t buffer = get_write_ptr(kCbPut);
            const km::Partition share = km::partition(
                command.a2, writer.me() - km::first(command), km::count(command)
            );
            for (uint32_t done = 0; done < share.len;) {
                const uint32_t n = share.len - done < step ? share.len - done : step;
                const uint32_t first = share.start + done;
                for (uint32_t k = 0; k < n; k++) {
                    noc_async_read_page(first + k, source, buffer + k * page);
                }
                noc_async_read_barrier();
                // the target is the second tensor of the four (another island's)
                if (
                    !km::write_pages(
                        writer, command, t1, target, to, first, n, buffer, kPutPort
                    )
                ) {
                    return false;  // gave up (recorded)
                }
                noc_async_write_barrier();
                done += n;
            }
        }
        return true;
    });
}
