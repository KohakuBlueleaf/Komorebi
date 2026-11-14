// Pages across islands, writer: WHERE answers which island holds a page of one of
// the four tensors whose placement the session wrote into its compile arguments.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_placement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

void kernel_main() {
    constexpr auto t0 = km::Placement<0>();
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
        }
        return true;
    });
}
