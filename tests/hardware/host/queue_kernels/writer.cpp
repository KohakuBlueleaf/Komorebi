// Queue order case, grid side, the tensor at a0: PUT's leader writes a2 into page a1;
// SUM's counts the pages below a1 that hold a2; BUSY's reads page 0 a1 times.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kHandlerPut = 1;
constexpr uint32_t kHandlerSum = 2;
constexpr uint32_t kHandlerBusy = 3;
constexpr uint32_t kCbScratch = 16;

void kernel_main() {
    constexpr auto pages_args = TensorAccessorArgs<0>();
    km::Writer writer;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    volatile uint32_t* word = reinterpret_cast<volatile uint32_t*>(scratch);
    // NoC barriers polled with the wait limit; false (no report) if it passed.
    auto reads_done = [&] {
        const bool ok = writer.poll(
            [&] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
        invalidate_l1_cache();
        return ok;
    };
    auto writes_done = [&] {
        return writer.poll(
            [&] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };
    writer.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (
            handler < kHandlerPut || handler > kHandlerBusy ||
            writer.me() != km::first(command)
        ) {
            return true;
        }
        const uint32_t tensor_argument = command.a0;
        const uint32_t page_or_count = command
                                           .a1;  // PUT: the page; SUM, BUSY: the count
        const uint32_t value = command.a2;
        const auto pages = TensorAccessor(pages_args, tensor_argument);
        if (handler == kHandlerPut) {
            word[0] = value;
            noc_async_write_page(page_or_count, pages, scratch);
            return writes_done();
        }
        uint32_t same = 0;
        for (uint32_t page = 0; page < page_or_count; page++) {
            noc_async_read_page(handler == kHandlerSum ? page : 0, pages, scratch);
            if (!reads_done()) {
                return false;
            }
            same += word[0] == value;
        }
        writer.result(same);
        return true;
    });
}
